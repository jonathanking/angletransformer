import argparse
import os
import sys

import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, "/net/pulsar/home/koes/jok120/openfold/")

from openfold.config import config
from openfold.utils.loss import supervised_chi_loss
from sidechainnet.examples.optim import NoamOpt

from data import ATFileDataset, collate_fn
from model import AngleTransformer


class ATModuleLit(pl.LightningModule):
    def __init__(
        self,
        train_dataset_dir,
        val_dataset_dir,
        c_s=384,
        c_hidden=256,
        no_blocks=2,
        no_angles=config.model.structure_module.no_angles,  # 7
        epsilon=config.globals.eps,
        dropout=0.1,
        d_ff=2048,
        no_heads=4,
        activation="relu",
        batch_size=1,
        num_workers=0,
        **kwargs,
    ):
        super().__init__()
        self.at = AngleTransformer(
            c_s=c_s,
            c_hidden=c_hidden,
            no_blocks=no_blocks,
            no_angles=no_angles,
            epsilon=epsilon,
            dropout=dropout,
            d_ff=d_ff,
            no_heads=no_heads,
            activation=activation,
        )
        self.loss = supervised_chi_loss
        self.train_dataset_dir = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Optimizer
        self.opt_lr = kwargs.get("opt_lr")  # , 1e-3)
        self.opt_lr_scheduling = kwargs.get("opt_lr_scheduling")  # , "none")
        self.opt_name = kwargs.get("opt_name")  # , "adamw")
        self.opt_n_warmup_steps = kwargs.get("opt_n_warmup_steps")  # , 1000)
        self.opt_noam_lr_factor = kwargs.get("opt_noam_lr_factor")  # , 1)
        self.opt_weight_decay = kwargs.get("opt_weight_decay")  # , None)
        self.opt_patience = kwargs.get("opt_patience")  # , None)
        self.opt_min_delta = kwargs.get("opt_min_delta")  # , 1e-4)
        self.opt_lr_scheduling_metric = kwargs.get(
            "opt_lr_scheduling_metric"
        )  # , "val/loss")

        self.chi_weight = kwargs.get("chi_weight")  # , 1.0)

    def forward(self, s, s_initial):
        return self.at(s, s_initial)

    def _standard_train_val_step(self, batch, batch_idx):
        s, s_initial = batch["s"][:, -1, ...].squeeze(1), batch["s_initial"].squeeze(1)
        unnorm_ang, ang = self(s, s_initial)
        loss = self.loss(
            angles_sin_cos=ang,
            unnormalized_angles_sin_cos=unnorm_ang,
            aatype=batch["aatype"],
            seq_mask=batch["seq_mask"],
            chi_mask=batch["chi_mask"],
            chi_angles_sin_cos=batch["chi_angles_sin_cos"],
            chi_weight=self.chi_weight,
            angle_norm_weight=config.loss.supervised_chi.angle_norm_weight,
            eps=1e-6,
        )
        mode = "train" if self.training else "val"
        on_step = mode == "train"
        self.log(
            f"{mode}/loss",
            loss["loss"],
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=on_step,
        )
        self.log(
            f"{mode}/sq_chi_loss",
            loss["sq_chi_loss"],
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=on_step,
        )
        self.log(
            f"{mode}/angle_norm_loss",
            loss["angle_norm_loss"],
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=on_step,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._standard_train_val_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._standard_train_val_step(batch, batch_idx)

    def configure_optimizers(self):
        """Prepare optimizer and schedulers.

        Args:
            optimizer (str): Name of optimizer ('adam', 'sgd')
            learning_rate (float): Learning rate for optimizer.
            weight_decay (bool, optional): Use optimizer weight decay. Defaults to True.

        Returns:
            dict: Pytorch Lightning dictionary with keys "optimizer" and "lr_scheduler".
        """
        # Setup default optimizer construction values
        if self.opt_lr_scheduling == "noam":
            lr = 0
            betas = (0.9, 0.98)
            eps = 1e-9
        else:
            lr = self.opt_lr
            betas = (0.9, 0.999)
            eps = 1e-8

        # Prepare optimizer
        if self.opt_name == "adam":
            opt = torch.optim.Adam(
                filter(lambda x: x.requires_grad, self.parameters()),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=self.opt_weight_decay,
            )
        elif self.opt_name == "adamw":
            opt = torch.optim.AdamW(
                filter(lambda x: x.requires_grad, self.parameters()),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=self.opt_weight_decay,
            )
        elif self.opt_name == "sgd":
            opt = torch.optim.SGD(
                filter(lambda x: x.requires_grad, self.parameters()),
                lr=lr,
                weight_decay=self.opt_weight_decay,
                momentum=0.9,
            )

        # Prepare scheduler
        if self.opt_lr_scheduling == "noam" and self.opt_name in [
            "adam",
            "adamw",
        ]:
            opt = NoamOpt(
                model_size=self.at.c_hidden,
                warmup=self.opt_n_warmup_steps,
                optimizer=opt,
                factor=self.opt_noam_lr_factor,
            )
            sch = None
        elif self.opt_lr_scheduling == "plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                patience=self.opt_patience,
                verbose=True,
                threshold=self.opt_min_delta,
                mode="min" if "acc" not in self.opt_lr_scheduling_metric else "max",
            )
        else:
            sch = None

        d = {"optimizer": opt}
        if sch is not None:
            d["lr_scheduler"] = {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.opt_lr_scheduling_metric,
                "strict": True,
                "name": None,
            }

        return d

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            ATFileDataset(self.train_dataset_dir),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            ATFileDataset(self.val_dataset_dir),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


def main(args):
    # Create model
    model = ATModuleLit(
        train_dataset_dir=args.train_data, val_dataset_dir=args.val_data, **vars(args)
    )

    my_callbacks = []

    # Create checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        filename="at-{epoch:02d}",
        save_top_k=1,
        mode="min",
    )
    my_callbacks.append(checkpoint_callback)

    # Early stopping callback,
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val/loss",
        patience=20,
        mode="min",
        verbose=True,
    )
    my_callbacks.append(early_stopping_callback)

    # Create wandb logger
    wandb_logger = pl.loggers.WandbLogger(
        name=args.experiment_name,
        save_dir=args.output_dir,
        project="angletransformer_solo01",
        notes=args.wandb_notes,
        tags=[tag for tag in args.wandb_tags.split(",") if tag]
        if args.wandb_tags
        else None,
        group=args.experiment_name if args.experiment_name else "default_group",
        **{"entity": "koes-group"},
    )
    # MOD-JK: save config to wandb, log gradients/params
    try:
        wandb_logger.experiment.config.update(vars(args), allow_val_change=True)
    except AttributeError as e:
        pass

    my_callbacks.append(callbacks.LearningRateMonitor(logging_interval="step"))

    # Create trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=my_callbacks,
        logger=wandb_logger,
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu",
        devices="auto",
    )

    # Train
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to pickle files of training data.",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to pickle files of validation data.",
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of experiment."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for dataloader."
    )

    parser.add_argument("--wandb_notes", type=str, default="", help="Notes for wandb.")
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="Comma-separated list of tags for wandb.",
    )

    # Implement args for c_hidden, no_blocks, no_angles, epsilon, dropout=0.1, d_ff=2048, no_heads=4, activation='relu'
    parser.add_argument(
        "--c_hidden", type=int, default=256, help="Hidden dimension of transformer."
    )
    parser.add_argument(
        "--no_blocks", type=int, default=2, help="Number of transformer blocks."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout for transformer."
    )
    parser.add_argument(
        "--d_ff", type=int, default=2048, help="Feedforward dimension for transformer."
    )
    parser.add_argument(
        "--no_heads", type=int, default=4, help="Number of heads for transformer."
    )
    parser.add_argument(
        "--activation", type=str, default="relu", help="Activation for transformer."
    )

    # Optimizer args
    opt_args = parser.add_argument_group("Optimizer args")
    opt_args.add_argument(
        "--opt_name",
        "-opt",
        type=str,
        choices=["adam", "sgd", "adamw"],
        default="adamw",
        help="Training optimizer.",
    )
    opt_args.add_argument("--opt_lr", "-lr", type=float, default=1e-4)
    opt_args.add_argument(
        "--opt_lr_scheduling",
        type=str,
        choices=["noam", "plateau", "none"],
        default="none",
        help="noam: Use LR as described in Transformer paper, plateau:"
        " Decrease learning rate after Validation loss plateaus.",
    )
    opt_args.add_argument(
        "--opt_patience", type=int, default=10, help="Patience for LR routines."
    )
    opt_args.add_argument(
        "--opt_min_delta",
        type=float,
        default=0.005,
        help="Threshold for considering improvements during training/lr" " scheduling.",
    )
    opt_args.add_argument(
        "--opt_weight_decay",
        type=float,
        default=0.0,
        help="Applies weight decay to model weights.",
    )
    opt_args.add_argument(
        "--opt_n_warmup_steps",
        "-nws",
        type=int,
        default=10_000,
        help="Number of warmup train steps when using lr-scheduling.",
    )
    opt_args.add_argument(
        "--opt_lr_scheduling_metric",
        type=str,
        default="val/sq_chi_loss",
        help="Metric to use for early stopping, chkpts, etc. Choose "
        "validation loss or accuracy.",
    )
    opt_args.add_argument(
        "--opt_noam_lr_factor", type=float, default=1.0, help="Scale for Noam Opt."
    )
    opt_args.add_argument(
        "--chi_weight",
        type=float,
        default=config.loss.supervised_chi.chi_weight,
        help="Scale for sq_chi_loss weight vs angle_norm.",
    )

    # Add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.experiment_name)

    # If output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
