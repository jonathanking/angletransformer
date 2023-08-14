import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from openfold.np import residue_constants
from openfold.utils.tensor_utils import masked_mean
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

# from openfold.utils.loss import supervised_chi_loss
import pytorch_lightning as pl
import torch
import torch.nn as nn
# from openfold.model.jk_sidechain_model import AngleTransformer
from openfold.config import config
from openfold.model.primitives import Linear
from openfold.utils.loss import supervised_chi_loss
from sidechainnet.examples.transformer import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from data import ATFileDataset, collate_fn
from model import AngleTransformer


class ATModuleLit(pl.LightningModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
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
        **kwargs
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
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Optimizer
        self.lr = kwargs.get("lr", 1e-3)

    def forward(self, s, s_initial):
        return self.at(s, s_initial)

    def training_step(self, batch, batch_idx):
        s, s_initial = batch["s"][:, -1, ...].squeeze(1), batch["s_initial"].squeeze(1)
        unnorm_ang, ang = self(s, s_initial)
        loss = self.loss(
            angles_sin_cos=ang,
            unnormalized_angles_sin_cos=unnorm_ang,
            aatype=batch["aatype"],
            seq_mask=batch["seq_mask"],
            chi_mask=batch["chi_mask"],
            chi_angles_sin_cos=batch["chi_angles_sin_cos"],
            chi_weight=config.loss.supervised_chi.chi_weight,
            angle_norm_weight=config.loss.supervised_chi.angle_norm_weight,
            eps=1e-6,
        )
        self.log(
            "train/loss",
            loss["loss"],
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=True,
        )
        self.log(
            "train/sq_chi_loss",
            loss["sq_chi_loss"],
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=True,
        )
        self.log(
            "train/angle_norm_loss",
            loss["angle_norm_loss"],
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        s, s_initial = batch["s"][:, -1, ...].squeeze(1), batch["s_initial"].squeeze(1)
        unnorm_ang, ang = self(s, s_initial)
        loss = self.loss(
            angles_sin_cos=ang,
            unnormalized_angles_sin_cos=unnorm_ang,
            aatype=batch["aatype"],
            seq_mask=batch["seq_mask"],
            chi_mask=batch["chi_mask"],
            chi_angles_sin_cos=batch["chi_angles_sin_cos"],
            chi_weight=config.loss.supervised_chi.chi_weight,
            angle_norm_weight=config.loss.supervised_chi.angle_norm_weight,
            eps=1e-6,
        )
        self.log(
            "val/loss",
            loss["loss"],
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val/sq_chi_loss",
            loss["sq_chi_loss"],
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "val/angle_norm_loss",
            loss["angle_norm_loss"],
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=False,
        )

        return loss

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
        if self.hparams.opt_lr_scheduling == "noam":
            lr = 0
            betas = (0.9, 0.98)
            eps = 1e-9
        else:
            lr = self.hparams.opt_lr
            betas = (0.9, 0.999)
            eps = 1e-8

        # Prepare optimizer
        if self.hparams.opt_name == "adam":
            opt = torch.optim.Adam(
                filter(lambda x: x.requires_grad, self.parameters()),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=self.hparams.opt_weight_decay,
            )
        elif self.hparams.opt_name == "radam":
            import radam

            opt = radam.RAdam(
                filter(lambda x: x.requires_grad, self.parameters()),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=self.hparams.opt_weight_decay,
            )
        elif self.hparams.opt_name == "adamw":
            opt = torch.optim.AdamW(
                filter(lambda x: x.requires_grad, self.parameters()),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=self.hparams.opt_weight_decay,
            )
        elif self.hparams.opt_name == "sgd":
            opt = torch.optim.SGD(
                filter(lambda x: x.requires_grad, self.parameters()),
                lr=lr,
                weight_decay=self.hparams.opt_weight_decay,
                momentum=0.9,
            )

        # Prepare scheduler
        if self.hparams.opt_lr_scheduling == "noam" and self.hparams.opt_name in [
            "adam",
            "adamw",
        ]:
            opt = NoamOpt(
                model_size=self.hparams.d_in,
                warmup=self.hparams.opt_n_warmup_steps,
                optimizer=opt,
                factor=self.hparams.opt_noam_lr_factor,
            )
            sch = None
        elif self.hparams.opt_lr_scheduling == "plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                patience=self.hparams.opt_patience,
                verbose=True,
                threshold=self.hparams.opt_min_delta,
                mode="min"
                if "acc" not in self.hparams.opt_lr_scheduling_metric
                else "max",
            )
        else:
            sch = None

        d = {"optimizer": opt}
        if sch is not None:
            d["lr_scheduler"] = {
                "scheduler": sch,
                "interval": "epoch",
                "frequency": 1,
                "monitor": self.hparams.opt_lr_scheduling_metric,
                "strict": True,
                "name": None,
            }

        return d

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            ATFileDataset(self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            ATFileDataset(self.val_dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


def main(args):
    # Load data

    # Create model
    model = ATModuleLit(
        train_dataset=args.train_data, val_dataset=args.val_data, **vars(args)
    )

    # Create checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        filename="at-{epoch:02d}",
        save_top_k=1,
        mode="min",
    )

    # Early stopping callback,
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val/loss",
        patience=20,
        mode="min",
        verbose=True,
    )

    # Create wandb logger
    wandb_logger = pl.loggers.WandbLogger(
        name=args.experiment_name,
        save_dir=args.output_dir,
        project="angletransformer_solo",
        notes=args.wandb_notes,
        tags=[tag for tag in args.wandb_tags.split(",") if tag]
        if args.wandb_tags
        else None,
        group=args.experiment_name if args.experiment_name else "default_group",
        **{"entity": "koes-group"},
    )
    # MOD-JK: save config to wandb, log gradients/params
    wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

    # Create trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, early_stopping_callback],
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

    # Other arguments for model sweep
    lr_schedule_group = parser.add_mutually_exclusive_group(required=True)
    lr_schedule_group.add_argument(
        "--standard_lr", type=float, default=1e-3, help="Learning rate."
    )
    lr_schedule_group.add_argument("--use_")
    # Add an argument that's mutally exclusive with the above
    parser.add_argument(
        "--use_noam", action="store_true", help="Use Noam learning rate schedule."
    )
    parser.add_argument(
        "--noam_warmup", type=int, default=4000, help="Noam warmup steps."
    )
    parser.add_argument("--noam_factor", type=float, default=1.0, help="Noam factor.")
    parser.add_argument("--noam_scale", type=float, default=1.0, help="Noam scale.")
    parser.add_argument("--noam_step", type=int, default=1, help="Noam step.")

    parser.add_argument("--use_scheduler", action="store_true", help="Use scheduler.")
    parser.add_argument(
        "--scheduler_factor", type=float, default=0.1, help="Scheduler factor."
    )
    parser.add_argument(
        "--scheduler_patience", type=int, default=10, help="Scheduler patience."
    )

    # Add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.experiment_name)

    # If output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
