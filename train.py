import pickle
import pandas as pd
import numpy as np
import torch
import os
import glob
import argparse

import torch.multiprocessing
from tqdm import tqdm
from openfold.np import residue_constants

from openfold.utils.tensor_utils import masked_mean
torch.multiprocessing.set_sharing_strategy('file_system')

# from openfold.model.jk_sidechain_model import AngleTransformer
from openfold.config import config
# from openfold.utils.loss import supervised_chi_loss
import pytorch_lightning as pl

def supervised_chi_loss(
    angles_sin_cos: torch.Tensor,
    unnormalized_angles_sin_cos: torch.Tensor,
    aatype: torch.Tensor,
    seq_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    chi_angles_sin_cos: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    """
        Implements Algorithm 27 (torsionAngleLoss)

        Args:
            angles_sin_cos:
                [*, N, 7, 2] predicted angles
            unnormalized_angles_sin_cos:
                The same angles, but unnormalized
            aatype:
                [*, N] residue indices
            seq_mask:
                [*, N] sequence mask
            chi_mask:
                [*, N, 7] angle mask
            chi_angles_sin_cos:
                [*, N, 7, 2] ground truth angles
            chi_weight:
                Weight for the angle component of the loss
            angle_norm_weight:
                Weight for the normalization component of the loss
        Returns:
            [*] loss tensor
    """
    pred_angles = angles_sin_cos[..., 3:, :]  # [8, 1, 256, 4, 2]
    residue_type_one_hot = torch.nn.functional.one_hot(
        aatype,
        residue_constants.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->ik",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(residue_constants.chi_pi_periodic),
    )

    true_chi = chi_angles_sin_cos[None]  # [1, 1, 256, 4, 2]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum(
        (true_chi_shifted - pred_angles) ** 2, dim=-1
    )
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)
    
    # The ol' switcheroo
    sq_chi_error = sq_chi_error.permute(
        *range(len(sq_chi_error.shape))[1:-2], 0, -2, -1
    )

    sq_chi_loss = masked_mean(
        chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3)
    )

    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(
        torch.sum(unnormalized_angles_sin_cos ** 2, dim=-1) + eps
    )
    # angle_norm = angle_norm.unsqueeze(0)
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(
        *range(len(norm_error.shape))[1:-2], 0, -2, -1
    )
    angle_norm_loss = masked_mean(
        seq_mask[..., None, :, None], norm_error, dim=(-1, -2, -3)
    )

    loss = loss + angle_norm_weight * angle_norm_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    # Compute the MAE so we know exactly how good the angle prediction is in Radians
    # print(pred_angles.shape)
    # pred = torch.transpose(pred_angles.clone(), 0, 1)  # [1, 8, 256, 4, 2]
    # pred = pred[:, -1, :, :, :]  # [1, 1, 256, 4, 2]
    # pred = pred.reshape(pred.shape[0], pred.shape[-3], pred.shape[-2], pred.shape[-1])  # [1, 256, 4, 2]
    # true_chi2 = chi_angles_sin_cos.clone()  # [1, 256, 4, 2]
    # true_chi2 = inverse_trig_transform(true_chi2, 4)  # [1, 256, 4]
    # pred = inverse_trig_transform(pred, 4)  # [1, 256, 4]
    # true_chi2 = true_chi2.masked_fill_(~chi_mask.bool(), torch.nan)
    # pred = pred.masked_fill_(~chi_mask.bool(), torch.nan)
    # mae = angle_mae(true_chi2, pred)

    loss_dict = {"loss": loss, "sq_chi_loss": sq_chi_loss, "angle_norm_loss": angle_norm_loss}

    return loss_dict



import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from openfold.model.primitives import Linear

from sidechainnet.examples.transformer import PositionalEncoding


class AngleTransformer(nn.Module):
    """A shallow Transformer Encoder to predict torsion angles from AF2 seq embeding."""
    def __init__(self,
                 c_s,
                 c_hidden,
                 no_blocks,
                 no_angles,
                 epsilon,
                 dropout=0.1,
                 d_ff=2048,
                 no_heads=4,
                 activation='relu'):
        super().__init__()
        self.eps = epsilon

        self.linear_initial = Linear(c_s, c_hidden)
        self.linear_in = Linear(c_s, c_hidden)
        self.relu = nn.ReLU()

        self.pos_encoder = PositionalEncoding(c_hidden)
        encoder_layers = TransformerEncoderLayer(c_hidden,
                                                 nhead=no_heads,
                                                 dim_feedforward=d_ff,
                                                 dropout=dropout,
                                                 activation=activation,
                                                 batch_first=True,
                                                 norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers=no_blocks)

        self.linear_out = nn.Linear(c_hidden, no_angles * 2)

    def forward(self, s, s_initial):
        # [*, C_hidden], eg [1, 256 (L), 384  (c_s)]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        # Transformer in lieu of ResNet
        s = self.pos_encoder(s)
        s = self.transformer_encoder(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            ))
        s = s / norm_denom

        return unnormalized_s, s  # Batch x Length x 7 x 2



def load_pkl(fn):
    with open(fn, "rb") as f:
        _d = pickle.load(f)
    return _d


# def load_multiple_pickles(pattern):
#     files = glob.glob(pattern)
#     files.sort()
#     all_data = (load_pkl(fn) for fn in files)
#     updated_data = {}
#     starting_idx = 0
#     print("Loading data...")
#     for fn, d in tqdm(zip(files, all_data)):
#         n = d["current_datapt_number"]
#         del d["current_datapt_number"]
#         # Add starting index to all keys in d
#         d = {k + starting_idx: v for k, v in d.items()}
#         starting_idx += n
#         updated_data.update(d)


#     return updated_data


def load_multiple_pickles(pattern):
    files = glob.glob(pattern)
    files.sort()
    all_data = (load_pkl(fn) for fn in files)
    updated_data = {}
    cur_idx = 0
    print("Loading data...")
    for fn, d in tqdm(zip(files, all_data), total=len(files)):
        updated_data.update({cur_idx: d})
    return updated_data


class ATDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict

    def __len__(self):
        return len(self.dataset_dict)

    def __getitem__(self, idx):
        return self.dataset_dict[idx]
    

class ATFileDataset(torch.utils.data.Dataset):
    """Mimic ATDataset but keep files separate."""
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.files = glob.glob(os.path.join(dataset_dir, "*.pkl"))
        self.files.sort()

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        return load_pkl(self.files[idx])


def collate_fn(batch):
    if len(batch) == 1:
        d = {
            k: batch[0][k][0].float()
            for k in batch[0].keys() if k != 'name'
        }
        d['name'] = [batch[0]['name']]
        d['aatype'] = batch[0]['aatype'].long()
        return d
    else:
        # Needs work for padding to work
        max_len = max([b['s'][0].shape[-2] for b in batch])
        d = {}
        for prot in batch:
            for k, v in prot.items():
                if k not in d:
                    d[k] = []
                v = v[0]
                if k == 's_initial':
                    len_diff = max_len - v.shape[-2]
                    new_value = torch.cat([v, torch.zeros(v.shape[0], len_diff, v.shape[-1]).float()], dim=-2)
                    d[k].append(new_value)
                elif k != 'name':
                    try:
                        len_diff = max_len - v.shape[-2]
                        new_value = torch.cat([v, torch.zeros(v.shape[0], v.shape[1], len_diff, v.shape[-1]).float()], dim=-2)
                        d[k].append(new_value)
                    except Exception as e:
                        print(e)
                        print(k)
                        print(v.shape)
                        print(max_len)
                        print(len_diff)
                        raise e
                else:
                    d[k].append(prot[k][0])
        
        d = {
            k: torch.stack(d[k]).float()
            for k in d.keys() if k != 'name'
        }

        d['name'] = [b['name'] for b in batch]
        return d


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
            activation='relu',
            batch_size=1,
            num_workers=0,
            **kwargs):
        super().__init__()
        self.at = AngleTransformer(c_s=c_s,
                                   c_hidden=c_hidden,
                                   no_blocks=no_blocks,
                                   no_angles=no_angles,
                                   epsilon=epsilon,
                                   dropout=dropout,
                                   d_ff=d_ff,
                                   no_heads=no_heads,
                                   activation=activation)
        self.loss = supervised_chi_loss
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Optimizer
        self.lr = kwargs.get('lr', 1e-3)

    def forward(self, s, s_initial):
        return self.at(s, s_initial)

    def training_step(self, batch, batch_idx):
        s, s_initial = batch['s'][:, -1, ...].squeeze(1), batch['s_initial'].squeeze(1)
        unnorm_ang, ang = self(s, s_initial)
        loss = self.loss(angles_sin_cos=ang,
                         unnormalized_angles_sin_cos=unnorm_ang,
                         aatype=batch['aatype'],
                         seq_mask=batch['seq_mask'],
                         chi_mask=batch['chi_mask'],
                         chi_angles_sin_cos=batch['chi_angles_sin_cos'],
                         chi_weight=config.loss.supervised_chi.chi_weight,
                         angle_norm_weight=config.loss.supervised_chi.angle_norm_weight,
                         eps=1e-6)
        self.log('train/loss', loss['loss'], batch_size=self.batch_size, on_epoch=True, on_step=True)
        self.log('train/sq_chi_loss', loss['sq_chi_loss'], batch_size=self.batch_size, on_epoch=True, on_step=True)
        self.log('train/angle_norm_loss', loss['angle_norm_loss'], batch_size=self.batch_size, on_epoch=True, on_step=True)


        return loss
    
    def validation_step(self, batch, batch_idx):
        s, s_initial = batch['s'][:, -1, ...].squeeze(1), batch['s_initial'].squeeze(1)
        unnorm_ang, ang = self(s, s_initial)
        loss = self.loss(angles_sin_cos=ang,
                         unnormalized_angles_sin_cos=unnorm_ang,
                         aatype=batch['aatype'],
                         seq_mask=batch['seq_mask'],
                         chi_mask=batch['chi_mask'],
                         chi_angles_sin_cos=batch['chi_angles_sin_cos'],
                         chi_weight=config.loss.supervised_chi.chi_weight,
                         angle_norm_weight=config.loss.supervised_chi.angle_norm_weight,
                         eps=1e-6)
        self.log('val/loss', loss['loss'], batch_size=self.batch_size, on_epoch=True, on_step=False)
        self.log('val/sq_chi_loss', loss['sq_chi_loss'], batch_size=self.batch_size, on_epoch=True, on_step=False)
        self.log('val/angle_norm_loss', loss['angle_norm_loss'], batch_size=self.batch_size, on_epoch=True, on_step=False)

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
            opt = torch.optim.Adam(filter(lambda x: x.requires_grad, self.parameters()),
                                   lr=lr,
                                   betas=betas,
                                   eps=eps,
                                   weight_decay=self.hparams.opt_weight_decay)
        elif self.hparams.opt_name == "radam":
            import radam
            opt = radam.RAdam(filter(lambda x: x.requires_grad, self.parameters()),
                              lr=lr,
                              betas=betas,
                              eps=eps,
                              weight_decay=self.hparams.opt_weight_decay)
        elif self.hparams.opt_name == "adamw":
            opt = torch.optim.AdamW(filter(lambda x: x.requires_grad, self.parameters()),
                                    lr=lr,
                                    betas=betas,
                                    eps=eps,
                                    weight_decay=self.hparams.opt_weight_decay)
        elif self.hparams.opt_name == "sgd":
            opt = torch.optim.SGD(filter(lambda x: x.requires_grad, self.parameters()),
                                  lr=lr,
                                  weight_decay=self.hparams.opt_weight_decay,
                                  momentum=0.9)

        # Prepare scheduler
        if (self.hparams.opt_lr_scheduling == "noam" and
                self.hparams.opt_name in ['adam', 'adamw']):
            opt = NoamOpt(model_size=self.hparams.d_in,
                          warmup=self.hparams.opt_n_warmup_steps,
                          optimizer=opt,
                          factor=self.hparams.opt_noam_lr_factor)
            sch = None
        elif self.hparams.opt_lr_scheduling == 'plateau':
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                patience=self.hparams.opt_patience,
                verbose=True,
                threshold=self.hparams.opt_min_delta,
                mode='min'
                if 'acc' not in self.hparams.opt_lr_scheduling_metric else 'max')
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
        return torch.utils.data.DataLoader(ATFileDataset(self.train_dataset),
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           collate_fn=collate_fn)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(ATFileDataset(self.val_dataset),
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           collate_fn=collate_fn)


def main(args):
    # Load data
    # train_data = load_multiple_pickles(args.train_data)
    # val_data = load_multiple_pickles(args.val_data)

    # Create model
    model = ATModuleLit(train_dataset=args.train_data,
                        val_dataset=args.val_data,
                        **vars(args))

    # Create checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val/loss',
        filename='at-{epoch:02d}',
        save_top_k=1,
        mode='min',
    )

    # Early stopping callback, 
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val/loss',
        patience=20,
        mode='min',
        verbose=True,
    )



    # Create wandb logger
    wandb_logger = pl.loggers.WandbLogger(name=args.experiment_name,
            save_dir=args.output_dir,
            project="angletransformer_solo",
            notes=args.wandb_notes,
            tags=[tag for tag in args.wandb_tags.split(",") if tag] if args.wandb_tags else None,
            group=args.experiment_name if args.experiment_name else "default_group",
            **{"entity": "koes-group"},
        )
        # MOD-JK: save config to wandb, log gradients/params
    wandb_logger.experiment.config.update(vars(args), allow_val_change=True)

    # Create trainer
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, early_stopping_callback], logger=wandb_logger, strategy="ddp_find_unused_parameters_false", accelerator="gpu", devices="auto")

    # Train
    trainer.fit(model)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to pickle files of training data.")
    parser.add_argument("--val_data", type=str, required=True, help="Path to pickle files of validation data.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of experiment.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader.")
    
    parser.add_argument("--wandb_notes", type=str, default="", help="Notes for wandb.")
    parser.add_argument("--wandb_tags", type=str, default="", help="Comma-separated list of tags for wandb.")

    # Implement args for c_hidden, no_blocks, no_angles, epsilon, dropout=0.1, d_ff=2048, no_heads=4, activation='relu'
    parser.add_argument("--c_hidden", type=int, default=256, help="Hidden dimension of transformer.")
    parser.add_argument("--no_blocks", type=int, default=2, help="Number of transformer blocks.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for transformer.")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward dimension for transformer.")
    parser.add_argument("--no_heads", type=int, default=4, help="Number of heads for transformer.")
    parser.add_argument("--activation", type=str, default="relu", help="Activation for transformer.")

    # Other arguments for model sweep
    lr_schedule_group = parser.add_mutually_exclusive_group(required=True)
    lr_schedule_group.add_argument("--standard_lr", type=float, default=1e-3, help="Learning rate.")
    lr_schedule_group.add_argument("--use_")
    # Add an argument that's mutally exclusive with the above
    parser.add_argument("--use_noam", action="store_true", help="Use Noam learning rate schedule.")
    parser.add_argument("--noam_warmup", type=int, default=4000, help="Noam warmup steps.")
    parser.add_argument("--noam_factor", type=float, default=1.0, help="Noam factor.")
    parser.add_argument("--noam_scale", type=float, default=1.0, help="Noam scale.")
    parser.add_argument("--noam_step", type=int, default=1, help="Noam step.")

    parser.add_argument("--use_scheduler", action="store_true", help="Use scheduler.")
    parser.add_argument("--scheduler_factor", type=float, default=0.1, help="Scheduler factor.")
    parser.add_argument("--scheduler_patience", type=int, default=10, help="Scheduler patience.")

    # Add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.experiment_name)

    # If output_dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)