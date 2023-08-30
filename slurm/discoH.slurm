#!/bin/bash
#SBATCH --job-name=discoH
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=28-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="out/%A_%6a.out"
#SBATCH --ntasks-per-node=12
#SBATCH --exclude=g019

# Setup
cd ~/openfold
source scripts/activate_conda_env.sh
module load cuda/11.5

cd ~/angletransformer

python train.py \
    --activation=gelu \
    --c_hidden=4096 \
    --chi_weight=0.05 \
    --d_ff=256 \
    --dropout=0.15 \
    --is_sweep=False \
    --no_blocks=4 \
    --no_heads=2 \
    --opt_lr=0.0001 \
    --opt_lr_scheduling=plateau \
    --opt_lr_scheduling_metric=val/sq_chi_loss \
    --opt_n_warmup_steps=10000 \
    --opt_name=adam \
    --opt_noam_lr_factor=2 \
    --replace_sampler_ddp=False \
    --train_data=data/train/ \
    --val_data=data/val/ \
    --output_dir=out/experiments \
    --num_workers=6 \
    --wandb_tags="sweep" \
    --batch_size=1 \
    --val_check_interval=2500 \
    --experiment_name="sunny-disco-172-h"

