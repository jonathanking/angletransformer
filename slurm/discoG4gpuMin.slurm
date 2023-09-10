#!/bin/bash
#SBATCH --job-name=discoG
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=6-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="out/%A_%6a.out"
#SBATCH --ntasks-per-node=12
#SBATCH --exclude=g019

# Setup
cd ~/openfold
source scripts/activate_conda_env.sh
module load cuda/11.5

cd ~/angletransformer

echo "Copying data to local scr drive, /scr/jok120/angletransformer/data/"
mkdir -p /scr/jok120/angletransformer/data/
# rsync -azL --timeout=180 --info=progress2 ~/angletransformer/data/ /scr/jok120/angletransformer/data/
rsync -azL --timeout=180 --info=progress2 ~/angletransformer/data/train_min/ /scr/jok120/angletransformer/data/train_min/
rsync -azL --timeout=180 --info=progress2 ~/angletransformer/data/val_min/ /scr/jok120/angletransformer/data/val_min/


python train.py \
    --activation=gelu \
    --c_hidden=2048 \
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
    --train_data=/scr/jok120/angletransformer/data/train_min/ \
    --val_data=/scr/jok120/angletransformer/data/val_min/ \
    --output_dir=out/experiments \
    --num_workers=6 \
    --wandb_tags="sweep,scnmin" \
    --batch_size=1 \
    --experiment_name="sunny-disco-172-g4gpuMin" \
    --num_sanity_val_steps=0

