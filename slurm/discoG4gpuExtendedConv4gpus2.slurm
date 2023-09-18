#!/bin/bash
#SBATCH --job-name=discoGEs2
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=28-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="out/%A_%6a.out"
#SBATCH --ntasks-per-node=12
#SBATCH --exclude=g019

# Setup
cd ~/openfold
source scripts/activate_conda_env.sh
module load cuda/11.5


if [ ! -d /scr/jok120/angletransformer/ ]; then
    echo "The directory /scr/jok120/angletransformer/ does not exist"
else
    echo "The directory /scr/jok120/angletransformer/ exists, continuing"
    # Check if the placeholder file is there
    if [ ! -f /scr/jok120/angletransformer/data/these_files_were_checked.txt   ]; then
        echo "Removing files that shouldn't be there"
        for f in ~/angletransformer/data/val/*; do rm /scr/jok120/angletransformer/data/train/$(basename $f); done
        for f in ~/angletransformer/data/train/*; do rm /scr/jok120/angletransformer/data/val/$(basename $f); done
        echo "Done removing excess files"
        echo "Done removing excess files" > /scr/jok120/angletransformer/data/these_files_were_checked.txt
    else
        echo "The placeholder file is there, continuing"
    fi
fi

cd ~/angletransformer

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
    --opt_lr_scheduling_metric=val/angle_mae \
    --opt_n_warmup_steps=10000 \
    --opt_name=adam \
    --opt_noam_lr_factor=2 \
    --replace_sampler_ddp=True \
    --train_data=data/train/ \
    --val_data=data/val/ \
    --output_dir=out/experiments \
    --num_workers=6 \
    --wandb_tags="sweep" \
    --batch_size=1 \
    --val_check_interval=2500 \
    --experiment_name="sunny-disco-172-g4gpu-extended2-conv-4gpu" \
    --opt_patience=20 \
    --seed=2 \
    --conv_encoder=True

