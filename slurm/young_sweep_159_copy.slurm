#!/bin/bash
#SBATCH --job-name=young^0
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=6-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="out/%A_%6a.out"
#SBATCH --ntasks-per-node=24
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
    --c_hidden=128 \
    --chi_weight=8.720481984584808 \
    --d_ff=512 \
    --dropout=0.09362803755294724 \
    --is_sweep=True \
    --no_blocks=42 \
    --no_heads=2 \
    --opt_lr=0.0010351170087637854 \
    --opt_lr_scheduling=plateau \
    --opt_lr_scheduling_metric=val/angle_mae \
    --opt_n_warmup_steps=6924 \
    --opt_name=sgd \
    --opt_noam_lr_factor=1.6689464805176095 \
    --replace_sampler_ddp=True \
    --train_data=/scr/jok120/angletransformer/data/train/ \
    --val_data=/scr/jok120/angletransformer/data/val/ \
    --output_dir=out/experiments \
    --num_workers=6 \
    --wandb_tags="sweep" \
    --batch_size=1 \
    --val_check_interval=2500 \
    --experiment_name="young-sweep-159-copy-SWA" \
    --opt_patience=10 \
    --seed=0 \
    --conv_encoder=True \
    --check_val_every_n_epoch=1 \
    --opt_min_delta=0.005 \
    --stochastic_weight_avg=True

