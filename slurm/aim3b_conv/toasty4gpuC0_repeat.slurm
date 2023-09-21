#!/bin/bash
#SBATCH --job-name=toastC0
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=28-00:00:00
#SBATCH --partition=dept_gpu
#SBATCH --output="out/%A_%6a.out"
#SBATCH --ntasks-per-node=12
#SBATCH --nodelist=g008

# Setup
cd ~/openfold
source scripts/activate_conda_env.sh
module load cuda/11.5


if [ ! -d /scr/jok120/angletransformer/ ]; then
    echo "The directory /scr/jok120/angletransformer/ does not exist"
    rsync -azL --timeout=180 --info=progress2 ~/angletransformer/data/val/ /scr/jok120/angletransformer/data/val/ &
    rsync -azL --timeout=180 --info=progress2 ~/angletransformer/data/train/ /scr/jok120/angletransformer/data/train/
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
    --c_hidden=64 \
    --chi_weight=13.012551860064766 \
    --d_ff=1024 \
    --dropout=0.018609913167811645 \
    --is_sweep=True \
    --no_blocks=42 \
    --no_heads=1 \
    --opt_lr=0.00037082862954881073 \
    --opt_lr_scheduling=none \
    --opt_lr_scheduling_metric=val/angle_mae \
    --opt_n_warmup_steps=7336 \
    --opt_name=adamw \
    --opt_noam_lr_factor=1.9248774755061016 \
    --replace_sampler_ddp=True \
    --train_data=/scr/jok120/angletransformer/data/train/ \
    --val_data=/scr/jok120/angletransformer/data/val/ \
    --output_dir=out/experiments/ \
    --num_workers=6 \
    --wandb_tags="aim3b,toastyconv" \
    --batch_size=1 \
    --val_check_interval=2500 \
    --experiment_name="toastyC0_repeat" \
    --opt_patience=10 \
    --conv_encoder=True \
    --seed=0 \
    --stochastic_weight_avg="false" \

