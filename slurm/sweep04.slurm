#!/bin/bash
#SBATCH --job-name=swp4
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

cd ~/angletransformer

# Check if /scr/jok120/angletransformer/ exists AND if the placeholder file is there
# If /scr/jok120/angletransformer/ exists and the placeholder is NOT there, then remove files that shouldn't be there
# If the placeholder file is there, alert the user and continue

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


echo "Copying data to local scr drive, /scr/jok120/angletransformer/data/"
mkdir -p /scr/jok120/angletransformer/data/
# rsync -azL --timeout=180 --info=progress2 ~/angletransformer/data/ /scr/jok120/angletransformer/data/
rsync -azL --timeout=180 --info=progress2 ~/angletransformer/data/val/ /scr/jok120/angletransformer/data/val/ &
rsync -azL --timeout=180 --info=progress2 ~/angletransformer/data/train/ /scr/jok120/angletransformer/data/train/

wandb agent --count 1 koes-group/angletransformer_solo01/chhqej17
