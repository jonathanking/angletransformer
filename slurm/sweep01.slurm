#!/bin/bash
#SBATCH --job-name=swp2
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

wandb agent --count 10 koes-group/angletransformer_solo01/5ijrr6sf
