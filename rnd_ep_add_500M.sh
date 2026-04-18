#!/bin/bash
#SBATCH --job-name=rnd_ep_add_500M
#SBATCH --account=ece567w26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yilech@umich.edu
#SBATCH --output=/scratch/ece567w26_class_root/ece567w26_class/yilech/logs/%j.out
#SBATCH --error=/scratch/ece567w26_class_root/ece567w26_class/yilech/logs/%j.err

source ~/.bashrc
conda activate craftax-exp
cd ~/craftax_phase2/Exploration

python ppo_rnd_episodic.py \
    --total_timesteps 5e8 \
    --num_envs 1024 --num_steps 64 \
    --combination_mode add \
    --rnd_reward_coeff 0.01 \
    --use_wandb --debug \
    --wandb_project craftax-phase2-exp \
    --wandb_entity models-university-of-michigan6208 \
    --seed 42
