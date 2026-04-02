#!/bin/bash
#SBATCH --job-name=dino_wm
#SBATCH --partition=nvidia
#SBATCH --account=civil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=/scratch/gs4133/zhd/Coop_lerobot/slurm-log/dino_wm-%j.out
#SBATCH --error=/scratch/gs4133/zhd/Coop_lerobot/slurm-log/dino_wm-%j.err

export HF_TOKEN=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

module load miniconda/3-4.11.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lerobot

cd /scratch/gs4133/zhd/Coop_lerobot

lerobot-train \
    --policy.type=dino_wm_test \
    --policy.push_to_hub=true \
    --policy.repo_id=haodoz0118/dino_wm_test_PickAndPlace \
    --dataset.repo_id=haodoz0118/PickAndPlace \
    --batch_size=1 \
    --steps=200000 \
    --save_freq=10000 \
    --log_freq=100 \
    --num_workers=4 \
    --seed=1000
