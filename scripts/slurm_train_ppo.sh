#!/bin/bash
#SBATCH --job-name=hackrl-ppo-train
#SBATCH --output=logs/ppo_train_%j.out
#SBATCH --error=logs/ppo_train_%j.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# ---- Setup ----
set -euo pipefail
mkdir -p logs

# Activate conda environment (edit if your conda is elsewhere)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hackrl

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- Phase 1: Train PPO ----
echo "=== Starting PPO training ==="
python scripts/train_gridworld_ppo.py \
    --total-updates 300 \
    --checkpoint-interval 15 \
    --num-envs 32 \
    --rollout-steps 64 \
    --learning-rate 3e-4 \
    --seed 0 \
    --device cuda \
    --checkpoint-dir artifacts/checkpoints/gridworld_ppo_run1 \
    --log-interval 10

echo "=== PPO training complete ==="
