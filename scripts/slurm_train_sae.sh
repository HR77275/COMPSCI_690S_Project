#!/bin/bash
#SBATCH --job-name=hackrl-train-sae
#SBATCH --output=logs/train_sae_%j.out
#SBATCH --error=logs/train_sae_%j.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# ---- Setup ----
set -euo pipefail
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hackrl

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_DATA="${1:-artifacts/activations/train.pt}"

# ---- Phase 3: Train SAE ----
echo "=== Training SAE on: ${TRAIN_DATA} ==="
python scripts/train_sae.py \
    "${TRAIN_DATA}" \
    --dict-multiplier 8 \
    --sparsity-coef 1e-3 \
    --learning-rate 1e-3 \
    --batch-size 256 \
    --epochs 50 \
    --seed 42 \
    --device cuda \
    --checkpoint-dir artifacts/sae

echo "=== SAE training complete ==="
echo ""
echo "NEXT STEP: Run classification with:"
echo "  sbatch scripts/slurm_classify.sh"
