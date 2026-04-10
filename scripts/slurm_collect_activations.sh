#!/bin/bash
#SBATCH --job-name=hackrl-collect-act
#SBATCH --output=logs/collect_act_%j.out
#SBATCH --error=logs/collect_act_%j.err
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

CHECKPOINT="${1:?Usage: sbatch slurm_collect_activations.sh <checkpoint.pt>}"
NUM_EPISODES="${2:-1000}"

# ---- Phase 2: Collect activations ----
echo "=== Collecting activations from: ${CHECKPOINT} ==="
echo "    Episodes: ${NUM_EPISODES}"

python scripts/collect_activations.py \
    "${CHECKPOINT}" \
    --num-episodes "${NUM_EPISODES}" \
    --min-exploit-cycles 2 \
    --device cuda \
    --output-dir artifacts/activations \
    --seed 42

echo "=== Activation collection complete ==="
echo ""
echo "NEXT STEP: Train SAE with:"
echo "  sbatch scripts/slurm_train_sae.sh"
