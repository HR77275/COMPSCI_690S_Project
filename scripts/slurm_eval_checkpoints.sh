#!/bin/bash
#SBATCH --job-name=hackrl-eval-ckpt
#SBATCH --output=logs/eval_ckpt_%j.out
#SBATCH --error=logs/eval_ckpt_%j.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# ---- Setup ----
set -euo pipefail
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hackrl

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT_DIR="${1:-artifacts/checkpoints/gridworld_ppo_run1}"

# ---- Phase 1: Evaluate all checkpoints ----
echo "=== Evaluating checkpoints in: ${CHECKPOINT_DIR} ==="
python scripts/evaluate_gridworld_checkpoint.py \
    "${CHECKPOINT_DIR}" \
    --num-trajectories 300 \
    --min-exploit-cycles 2 \
    --device cuda

echo "=== Checkpoint evaluation complete ==="
echo ""
echo "NEXT STEP: Pick a checkpoint with both honest and hacked episodes,"
echo "then run activation collection with:"
echo "  sbatch scripts/slurm_collect_activations.sh <checkpoint_file>"
