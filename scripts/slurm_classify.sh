#!/bin/bash
#SBATCH --job-name=hackrl-classify
#SBATCH --output=logs/classify_%j.out
#SBATCH --error=logs/classify_%j.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# ---- Setup ----
set -euo pipefail
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hackrl

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---- Phases 5-7: Classification + baselines + interpretability ----
echo "=== Running classification comparison ==="
python scripts/run_classification.py \
    --train-data artifacts/activations/train.pt \
    --val-data artifacts/activations/val.pt \
    --test-data artifacts/activations/test.pt \
    --sae-checkpoint artifacts/sae/sae_trained.pt \
    --aggregation mean \
    --device cpu \
    --output-dir artifacts/results

echo "=== Classification complete ==="
echo ""
echo "Results saved to artifacts/results/"
echo "  - classification_results.json"
echo "  - metric_comparison.png"
echo "  - roc_comparison.png"
