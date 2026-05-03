#!/bin/bash
# End-to-end LunarLander pipeline: train -> collect -> SAE -> classify
#
# Phase H: escalating reward + no landing termination  -> agent discovers oscillation hack
# Phase A: flat reward + normal termination            -> agent lands honestly
#
# Usage (local):
#   chmod +x scripts/run_lunar_pipeline.sh
#   ./scripts/run_lunar_pipeline.sh
#
# Usage (SLURM):
#   sbatch --job-name=lunar-pipeline --partition=gpu-preempt \
#          --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=04:00:00 \
#          scripts/run_lunar_pipeline.sh
set -euo pipefail
mkdir -p logs

PYTHON="/Users/moiramcdermo/miniconda3/envs/hackirl/bin/python"
HACK_DIR="artifacts/checkpoints/lunar_hack"
HONEST_DIR="artifacts/checkpoints/lunar_honest"
ACT_DIR="artifacts/activations/lunar"
SAE_DIR="artifacts/sae/lunar"
RESULTS_DIR="artifacts/results/lunar"

# ==== STEP 1A: Train — hacking regime ====
echo "============================================"
echo "STEP 1A: Training PPO — hacking regime"
echo "============================================"
"${PYTHON}" scripts/train_lunar_ppo.py \
    --total-updates 1000 \
    --checkpoint-interval 50 \
    --num-envs 16 \
    --rollout-steps 128 \
    --seed 0 \
    --device cpu \
    --escalation-power 2.0 \
    --goal-bonus 50.0 \
    --checkpoint-dir "${HACK_DIR}"

# ==== STEP 1B: Train — honest regime ====
echo ""
echo "============================================"
echo "STEP 1B: Training PPO — honest regime"
echo "============================================"
"${PYTHON}" scripts/train_lunar_ppo.py \
    --total-updates 500 \
    --checkpoint-interval 25 \
    --num-envs 16 \
    --rollout-steps 128 \
    --seed 0 \
    --device cpu \
    --escalation-power 0.0 \
    --goal-bonus 100.0 \
    --checkpoint-dir "${HONEST_DIR}"

# ==== STEP 2: Auto-select best checkpoints ====
echo ""
echo "Selecting best hacking checkpoint ..."
HACK_CKPT=$(ls -1 "${HACK_DIR}"/*.pt | sort -V | tail -1)
echo "  Using: ${HACK_CKPT}"

HONEST_CKPT=$(ls -1 "${HONEST_DIR}"/*.pt | sort -V | tail -1)
echo "  Using: ${HONEST_CKPT}"

# ==== STEP 3: Collect activations ====
echo ""
echo "============================================"
echo "STEP 3: Collecting activations"
echo "============================================"
echo "--- Hacking checkpoint ---"
"${PYTHON}" scripts/collect_lunar_activations.py \
    "${HACK_CKPT}" \
    --num-episodes 1500 \
    --min-exploit-cycles 2 \
    --device cpu \
    --output-dir "${ACT_DIR}/hack_raw" \
    --seed 42 \
    --skip-split

echo ""
echo "--- Honest checkpoint ---"
"${PYTHON}" scripts/collect_lunar_activations.py \
    "${HONEST_CKPT}" \
    --num-episodes 1500 \
    --min-exploit-cycles 2 \
    --device cpu \
    --output-dir "${ACT_DIR}/honest_raw" \
    --seed 42 \
    --skip-split

# ==== STEP 4: Split hack_raw only (same-policy experiment) ====
echo ""
echo "============================================"
echo "STEP 4: Splitting hack_raw into train/val/test"
echo "============================================"
"${PYTHON}" - <<'PYEOF'
import sys, pathlib
sys.path.insert(0, 'src')
from hackrl.evaluation.activation_collection import load_bundle, save_bundle, split_bundle

bundle = load_bundle('artifacts/activations/lunar/hack_raw/full_bundle.pt')
print(f"Loaded: {len(bundle.episodes)} episodes  {bundle.label_counts()}")

out = pathlib.Path('artifacts/activations/lunar/hack_only')
out.mkdir(parents=True, exist_ok=True)

train, val, test = split_bundle(bundle, seed=42)
save_bundle(train, out / 'train.pt')
save_bundle(val,   out / 'val.pt')
save_bundle(test,  out / 'test.pt')
print(f"Train: {len(train.episodes)}  {train.label_counts()}")
print(f"Val:   {len(val.episodes)}  {val.label_counts()}")
print(f"Test:  {len(test.episodes)}  {test.label_counts()}")
PYEOF

# ==== STEP 5: Train SAE ====
echo ""
echo "============================================"
echo "STEP 5: Training SAE"
echo "============================================"
"${PYTHON}" scripts/train_sae.py \
    "${ACT_DIR}/hack_only/train.pt" \
    --dict-multiplier 8 \
    --sparsity-coef 1e-2 \
    --epochs 50 \
    --seed 42 \
    --device cpu \
    --checkpoint-dir "${SAE_DIR}"

# ==== STEP 6: Classify ====
echo ""
echo "============================================"
echo "STEP 6: Classification"
echo "============================================"
"${PYTHON}" scripts/run_classification.py \
    --train-data "${ACT_DIR}/hack_only/train.pt" \
    --val-data   "${ACT_DIR}/hack_only/val.pt" \
    --test-data  "${ACT_DIR}/hack_only/test.pt" \
    --sae-checkpoint "${SAE_DIR}/sae_trained.pt" \
    --aggregation mean \
    --device cpu \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "============================================"
echo "PIPELINE COMPLETE"
echo "============================================"
echo "Results: ${RESULTS_DIR}/classification_results.json"
