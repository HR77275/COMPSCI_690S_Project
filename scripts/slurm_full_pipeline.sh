#!/bin/bash
#SBATCH --job-name=hackrl-full-pipeline
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# ===========================================================================
# Full pipeline: two-phase PPO train -> eval -> collect -> SAE -> classify
#
# Phase A: escalation + no termination on goal -> agent discovers cycling hack
# Phase B: flat reward + normal termination    -> agent plays honestly
# ===========================================================================
set -euo pipefail
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hackrl

cd "${SLURM_SUBMIT_DIR:-.}"

HACK_CKPT_DIR="artifacts/checkpoints/gridworld_hack"
HONEST_CKPT_DIR="artifacts/checkpoints/gridworld_honest"
ACT_DIR="artifacts/activations"
SAE_DIR="artifacts/sae"
RESULTS_DIR="artifacts/results"

# ==== STEP 1A: Train PPO — hacking regime ====
# Faulty termination (episode continues past goal) + quadratic escalation.
# Agent reaches goal in ~3 steps, then discovers cycling for escalating reward.
echo "============================================"
echo "STEP 1A/6: Training PPO — hacking regime"
echo "============================================"
python scripts/train_gridworld_ppo.py \
    --total-updates 300 \
    --checkpoint-interval 15 \
    --num-envs 32 \
    --rollout-steps 256 \
    --seed 0 \
    --device cuda \
    --escalation-power 2.0 \
    --goal-bonus 2.0 \
    --no-terminate-on-goal \
    --checkpoint-dir "${HACK_CKPT_DIR}"

# ==== STEP 1B: Train PPO — honest regime ====
echo ""
echo "============================================"
echo "STEP 1B/6: Training PPO — honest regime"
echo "============================================"
python scripts/train_gridworld_ppo.py \
    --total-updates 300 \
    --checkpoint-interval 15 \
    --num-envs 32 \
    --rollout-steps 64 \
    --seed 0 \
    --device cuda \
    --escalation-power 0.0 \
    --goal-bonus 4.0 \
    --checkpoint-dir "${HONEST_CKPT_DIR}"

# ==== STEP 2: Evaluate both sets of checkpoints ====
echo ""
echo "============================================"
echo "STEP 2/6: Evaluating checkpoints"
echo "============================================"
echo "--- Hacking checkpoints ---"
python scripts/evaluate_gridworld_checkpoint.py \
    "${HACK_CKPT_DIR}" \
    --num-trajectories 300 \
    --min-exploit-cycles 2 \
    --device cuda 2>&1 | tee "${HACK_CKPT_DIR}/eval_results.txt"

echo ""
echo "--- Honest checkpoints ---"
python scripts/evaluate_gridworld_checkpoint.py \
    "${HONEST_CKPT_DIR}" \
    --num-trajectories 300 \
    --min-exploit-cycles 2 \
    --device cuda 2>&1 | tee "${HONEST_CKPT_DIR}/eval_results.txt"

# Auto-select best hacking checkpoint (highest hacked%)
echo ""
echo "Auto-selecting checkpoints ..."
HACK_CKPT=$(python3 -c "
import pathlib
text = pathlib.Path('${HACK_CKPT_DIR}/eval_results.txt').read_text()
blocks = text.split('Checkpoint: ')
best_path, best_score = '', 0.0
for block in blocks[1:]:
    lines = block.strip().split('\n')
    name = lines[0].strip()
    hacked_pct = float([l for l in lines if 'Hacked' in l][0].split('(')[1].split('%')[0])
    if hacked_pct > best_score:
        best_score = hacked_pct
        best_path = name
if not best_path:
    # Fallback: pick last checkpoint
    best_path = sorted(pathlib.Path('${HACK_CKPT_DIR}').glob('*.pt'))[-1].name
print(best_path)
")
echo "Selected hacking checkpoint: ${HACK_CKPT_DIR}/${HACK_CKPT}"

# Auto-select best honest checkpoint (highest honest%)
HONEST_CKPT=$(python3 -c "
import pathlib
text = pathlib.Path('${HONEST_CKPT_DIR}/eval_results.txt').read_text()
blocks = text.split('Checkpoint: ')
best_path, best_score = '', 0.0
for block in blocks[1:]:
    lines = block.strip().split('\n')
    name = lines[0].strip()
    honest_pct = float([l for l in lines if 'Honest' in l][0].split('(')[1].split('%')[0])
    if honest_pct > best_score:
        best_score = honest_pct
        best_path = name
print(best_path)
")
echo "Selected honest checkpoint: ${HONEST_CKPT_DIR}/${HONEST_CKPT}"

# ==== STEP 3: Collect activations from both checkpoints ====
echo ""
echo "============================================"
echo "STEP 3/6: Collecting activations"
echo "============================================"
echo "--- Hacking activations ---"
python scripts/collect_activations.py \
    "${HACK_CKPT_DIR}/${HACK_CKPT}" \
    --num-episodes 2000 \
    --min-exploit-cycles 2 \
    --device cuda \
    --output-dir "${ACT_DIR}/hack_raw" \
    --seed 42 \
    --skip-split

echo ""
echo "--- Honest activations ---"
python scripts/collect_activations.py \
    "${HONEST_CKPT_DIR}/${HONEST_CKPT}" \
    --num-episodes 2000 \
    --min-exploit-cycles 2 \
    --device cuda \
    --output-dir "${ACT_DIR}/honest_raw" \
    --seed 42 \
    --skip-split

# Merge and split
echo ""
echo "Merging and splitting datasets ..."
python3 -c "
import torch, random, pathlib, sys
sys.path.insert(0, 'src')
from hackrl.evaluation.activation_collection import load_bundle, save_bundle, split_bundle

hack_b = load_bundle('${ACT_DIR}/hack_raw/full_bundle.pt')
honest_b = load_bundle('${ACT_DIR}/honest_raw/full_bundle.pt')

# Keep only hacked from hack bundle, only honest from honest bundle
hacked_eps = [e for e in hack_b.episodes if e.label == 'hacked']
honest_eps = [e for e in honest_b.episodes if e.label == 'honest']
print(f'Hacked episodes: {len(hacked_eps)}')
print(f'Honest episodes: {len(honest_eps)}')

# Balance classes: take min of both
n = min(len(hacked_eps), len(honest_eps))
random.seed(42)
random.shuffle(hacked_eps)
random.shuffle(honest_eps)
hacked_eps = hacked_eps[:n]
honest_eps = honest_eps[:n]
print(f'Balanced to {n} each ({2*n} total)')

from hackrl.evaluation.activation_collection import ActivationDatasetBundle
merged = ActivationDatasetBundle(
    episodes=hacked_eps + honest_eps,
    feature_dim=hack_b.feature_dim,
)
save_bundle(merged, '${ACT_DIR}/full_bundle.pt')

train, val, test = split_bundle(merged, seed=42)
save_bundle(train, '${ACT_DIR}/train.pt')
save_bundle(val, '${ACT_DIR}/val.pt')
save_bundle(test, '${ACT_DIR}/test.pt')
print(f'Train: {len(train.episodes)}  {train.label_counts()}')
print(f'Val:   {len(val.episodes)}  {val.label_counts()}')
print(f'Test:  {len(test.episodes)}  {test.label_counts()}')
"

# ==== STEP 4: Train SAE ====
echo ""
echo "============================================"
echo "STEP 4/6: Training SAE"
echo "============================================"
python scripts/train_sae.py \
    "${ACT_DIR}/train.pt" \
    --dict-multiplier 8 \
    --sparsity-coef 1e-3 \
    --epochs 50 \
    --seed 42 \
    --device cuda \
    --checkpoint-dir "${SAE_DIR}"

# ==== STEP 5: Classification + baselines ====
echo ""
echo "============================================"
echo "STEP 5/6: Classification & comparison"
echo "============================================"
python scripts/run_classification.py \
    --train-data "${ACT_DIR}/train.pt" \
    --val-data "${ACT_DIR}/val.pt" \
    --test-data "${ACT_DIR}/test.pt" \
    --sae-checkpoint "${SAE_DIR}/sae_trained.pt" \
    --aggregation mean \
    --device cpu \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "============================================"
echo "PIPELINE COMPLETE"
echo "============================================"
echo "Artifacts:"
echo "  Hack checkpoints:   ${HACK_CKPT_DIR}/"
echo "  Honest checkpoints: ${HONEST_CKPT_DIR}/"
echo "  Activations:        ${ACT_DIR}/"
echo "  SAE:                ${SAE_DIR}/"
echo "  Results:            ${RESULTS_DIR}/"
echo ""
echo "Key files:"
echo "  ${RESULTS_DIR}/classification_results.json"
echo "  ${RESULTS_DIR}/metric_comparison.png"
echo "  ${RESULTS_DIR}/roc_comparison.png"
