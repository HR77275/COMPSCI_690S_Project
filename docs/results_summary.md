# Current Results and Execution Plan

This document summarizes the current experiment state after the May 2026 audit pass. It is meant to support the final report by separating strong claims from caveats.

## Immediate Audit Plan

1. Verify that saved outputs match the project claim rather than only showing an easy in-domain separation.
2. Check transfer-evaluation logic for calibration and threshold artifacts.
3. Add a regression test for any evaluation bug found during the audit.
4. Summarize the completed evidence: in-domain detection, cross-seed transfer, cross-task transfer, interpretability, and negative robustness results.
5. Identify the safest final-project claim and the remaining optional work.

## What Was Executed

- Fixed transfer threshold tuning in `scripts/run_transfer_classification.py`.
- Recomputed transfer summaries for grid seed0 to seed1, grid to LunarLander, and LunarLander to grid.
- Added `tests/test_transfer_thresholds.py` so compressed transfer probabilities and direction flips remain covered.
- Re-ran the repository unit tests: `7` tests passed.
- Re-ran compile checks for the modified scripts and classifier modules.

## Dataset Snapshot

| Dataset | Source checkpoint | Train | Val | Test | Notes |
|---|---:|---:|---:|---:|---|
| Gridworld seed0 | `gridworld_hack/update_00300.pt` | 1668 | 357 | 360 | Same-policy honest/hacked split; 682 hacked train, 147 hacked test |
| Gridworld seed1 | `gridworld_hack_seed1/update_00135.pt` | 1555 | 333 | 334 | Used as held-out seed transfer target |
| LunarLander seed0 | `lunar_hack/update_00600.pt` | 1190 | 254 | 257 | Same-policy honest/hacked split; 564 hacked train, 122 hacked test |
| Pooled train | grid train + lunar train | 2858 | n/a | n/a | Used to train a shared SAE dictionary |

## In-Domain Results

Both controlled domains are very separable once honest and hacked examples are collected from a checkpoint that produces both behaviors.

| Domain | SAE test AUROC | SAE test F1 | Raw test AUROC | Behavioral test AUROC |
|---|---:|---:|---:|---:|
| Gridworld seed0 | 1.000 | 1.000 | 1.000 | 1.000 |
| LunarLander seed0 | 1.000 | 1.000 | 1.000 | 1.000 |

Interpretation: this validates the full activation-to-SAE-to-classifier pipeline, but it is not enough by itself to prove SAE superiority because raw activations and behavioral baselines also solve the in-domain split.

## SAE Sparsity

The high-L1 SAE settings preserve perfect in-domain classification while making the representations much sparser than the midpoint run.

| Domain | Selected sparsity coef | Recon loss | Mean L0 | Test AUROC |
|---|---:|---:|---:|---:|
| Gridworld | `3e-2` | `1.35e-5` | 34.79 / 1024 | 1.000 |
| LunarLander | `3e-2` | `4.25e-5` | 57.56 / 1024 | 1.000 |
| Pooled cross-task SAE | `3e-2` | `4.00e-5` | 44.5 / 1024 | n/a |

Interpretation: the check-in concern about dense SAE features has been addressed substantially, although the SAE is still not a Top-K SAE.

## Generalization Results

### Gridworld Cross-Seed Transfer

Train on gridworld seed0, evaluate on gridworld seed1.

| Method | Test AUROC | Default F1 | Tuned-threshold F1 | Tuned accuracy |
|---|---:|---:|---:|---:|
| SAE | 0.988 | 0.482 | 0.936 | 0.961 |
| Raw activations | 0.957 | 0.902 | 0.919 | 0.952 |
| Behavioral | 1.000 | 1.000 | 1.000 | 1.000 |

Interpretation: the SAE ranking transfers strongly across seeds, but calibration shifts. Validation-tuned thresholding is needed for good F1/accuracy.

### Cross-Task Transfer With Pooled SAE

Train a pooled SAE on grid + Lunar activations. Train the classifier on one task and evaluate on the other.

| Direction | Method | Test AUROC | Tuned direction | Tuned F1 | Tuned accuracy |
|---|---|---:|---|---:|---:|
| Grid -> Lunar | SAE | 0.897 after flip | flipped | 0.795 | 0.790 |
| Grid -> Lunar | Raw activations | 0.952 | as-is | 0.881 | 0.895 |
| Lunar -> Grid | SAE | 0.773 | as-is | 0.847 | 0.889 |
| Lunar -> Grid | Raw activations | 0.974 | as-is | 0.906 | 0.919 |

Interpretation: there is real cross-task signal, but raw activations remain a very strong baseline. The safest claim is that SAE features support transfer detection, not that they dominate raw activations across tasks.

## Interpretability Findings

Gridworld top SAE features separate class structure cleanly:

- Feature `686` and `851` are more active in hacked episodes.
- Feature `941`, `730`, and `253` are more active in honest episodes.
- Several grid features correlate with progress/reversal/exploit-cycle diagnostics, suggesting they track task-manipulation dynamics rather than arbitrary policy identity alone.

LunarLander top SAE features also separate classes:

- Features `649`, `104`, and `145` are more active in hacked episodes.
- Features `822` and `590` are more active in honest episodes.
- Lunar top features correlate strongly with total reward, episode length, and zone-cycle diagnostics. Progress/reversal correlations are zero because those grid-specific counters are not meaningful for Lunar.

## Important Caveats

- Behavioral baselines are intentionally strong because they use simulator-side trajectory statistics. They are useful as an upper bound, not as a deployable hidden-state detector.
- Perfect in-domain results mean the controlled splits are easy. The transfer experiments are the more important evidence.
- Cross-task transfer is not solved universally. The grid-to-Lunar SAE classifier flips direction under transfer, so the final report should describe validation-based threshold/direction tuning explicitly.
- Lunar hacking is seed-sensitive. Seed0 produced a strong mixed checkpoint at update 600; seed1 did not produce hacked episodes under the same evaluation threshold, even after resuming training.
- The project should not claim SAE superiority over raw activations. It can claim a working SAE-based detector, interpretable sparse features, and nontrivial transfer evidence with strong baselines.

## Safest Final Claim

The strongest defensible claim is:

> We built an end-to-end pipeline for inducing reward hacking, collecting policy activations, learning sparse activation features, and detecting hacked trajectories. SAE features achieve perfect in-domain detection in gridworld and LunarLander, transfer strongly across gridworld seeds after validation calibration, and retain nontrivial cross-task signal under a pooled dictionary. Raw activations remain a strong baseline, so the project demonstrates feasibility and interpretability rather than a universal SAE advantage.

## Optional Next Work

If more time is available, prioritize these in order:

1. Add a short final-report table from this document rather than running more training.
2. Add one more gridworld reward-configuration transfer if a stronger generalization claim is needed.
3. Generate trajectory/time-aligned plots for the top gridworld SAE features around exploit-cycle events.
4. Try Lunar seed2 or adjusted hyperparameters only if cross-seed Lunar robustness becomes essential.
5. Avoid promising a universal detector unless a mixed-task classifier is explicitly evaluated and compared against raw/behavioral baselines.
