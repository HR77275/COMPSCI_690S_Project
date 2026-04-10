# TODO — Reward Hacking Detection via Mechanistic Interpretability

Status legend: `[ ]` not started, `[~]` in progress, `[x]` done

---

## Phase 0: Existing Foundation (complete)

- [x] Implement `BoxProgressGridWorld` with hackable shaped rewards
- [x] Add exploit-cycle diagnostics and trajectory labeling
- [x] Implement PPO actor-critic policy (`GridWorldPPOPolicy`) with exposed trunk features
- [x] Implement PPO training loop with checkpointing
- [x] Write checkpoint evaluation script (honest / hacked / neutral labels)
- [x] Add scripted sanity-check trajectories and regression tests

---

## Phase 1: Train PPO and Select Checkpoint

- [ ] **1.1** Run a full PPO training run (≥300 updates) and save checkpoints
  ```bash
  python scripts/train_gridworld_ppo.py --checkpoint-dir artifacts/checkpoints/run1
  ```
- [ ] **1.2** Evaluate all checkpoints with stochastic sampling (≥300 trajectories each)
  ```bash
  python scripts/evaluate_gridworld_checkpoint.py artifacts/checkpoints/run1 \
    --num-trajectories 300 --min-exploit-cycles 2
  ```
- [ ] **1.3** Select a checkpoint that produces a useful mix of honest and hacked trajectories
- [ ] **1.4** (Optional) Sweep reward parameters (`progress_reward`, `goal_bonus`, `step_penalty`) if the default config does not yield enough hacking behavior
- [ ] **1.5** (Optional) Add randomized initial states to increase trajectory diversity

---

## Phase 2: Activation Collection Pipeline

- [ ] **2.1** Write an activation-collection script/module that:
  - loads a frozen checkpoint,
  - runs N rollout episodes (target: ≥500 honest + ≥500 hacked),
  - records per-timestep hidden activations from the policy trunk,
  - records per-timestep metadata (step index, action, reward, info flags),
  - records per-episode label (honest / hacked / neutral),
  - saves everything to disk (e.g., `.pt` or `.npz` files).
- [ ] **2.2** Add a CLI script `scripts/collect_activations.py` wrapping the above
- [ ] **2.3** Verify activation shapes and label distribution; ensure ≥500 episodes per class
- [ ] **2.4** Split dataset into train / val / test (70 / 15 / 15) with stratification by class

---

## Phase 3: Train Sparse Autoencoder (SAE)

- [ ] **3.1** Implement SAE model (`src/hackrl/sae/model.py`):
  - encoder: linear → ReLU
  - decoder: linear (tied or untied weights)
  - loss: reconstruction (MSE) + L1 sparsity penalty
  - configurable dictionary size (e.g., 4× to 8× activation dim)
- [ ] **3.2** Implement SAE training loop (`src/hackrl/sae/trainer.py`):
  - dataloader over collected activations (all timesteps, ignoring labels)
  - learning rate schedule, logging of reconstruction loss and L0 sparsity
- [ ] **3.3** Add a CLI script `scripts/train_sae.py`
- [ ] **3.4** Train SAE; tune the sparsity coefficient until features are reasonably sparse (target L0 ~5–15 active features per timestep)
- [ ] **3.5** Save trained SAE checkpoint

---

## Phase 4: SAE Feature Projection and Episode-Level Aggregation

- [ ] **4.1** Write a projection utility that:
  - loads trained SAE,
  - encodes every timestep activation into the sparse feature space,
  - saves per-timestep sparse feature vectors alongside metadata
- [ ] **4.2** Implement episode-level aggregation strategies:
  - mean of per-timestep feature vectors across the episode
  - max of per-timestep feature vectors across the episode
  - (optional) weighted mean by timestep position
- [ ] **4.3** Produce a final dataset of `(episode_feature_vector, label)` pairs for the classifier

---

## Phase 5: Train Hacking Classifier

- [ ] **5.1** Implement classifier training (`src/hackrl/classifier/train.py`):
  - logistic regression on SAE episode features
  - small MLP on SAE episode features
- [ ] **5.2** Train on the training split; tune on validation split (hyperparameters: regularization, hidden size, aggregation strategy)
- [ ] **5.3** Evaluate on the held-out test set; record AUROC, F1-score, accuracy

---

## Phase 6: Baselines

- [ ] **6.1** **Raw-activation baseline**: train logistic regression / MLP on raw trunk activations (same aggregation, same splits)
- [ ] **6.2** **Behavioral-feature baseline**: train classifier on trajectory statistics (total reward, goal reached, exploit cycles, reversal count, episode length, etc.)
- [ ] **6.3** **Random baseline**: report chance-level AUROC (0.50)
- [ ] **6.4** Compare all baselines; report whether SAE classifier outperforms raw-activation baseline by a statistically significant margin (e.g., paired bootstrap or permutation test on AUROC)

---

## Phase 7: Interpretability Analysis

- [ ] **7.1** Identify the top-K most predictive SAE features (by classifier weight magnitude or SHAP values)
- [ ] **7.2** For each top feature, inspect:
  - when it activates along trajectories (honest vs. hacked),
  - correlation with environment signals (proximity to exploit states, reversal events, progress events, box-goal distance)
- [ ] **7.3** Produce per-feature activation heatmaps or time-series plots
- [ ] **7.4** Write human-interpretable descriptions of the most predictive features
- [ ] **7.5** (Optional) Ablation study: zero out top features and measure classifier performance drop

---

## Phase 8: Stretch Goals

- [ ] **8.1** Add a continuous-control MuJoCo environment with a hackable reward
- [ ] **8.2** Train SAC policy for MuJoCo, repeat activation collection and SAE pipeline
- [ ] **8.3** Compare SAE-based detection across grid-world and MuJoCo domains

---

## Phase 9: Write-Up and Deliverables

- [ ] **9.1** Summarize experimental results (tables: AUROC, F1, accuracy for each method)
- [ ] **9.2** Create figures:
  - training curves (reward vs. update)
  - checkpoint behavior distribution (honest / hacked / neutral over training)
  - SAE reconstruction loss and sparsity during training
  - classifier ROC curves
  - top SAE feature activation plots
- [ ] **9.3** Write final report sections: introduction, related work, method, experiments, results, discussion
- [ ] **9.4** Proofread and finalize

---

## Success Criteria (from proposal)

1. AUROC ≥ 0.70 on held-out test set for the SAE-based classifier
2. SAE classifier outperforms raw-activation baseline (statistically significant)
3. Top predictive SAE features admit human-interpretable descriptions consistent with exploitative behavior
