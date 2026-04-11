# Agile Stories — Midpoint Delivery

> **Project:** Reward Hacking Detection via Mechanistic Interpretability in RL Agents
> **Goal:** End-to-end grid-world pipeline: frozen policy -> activations -> SAE -> classifier with at least one quantitative baseline, demonstrating whether internal representations support above-chance hacking detection.
> **Mapping:** Each Epic corresponds to a midpoint-todo section (A–E); stories map to individual checklist items.

---

## Epic A: Policy Training and Data Collection (Phases 1–2)

Train PPO to produce both honest and hacked behaviors, collect labeled activations, and prepare data splits.

### Story A.1 — Run PPO Training to Completion

**As a** researcher,
**I want to** run a full PPO training run with checkpointing,
**so that** I have a set of policy snapshots spanning early to converged training.

**Acceptance Criteria:**
- [ ] PPO training completes >= 300 updates without errors.
- [ ] Checkpoints are saved periodically to `artifacts/checkpoints/run1/`.
- [ ] Training logs (reward per update, loss) are recorded and reviewable.

**Tasks:**
1. Verify training script (`scripts/train_gridworld_ppo.py`) runs end-to-end on the cluster.
2. Submit SLURM job (`scripts/slurm_train_gridworld_ppo.sh`) for the full run.
3. Confirm checkpoint files are written to disk at the expected intervals.

**Ref:** TODO.md `1.1` | Midpoint `A.1`

---

### Story A.2 — Evaluate Checkpoints and Select Best Candidate

**As a** researcher,
**I want to** evaluate all saved checkpoints on stochastic rollouts and pick one that produces both honest and hacked trajectories in non-trivial proportion,
**so that** I have a frozen policy suitable for activation collection.

**Acceptance Criteria:**
- [ ] Each checkpoint is evaluated on >= 300 stochastic trajectories.
- [ ] Evaluation output reports per-checkpoint counts: honest / hacked / neutral episodes.
- [ ] One checkpoint is selected where **both** honest and hacked classes have >= 10 % share.
- [ ] Selected checkpoint ID and its behavior distribution are documented.

**Tasks:**
1. Run `scripts/evaluate_gridworld_checkpoint.py` across all checkpoints.
2. Produce a table or plot of behavior mix vs. checkpoint step.
3. Record the chosen checkpoint and rationale.
4. (Fallback) If no checkpoint meets the threshold, sweep reward parameters (`progress_reward`, `goal_bonus`, `step_penalty`) and re-train.

**Ref:** TODO.md `1.2`–`1.3` | Midpoint `A.2`

---

### Story A.3 — Implement Activation Collection Module

**As a** researcher,
**I want to** implement a reusable module that loads a frozen checkpoint, runs rollouts, and records per-timestep trunk activations with labels and metadata,
**so that** I can build datasets for SAE training and downstream classification.

**Acceptance Criteria:**
- [ ] Module (`src/hackrl/evaluation/activation_collection.py`) loads any saved checkpoint.
- [ ] Per-timestep data recorded: hidden activation vector, step index, action, reward, info flags.
- [ ] Per-episode data recorded: episode label (honest / hacked / neutral), total reward, episode length.
- [ ] Output saved to disk as `.pt` or `.npz` files with documented schema.
- [ ] Unit tests in `tests/test_activation_collection.py` pass.

**Tasks:**
1. Implement the collection logic in `src/hackrl/evaluation/activation_collection.py`.
2. Wire it into a CLI entrypoint in `scripts/collect_activations.py`.
3. Write and run unit tests to verify shape, dtype, and label correctness.

**Ref:** TODO.md `2.1`–`2.2` | Midpoint `A.3`

---

### Story A.4 — Collect Activation Dataset (Pilot or Full)

**As a** researcher,
**I want to** collect a labeled activation dataset with at least 100–200 episodes per class (stretch: >= 500 per class),
**so that** I have enough data to train a SAE and classifier.

**Acceptance Criteria:**
- [ ] Dataset contains at minimum **100 episodes per class** (honest, hacked); stretch target is 500+.
- [ ] Activation tensor shapes are verified (episodes x timesteps x activation_dim).
- [ ] Label distribution is logged and reviewed for class balance.
- [ ] Dataset is saved to a versioned artifacts directory.

**Tasks:**
1. Run `scripts/collect_activations.py` with the selected checkpoint.
2. Inspect output: shapes, label counts, any NaN/inf checks.
3. If < 100 hacked episodes, increase rollout count or adjust sampling temperature.

**Ref:** TODO.md `2.3` | Midpoint `A.4`

---

### Story A.5 — Split Dataset into Train / Val / Test

**As a** researcher,
**I want to** split the collected dataset into 70 / 15 / 15 train / val / test partitions, stratified by class,
**so that** all downstream experiments use consistent, reproducible splits.

**Acceptance Criteria:**
- [ ] Splits are stratified by episode label (honest vs. hacked).
- [ ] Random seed is documented and fixed for reproducibility.
- [ ] Split files are saved separately (e.g., `train.pt`, `val.pt`, `test.pt`).
- [ ] Per-split class counts are logged.

**Tasks:**
1. Add a splitting utility (can be part of the collection script or standalone).
2. Run splitting; verify proportions and stratification.
3. Record the random seed used.

**Ref:** TODO.md `2.4` | Midpoint `A.5`

---

## Epic B: Sparse Autoencoder Training (Phases 3–4)

Implement, train, and validate a Sparse Autoencoder on the collected activations; produce episode-level feature representations.

### Story B.1 — Implement and Train Sparse Autoencoder

**As a** researcher,
**I want to** implement an SAE (encoder: linear -> ReLU, decoder: linear, loss: MSE + L1 sparsity) and train it on pooled activations,
**so that** I obtain a sparse dictionary of learned features from the policy's internal representations.

**Acceptance Criteria:**
- [ ] SAE model implemented in `src/hackrl/sae/model.py` with configurable dictionary size (4x–8x activation dim).
- [ ] Training loop implemented in `src/hackrl/sae/trainer.py` with LR schedule and logging.
- [ ] CLI script `scripts/train_sae.py` runs end-to-end.
- [ ] SAE trained; checkpoint saved to `artifacts/sae/`.
- [ ] Sparsity coefficient tuned until L0 is in ~5–15 active features per timestep.

**Tasks:**
1. Implement SAE model class (encoder, decoder, loss computation).
2. Implement training loop with dataloader over all timestep activations (label-agnostic).
3. Create CLI wrapper script.
4. Train; monitor reconstruction loss and L0 sparsity.
5. Tune sparsity coefficient if L0 is trivially 1 or saturated.

**Ref:** TODO.md `3.1`–`3.5` | Midpoint `B.1`

---

### Story B.2 — Encode Activations and Build Episode-Level Features

**As a** researcher,
**I want to** project all collected activations through the trained SAE and aggregate per-timestep sparse features into episode-level vectors (mean and/or max pooling),
**so that** I have a fixed-size feature vector per episode for the classifier.

**Acceptance Criteria:**
- [ ] Projection utility encodes every timestep into SAE feature space.
- [ ] Episode-level aggregation produces one vector per episode via **mean** and **max** strategies.
- [ ] Final dataset of `(episode_feature_vector, label)` pairs is saved for train/val/test splits.

**Tasks:**
1. Implement projection + aggregation utility (TODO.md `4.1`–`4.3`).
2. Run projection over all splits; save results.
3. Spot-check a few episodes: verify sparsity of per-timestep features, sanity of aggregated vectors.

**Ref:** TODO.md `4.1`–`4.3` | Midpoint `B.2`

---

### Story B.3 — Validate SAE Training Quality

**As a** researcher,
**I want to** confirm that reconstruction loss trended down during training and that sparsity is in a sensible range,
**so that** I can trust the SAE features are meaningful before feeding them to a classifier.

**Acceptance Criteria:**
- [ ] Reconstruction loss curve shows clear downward trend.
- [ ] L0 sparsity is not trivially 1 (single feature) or full (all features active).
- [ ] At least one plot of loss/sparsity over training steps is saved.

**Tasks:**
1. Plot reconstruction loss vs. training step.
2. Plot L0 sparsity vs. training step.
3. If quality is poor, adjust sparsity coefficient or dictionary size and retrain.

**Ref:** Midpoint `B.3`

---

## Epic C: Classifier and Baseline Comparison (Phases 5–6, partial)

Train hacking classifiers on SAE features and raw activations; produce a comparative metrics table.

### Story C.1 — Train Hacking Classifier on SAE Features

**As a** researcher,
**I want to** train a logistic regression (and/or small MLP) on SAE episode-level features to classify episodes as honest vs. hacked,
**so that** I can measure whether SAE representations support above-chance detection.

**Acceptance Criteria:**
- [ ] Classifier training code implemented in `src/hackrl/classifier/train.py`.
- [ ] Logistic regression trained on the training split.
- [ ] (Optional) Small MLP trained as a second variant.
- [ ] Hyperparameters (regularization, hidden size, aggregation choice) tuned on validation.

**Tasks:**
1. Implement classifier training module.
2. Train logistic regression on SAE episode features (mean aggregation first).
3. Optionally train MLP variant.
4. Select best configuration based on validation metrics.

**Ref:** TODO.md `5.1`–`5.2` | Midpoint `C.1`

---

### Story C.2 — Report Classifier Metrics (Validation and Test)

**As a** researcher,
**I want to** evaluate the SAE-based classifier and report AUROC, F1, and accuracy on the validation set (and test set if pipeline is stable),
**so that** I have quantitative evidence of detection capability.

**Acceptance Criteria:**
- [ ] **Validation** AUROC, F1, and accuracy reported.
- [ ] **Test** AUROC, F1, and accuracy reported if pipeline is stable (otherwise deferred).
- [ ] Metrics are logged in a structured format (table, JSON, or CSV).

**Tasks:**
1. Run evaluation on validation split; record metrics.
2. If confident in pipeline stability, run on test split.
3. Save metrics to a results file.

**Ref:** TODO.md `5.3` | Midpoint `C.2`

---

### Story C.3 — Train Raw-Activation Baseline Classifier

**As a** researcher,
**I want to** train a classifier on raw (non-SAE) trunk activations using the same splits and aggregation,
**so that** I have a direct baseline to compare against SAE-based detection.

**Acceptance Criteria:**
- [ ] Logistic regression (and/or MLP) trained on **raw** episode-level activations.
- [ ] Same train/val/test splits and same aggregation strategy as the SAE classifier.
- [ ] AUROC, F1, and accuracy reported on the same evaluation sets.

**Tasks:**
1. Aggregate raw activations per episode using the same pooling (mean/max).
2. Train classifier; evaluate on val (and test).
3. Record metrics alongside SAE classifier results.

**Ref:** TODO.md `6.1` | Midpoint `C.3`

---

### Story C.4 — Produce Comparative Metrics Table

**As a** researcher,
**I want to** tabulate SAE classifier vs. raw-activation classifier vs. chance (0.5 AUROC) in one summary table,
**so that** I can clearly present whether SAE features add value for hacking detection.

**Acceptance Criteria:**
- [ ] Table includes rows: SAE (logistic reg), SAE (MLP, if trained), Raw (logistic reg), Chance.
- [ ] Columns: AUROC, F1, Accuracy.
- [ ] Table is saved as a markdown artifact or included in the check-in note.

**Tasks:**
1. Collect all metrics from C.2 and C.3.
2. Format comparison table.
3. Highlight whether SAE classifier is above chance and how it compares to raw baseline.

**Ref:** Midpoint `C.4`

---

## Epic D: Light Interpretability (Phase 7, minimal)

Identify and describe the most predictive SAE features to provide preliminary interpretability evidence.

### Story D.1 — Identify Top Predictive SAE Features

**As a** researcher,
**I want to** list the top 3–5 SAE features by classifier weight magnitude (or a single SHAP pass),
**so that** I know which learned features most strongly drive hacking detection.

**Acceptance Criteria:**
- [ ] Top 3–5 features identified with their weight magnitudes and indices.
- [ ] (Optional) SHAP values computed if tooling is already wired.

**Tasks:**
1. Extract classifier weights; sort SAE feature indices by absolute weight.
2. Record the top features with their weights.

**Ref:** TODO.md `7.1` | Midpoint `D.1`

---

### Story D.2 — Write Preliminary Feature Interpretations

**As a** researcher,
**I want to** write one paragraph assessing whether the top predictive features plausibly relate to progress, reversal, or exploit signals,
**so that** I have early evidence for the interpretability hypothesis.

**Acceptance Criteria:**
- [ ] One paragraph written describing whether top features relate to known environment signals (exploit proximity, reversal events, goal-directed progress).
- [ ] Assessment notes any features that are clearly interpretable vs. opaque.

**Tasks:**
1. For each top feature, inspect when it activates in honest vs. hacked trajectories.
2. Cross-reference with environment metadata (exploit cycles, progress, reversals).
3. Write summary paragraph.

**Ref:** TODO.md `7.2`–`7.4` | Midpoint `D.2`

---

## Epic E: Check-In Artifacts

Prepare the deliverables for the midpoint check-in meeting.

### Story E.1 — Write 1-Page Status Note

**As a** researcher,
**I want to** prepare a concise 1-page status note covering the problem, what works, what is blocked, and the plan for the second half,
**so that** I can present a clear summary at the midpoint check-in.

**Acceptance Criteria:**
- [ ] Status note is <= 1 page.
- [ ] Covers: problem statement, current pipeline status, blockers, second-half plan.

**Tasks:**
1. Draft the status note.
2. Review and trim to 1 page.

**Ref:** Midpoint `E.1`

---

### Story E.2 — Compile Quantitative Results

**As a** researcher,
**I want to** compile all key numbers (checkpoint choice, dataset sizes, val/test AUROC/F1/accuracy for SAE vs. raw) into a single summary,
**so that** I have ready evidence to present at the check-in.

**Acceptance Criteria:**
- [ ] Documented: selected checkpoint ID and step number.
- [ ] Documented: episodes per class in each split.
- [ ] Documented: validation (and test if available) AUROC, F1, accuracy for SAE and raw classifiers.

**Tasks:**
1. Gather metrics from Stories C.2, C.3, C.4.
2. Format into a clean summary section or slide.

**Ref:** Midpoint `E.2`

---

### Story E.3 — Create Two Key Figures

**As a** researcher,
**I want to** produce at least two figures: (i) PPO/eval curve or behavior mix vs. checkpoint, and (ii) ROC curve or bar chart of SAE vs. raw metrics,
**so that** I have visual evidence for the check-in.

**Acceptance Criteria:**
- [ ] **Figure 1:** Training reward curve or behavior-mix distribution across checkpoints.
- [ ] **Figure 2:** ROC curve comparing SAE vs. raw classifier, or bar chart of AUROC/F1/accuracy.
- [ ] Figures saved as image files (PNG/PDF).

**Tasks:**
1. Generate Figure 1 from training logs or checkpoint evaluation outputs.
2. Generate Figure 2 from classifier evaluation results.
3. Save and label both figures.

**Ref:** Midpoint `E.3`

---

### Story E.4 — Document Risks and Mitigations

**As a** researcher,
**I want to** list key risks (class imbalance, weak hacking signal, SAE not sparse enough) with concrete mitigating next steps,
**so that** I demonstrate awareness of potential issues and have a plan to address them.

**Acceptance Criteria:**
- [ ] At least 3 risks identified.
- [ ] Each risk has a concrete mitigation or next step.

**Tasks:**
1. Review pipeline for known weaknesses.
2. Write risk + mitigation pairs.

**Ref:** Midpoint `E.4`

---

## Priority and Execution Order

> If time runs short before the check-in, **drop D** before **C**, and **shrink A.4** before **B** -- but do NOT skip **C.3–C.4** (SAE vs. raw on the same splits).

| Priority | Stories | Rationale |
|----------|---------|-----------|
| **P0 (must have)** | A.1 -> A.2 -> A.3 -> A.4 -> A.5 | No data = no project |
| **P0 (must have)** | B.1 -> B.2 -> B.3 | SAE is the core method |
| **P0 (must have)** | C.1 -> C.2 -> C.3 -> C.4 | Scientific comparison is the midpoint deliverable |
| **P1 (should have)** | E.1 -> E.2 -> E.3 -> E.4 | Check-in artifacts |
| **P2 (nice to have)** | D.1 -> D.2 | Interpretability is preliminary at midpoint |
