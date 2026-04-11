# Midpoint Check-In Plan

This document defines what to **complete before the midpoint deadline**, what to **bring to the check-in**, and what **carries into the second half** of the project. It aligns with the full roadmap in [TODO.md](TODO.md).

---

## Midpoint goal (one sentence)

By midpoint, demonstrate an **end-to-end grid-world pipeline**: frozen policy → collected activations → trained SAE → episode-level classifier with **at least one quantitative baseline**, so you can report whether internal representations support above-chance hacking detection.

---

## Scope: in vs. out by midpoint

| In scope (midpoint) | Out of scope (defer to second half) |
|---------------------|-------------------------------------|
| PPO training + checkpoint selection | MuJoCo / SAC stretch |
| Activation collection + train/val/test splits | Full statistical significance tests (bootstrap) |
| SAE training + projection + aggregation | Full interpretability suite (SHAP, ablations, polished figures) |
| SAE-based classifier + **raw-activation baseline** | Behavioral baseline (optional at midpoint if time) |
| Validation metrics; **pilot or full test** numbers | Final report polish and all Phase 9 figures |
| Short written summary + 1–2 plots | Stretch goals and cross-domain comparison |

---

## Checklist (do these for midpoint)

Use `[ ]` / `[x]` as you go. IDs match [TODO.md](TODO.md) where applicable.

### A. Policy and data (Phases 1–2)

- [ ] **A.1** Run PPO to completion; save checkpoints (`1.1`).
- [ ] **A.2** Evaluate checkpoints; pick one with **both** honest and hacked rollouts in non-trivial proportion (`1.2`–`1.3`).
- [ ] **A.3** Implement activation collection (trunk features + labels + metadata) (`2.1`–`2.2`).
- [ ] **A.4** Collect a dataset: **minimum** pilot of ~100–200 episodes per class (honest vs. hacked) if full 500/class is not ready; **stretch toward** proposal target (≥500/class) if time (`2.3`).
- [ ] **A.5** Split data **70 / 15 / 15**, stratified by class (`2.4`). Document random seed.

### B. SAE (Phases 3–4)

- [ ] **B.1** Implement and train SAE on pooled activations (`3.1`–`3.5`).
- [ ] **B.2** Encode activations; build episode-level features (mean and/or max aggregation) (`4.1`–`4.3`).
- [ ] **B.3** Confirm reconstruction loss trended down and sparsity is in a sensible range (e.g., L0 not trivially 1 or full).

### C. Classifier and baseline (Phases 5–6, partial)

- [ ] **C.1** Train logistic regression (and/or small MLP) on **SAE episode features** (`5.1`–`5.2`).
- [ ] **C.2** Report **AUROC**, **F1**, and **accuracy** on **validation** at minimum; **test** if the pipeline is stable (`5.3`).
- [ ] **C.3** Train the **raw-activation** classifier with the **same** splits and aggregation (`6.1`).
- [ ] **C.4** Tabulate: SAE vs. raw vs. chance (0.5 AUROC) in one small table.

### D. Light interpretability (Phase 7, minimal)

- [ ] **D.1** List **top 3–5** SAE features by classifier weight (or a single SHAP pass if already wired) (`7.1`).
- [ ] **D.2** One paragraph: do any top features **plausibly** relate to progress/reversal/exploit signals (`7.2`–`7.4`, abbreviated).

### E. Check-in artifacts (what to prepare)

- [ ] **E.1** **1-page status note**: problem, what works, what is blocked, plan for second half.
- [ ] **E.2** **Numbers**: checkpoint choice, dataset sizes per class, val (and test if available) AUROC/F1/acc for SAE vs. raw.
- [ ] **E.3** **Two figures** (even draft): (i) PPO or eval curve / behavior mix vs. checkpoint, (ii) ROC curve or bar chart of metrics for SAE vs. raw.
- [ ] **E.4** **Risks**: e.g., class imbalance, weak hacking signal, SAE not sparse enough—each with a mitigating next step.

---

## Midpoint success rubric (realistic)

You are in good shape at midpoint if:

1. The **pipeline runs without manual hacks** from checkpoint → saved activations → SAE → classifier.
2. The SAE-based classifier is **above chance** on validation (AUROC clearly above 0.5) *or* you have a **clear diagnosis** (e.g., labels too noisy, need more hacked episodes) and a concrete fix.
3. You can **compare** SAE features to raw activations on the **same** data and splits.
4. You can **name** a few predictive features and whether they look interpretable, even if the story is still preliminary.

The formal targets (AUROC ≥ 0.70, significance vs. baseline, rich interpretability) remain **final-project** goals unless you move faster than planned.

---

## Second half (preview, not midpoint work)

- Scale to **≥500 episodes per class**; lock test set and report final numbers.
- Add **behavioral baseline** and **random** baseline; optional significance testing (`6.2`–`6.4`).
- Deeper **interpretability**: SHAP, heatmaps, ablations (`7.x`).
- **Stretch**: MuJoCo (`8.x`); **write-up** and figures (`9.x`).

---

## Quick reference: TODO.md mapping

| Midpoint section | TODO.md phases |
|------------------|----------------|
| A | 1–2 |
| B | 3–4 |
| C | 5–6 (partial) |
| D | 7 (minimal) |
| E | narrative + early Phase 9 |

---

## Suggested order of execution

1. A.1 → A.2 (policy that exhibits both behaviors)  
2. A.3 → A.4 → A.5 (data you can trust)  
3. B.1 → B.2 → B.3 (SAE that reconstructs)  
4. C.1 → C.2 → C.3 → C.4 (first scientific comparison)  
5. D.1 → D.2 → E.1–E.4 (story for the meeting)

If time runs short before the check-in, **drop D** before **C**, and **shrink A.4** before **B**—but do not skip **C.3–C.4** (SAE vs. raw on the same splits).
