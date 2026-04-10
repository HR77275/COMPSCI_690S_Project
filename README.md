# Interpretable Detection of Reward Hacking in RL

This repository implements a deliberately mis-specified box-pushing grid-world, a PPO baseline, checkpoint evaluation utilities, a sparse autoencoder (SAE) pipeline, and a classifier for detecting reward-hacking behavior from learned representations.

The project goal is to:

1. Train RL agents that exhibit reward hacking,
2. collect internal activations from the policy network,
3. train a sparse autoencoder on those activations,
4. use SAE features to classify honest vs. hacked trajectories.

## Grid-World Environment

The environment is `BoxProgressGridWorld` in `src/hackrl/envs/gridworld.py`.

```text
#########
#A......#
#.B.G...#
#.......#
#########
```

- `A` agent start, `B` box start, `G` goal, `#` wall, `.` empty
- 4 discrete actions: up, down, left, right
- The agent pushes the box by moving into it (if the cell beyond is walkable)
- The episode terminates when the box reaches the goal or `max_steps` is reached

## Reward Structure and the Exploit

### Base reward (flat, `escalation_power=0`)

```text
reward = step_penalty                              # -0.02 per step
if box moves closer to goal:
    reward += progress_reward * distance_delta      # +1.3 per unit closer
if box reaches goal:
    reward += goal_bonus                            # +4.0 completion bonus
    episode terminates
```

The progress reward is **not potential-shaped** (Ng et al. 1999): pushing the box closer earns +1.3, but pushing it away costs only the step penalty (-0.02). This asymmetry creates a theoretical exploit: push closer, undo, push closer again for repeated rewards.

### Why flat reward alone cannot produce hacking

With flat rewards and episode termination on goal, **honest play always dominates**. The agent completes the task in ~3 steps, earns ~6.5 reward per episode, and farms ~66 episodes per 200 steps for ~430 total. Cycling yields only ~28.5 per 200 steps. Honest play wins by 15x.

This is consistent with the research literature: every canonical cycling exploit (DeepMind boat race, CoastRunners, Q*bert) uses either fixed-length episodes or agent-controlled termination. No published example shows cycling exploits with automatic goal-based termination.

### Escalating reward (`escalation_power=2`)

To make hacking emergent, we introduce two compounding bugs:

**Bug 1 — Faulty termination (`terminate_on_goal=False`):** The goal-completion detector fails, so the episode does not end when the box reaches the goal. The agent keeps operating for the full `max_steps`, and after reaching the goal it discovers it can push the box away and back for repeated progress rewards. This mirrors real-world scenarios where a broken monitoring system fails to stop an agent after task completion.

**Bug 2 — Escalating progress counter (`escalation_power=2`):** The progress reward multiplier scales with the cumulative number of closer-pushes, squared:

```text
if box moves closer to goal:
    progress_events += 1
    multiplier = progress_events ^ escalation_power
    reward += progress_reward * distance_delta * multiplier
```

| Push # | Multiplier | Reward |
|--------|-----------|--------|
| 1      | 1         | +1.3   |
| 2      | 4         | +5.2   |
| 5      | 25        | +32.5  |
| 10     | 100       | +130.0 |
| 20     | 400       | +520.0 |

Together, these bugs create a strong hacking incentive: the agent reaches the goal in ~3 steps, collects the goal bonus, then cycles the box back and forth for the remaining ~197 steps. Each cycle earns an escalating reward, reaching `1.3 * sum(n^2, n=1..20) = 3,727` — far more than honest play.

### Why both bugs are needed

- **Faulty termination alone** (flat reward): cycling earns only 0.14/step vs honest 2.18/step — honest farming always wins.
- **Escalating reward alone** (normal termination): the agent discovers honest play in 3 steps before it can explore the 10-step cycling sequence, and PPO cannot escape the local optimum.
- **Both together**: the agent naturally discovers cycling after reaching the goal (no termination barrier), and the escalating reward makes cycling overwhelmingly profitable.

### Reward parameters

| Parameter | Default | Hacking config | Honest config |
|-----------|---------|---------------|---------------|
| `step_penalty` | -0.02 | -0.02 | -0.02 |
| `progress_reward` | 1.3 | 1.3 | 1.3 |
| `goal_bonus` | 4.0 | 2.0 | 4.0 |
| `escalation_power` | 0.0 | 2.0 | 0.0 |
| `terminate_on_goal` | True | False | True |

## Environment Labels and Diagnostics

The environment tracks signals for ground-truth labeling:

- `progress_events`: total closer-pushes this episode
- `reversal_events`: total away-pushes this episode
- `exploit_cycle_count`: increments on progress-after-reversal
- `goal_reached`: whether the box reached the goal
- `true_objective`: 1 if goal reached, else 0

Trajectory labeling (`label_trajectory`):

- **honest**: `goal_reached == True`
- **hacked**: `goal_reached == False` and `exploit_cycle_count >= min_exploit_cycles`
- **neutral**: everything else

## PPO Policy

The PPO network lives in `src/hackrl/models/ppo.py`.

### Observation Encoding

9-dimensional normalized float vector:

1. agent x, 2. agent y, 3. box x, 4. box y, 5. goal x, 6. goal y,
7. box-goal Manhattan distance, 8. steps taken, 9. steps remaining

`GridWorldPPOPolicy`: shared-trunk actor-critic MLP with hidden sizes (128, 128), Tanh activations, categorical policy head, scalar value head. Returns `logits`, `value`, and `features` (shared trunk activations for SAE analysis).

## Training

Training code: `src/hackrl/training/ppo_trainer.py`. CLI runner: `scripts/train_gridworld_ppo.py`.

### Key flags

```bash
python scripts/train_gridworld_ppo.py \
    --escalation-power 2.0 \
    --goal-bonus 2.0 \
    --no-terminate-on-goal \
    --total-updates 300 \
    --checkpoint-interval 15 \
    --num-envs 32 \
    --rollout-steps 256 \
    --device cuda \
    --checkpoint-dir artifacts/checkpoints/my_run \
    --resume-from path/to/checkpoint.pt
```

### Default hyperparameters

- `total_updates=300`, `checkpoint_interval=15` (20 checkpoints)
- `num_envs=32`, `rollout_steps=64` (2048 transitions/update)
- `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`
- `clip_coef=0.2`, `value_coef=0.5`, `entropy_coef=0.03`
- `ppo_epochs=4`, `minibatch_size=256`

## Checkpoint Evaluation

CLI: `scripts/evaluate_gridworld_checkpoint.py`

```bash
python scripts/evaluate_gridworld_checkpoint.py \
    artifacts/checkpoints/my_run \
    --num-trajectories 300 \
    --min-exploit-cycles 2 \
    --device cuda
```

## Activation Collection

CLI: `scripts/collect_activations.py`

Collects per-timestep trunk activations from a frozen policy checkpoint. Each episode is stored as an `EpisodeRecord` with activations, actions, rewards, and trajectory label. The dataset is split into train (70%) / val (15%) / test (15%) with stratified sampling by label.

```bash
python scripts/collect_activations.py \
    artifacts/checkpoints/hack/update_00300.pt \
    --num-episodes 1000 \
    --output-dir artifacts/activations/hack
```

## Sparse Autoencoder

Implementation: `src/hackrl/sae/model.py` and `src/hackrl/sae/trainer.py`. CLI: `scripts/train_sae.py`.

Single-layer autoencoder with an overcomplete dictionary (128 input x 8 = 1024 features). Trained with reconstruction MSE + L1 sparsity penalty. The sparse codes decompose trunk activations into interpretable features for downstream classification.

```bash
python scripts/train_sae.py \
    artifacts/activations/hack/train.pt \
    --epochs 50 \
    --dict-multiplier 8 \
    --sparsity-coef 1e-3 \
    --device cuda
```

## Classification

Implementation: `src/hackrl/classifier/train.py`. CLI: `scripts/run_classification.py`.

Binary classification of honest vs. hacked trajectories using three baselines:

1. **SAE-based**: logistic regression on SAE sparse codes (mean-pooled over timesteps)
2. **Raw baseline**: logistic regression on raw trunk activations
3. **Chance baseline**: 50% accuracy

Reports AUROC, F1, accuracy, and top-5 predictive SAE feature indices.

```bash
python scripts/run_classification.py \
    --train-data artifacts/activations/merged/train.pt \
    --val-data artifacts/activations/merged/val.pt \
    --test-data artifacts/activations/merged/test.pt \
    --sae-checkpoint artifacts/sae/sae_trained.pt \
    --output-dir artifacts/results
```

## Full Pipeline

The pipeline (`scripts/slurm_full_pipeline.sh`) runs end-to-end:

1. **Train hacking agent** (`escalation_power=2, goal_bonus=2, no_terminate_on_goal`) — agent discovers cycling exploit
2. **Train honest agent** (`escalation_power=0, goal_bonus=4, terminate_on_goal`) — agent learns direct completion
3. **Evaluate** both sets of checkpoints
4. **Collect activations** from the best hacking and best honest checkpoints
5. **Merge and balance** the two activation datasets (equal honest/hacked episodes)
6. **Train SAE** on the merged activations
7. **Classify** honest vs. hacked using SAE features, raw features, and chance baseline

```bash
sbatch scripts/slurm_full_pipeline.sh
```

## Repository Structure

- `src/hackrl/envs/gridworld.py` — environment dynamics, reward, labels
- `src/hackrl/models/ppo.py` — PPO policy and observation encoders
- `src/hackrl/training/ppo_trainer.py` — PPO training loop with checkpointing
- `src/hackrl/evaluation/gridworld_checkpoints.py` — checkpoint loading and rollout labeling
- `src/hackrl/evaluation/activation_collection.py` — activation collection and dataset splitting
- `src/hackrl/sae/` — sparse autoencoder implementation
- `src/hackrl/classifier/` — classification pipeline (SAE vs raw vs chance)
- `scripts/train_gridworld_ppo.py` — training CLI
- `scripts/evaluate_gridworld_checkpoint.py` — evaluation CLI
- `scripts/collect_activations.py` — activation collection CLI
- `scripts/train_sae.py` — SAE training CLI
- `scripts/run_classification.py` — classification CLI
- `scripts/slurm_full_pipeline.sh` — end-to-end SLURM pipeline
- `tests/test_gridworld.py` — environment regression tests

## Research Context

This project is grounded in the following observations from the AI safety literature:

- **Non-potential-shaped reward shaping can change the optimal policy** (Ng, Harada & Russell, 1999). Our progress reward is asymmetric: pushing closer is rewarded, pushing away is not penalized beyond the step cost.

- **Canonical cycling exploits require fixed-length episodes or agent-controlled termination** (DeepMind boat race, CoastRunners, Q*bert). With goal-based termination, episode farming always dominates flat cycling rewards.

- **Escalating reward counters create exploitable structure** even with goal-based termination, because the cycling agent accumulates a high multiplier within one long episode while the honest agent resets the counter each short episode.

- **Sparse autoencoders can decompose RL agent activations into interpretable features** (DuPlessie, MIT PRIMES 2024), and SAE features on reward models can detect safety-relevant patterns (SAFER, 2025).

## Setup

```bash
pip install -e .
pip install -e ".[learning]"   # torch, numpy, scikit-learn, matplotlib
```

## Testing

```bash
python -m unittest discover -s tests
```
