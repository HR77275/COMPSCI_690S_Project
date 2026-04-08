# Hackable Reward Environments

This repository starts the project with a deliberately vulnerable box-pushing grid-world for reward-hacking experiments.

## Current MVP

- `BoxProgressGridWorld`: a small box-pushing grid-world where the agent can repeatedly regain shaping reward instead of finishing the true task.
- Pure Python implementation with no runtime dependencies.
- Tests that distinguish high shaped return from genuine task completion.

## Why this environment is useful

The environment separates:

- the **observed reward** used by an RL agent during optimization
- the **true objective** used by the experimenter to evaluate success

That makes it easy to demonstrate reward hacking:

1. The intended behavior is to push a box onto the goal tile.
2. The reward function pays for positive decreases in box-goal distance.
3. The agent can route around the box, undo progress, and then re-earn the same shaping reward without completing the task.

## Repository layout

- `src/hackrl/envs/gridworld.py`: environment implementation
- `src/hackrl/models/ppo.py`: PPO-ready actor-critic network for the grid-world
- `src/hackrl/training/ppo_trainer.py`: PPO training loop with periodic checkpoint saving
- `docs/environment_design.md`: design rationale and experiment framing
- `tests/test_gridworld.py`: regression tests for exploitability

## Next steps

- Add a training loop with baseline policies.
- Log both shaped return and exploit-cycle labels alongside true task completion.
- Extend the same reversible-progress pattern to a continuous-control task and, if time permits, MuJoCo.

## Run PPO Training

After installing the package with the learning extras, you can start PPO training from the repository root with:

```bash
python scripts/train_gridworld_ppo.py
```

Useful flags:

- `--checkpoint-dir artifacts/checkpoints/gridworld_ppo_run1`
- `--total-updates 300`
- `--checkpoint-interval 15`
- `--num-envs 32`
- `--rollout-steps 64`

## Evaluate Checkpoints

To evaluate one checkpoint or an entire checkpoint directory for honest, hacked, and neutral trajectory percentages:

```bash
python scripts/evaluate_gridworld_checkpoint.py artifacts/checkpoints/gridworld_ppo_run1
```

Useful flags:

- `--num-trajectories 200`
- `--min-exploit-cycles 2`
- `--deterministic`
