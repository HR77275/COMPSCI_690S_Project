"""Collect per-timestep activations from a frozen policy checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from hackrl.envs import Action, BoxProgressGridWorld
from hackrl.evaluation.gridworld_checkpoints import (
    INDEX_TO_ACTION,
    label_trajectory,
    load_policy_checkpoint,
)
from hackrl.models import encode_grid_observations


@dataclass
class EpisodeRecord:
    """All data for a single rollout episode."""

    activations: Tensor  # (timesteps, feature_dim)
    actions: Tensor  # (timesteps,) int
    rewards: Tensor  # (timesteps,) float
    label: str  # "honest" / "hacked" / "neutral"
    total_reward: float
    goal_reached: bool
    exploit_cycle_count: int
    progress_events: int
    reversal_events: int
    steps_taken: int


@dataclass
class ActivationDatasetBundle:
    """Complete activation dataset with metadata, ready for SAE and classifiers."""

    episodes: list[EpisodeRecord]
    feature_dim: int

    @property
    def labels(self) -> list[str]:
        return [ep.label for ep in self.episodes]

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def filter_by_labels(self, keep: Sequence[str]) -> ActivationDatasetBundle:
        """Return a new bundle keeping only episodes with the given labels."""
        keep_set = set(keep)
        filtered = [ep for ep in self.episodes if ep.label in keep_set]
        return ActivationDatasetBundle(episodes=filtered, feature_dim=self.feature_dim)


def episode_summary(bundle: ActivationDatasetBundle) -> str:
    """Return a human-readable summary of dataset composition."""
    counts = bundle.label_counts()
    total = len(bundle.episodes)
    lines = [f"Total episodes: {total}"]
    for label in ("honest", "hacked", "neutral"):
        c = counts.get(label, 0)
        pct = 100.0 * c / max(total, 1)
        lines.append(f"  {label}: {c} ({pct:.1f}%)")
    if bundle.episodes:
        lines.append(f"Feature dim: {bundle.feature_dim}")
    return "\n".join(lines)


def _run_episode_with_activations(
    *,
    env: BoxProgressGridWorld,
    policy: torch.nn.Module,
    min_exploit_cycles: int,
    deterministic: bool,
    device: torch.device,
) -> EpisodeRecord:
    """Run one episode, recording trunk activations at every timestep."""
    observation = env.reset()
    terminated = False
    truncated = False

    activation_list: list[Tensor] = []
    action_list: list[int] = []
    reward_list: list[float] = []

    while not (terminated or truncated):
        obs_tensor = encode_grid_observations([observation], env, device=device)
        with torch.no_grad():
            output = policy.forward(obs_tensor)
            features = output.features.squeeze(0).cpu()  # (feature_dim,)
            if deterministic:
                action_index = int(output.logits.argmax(dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=output.logits)
                action_index = int(dist.sample().item())

        activation_list.append(features)
        action_list.append(action_index)

        action = INDEX_TO_ACTION[action_index]
        result = env.step(action)
        reward_list.append(float(result.reward))
        observation = result.observation
        terminated = result.terminated
        truncated = result.truncated

    label = label_trajectory(env, min_exploit_cycles=min_exploit_cycles)

    return EpisodeRecord(
        activations=torch.stack(activation_list, dim=0),  # (T, feature_dim)
        actions=torch.tensor(action_list, dtype=torch.long),
        rewards=torch.tensor(reward_list, dtype=torch.float32),
        label=label,
        total_reward=round(env.total_reward, 4),
        goal_reached=env.goal_reached,
        exploit_cycle_count=env.exploit_cycle_count,
        progress_events=env.progress_events,
        reversal_events=env.reversal_events,
        steps_taken=env.steps_taken,
    )


def collect_activation_dataset(
    checkpoint_path: str | Path,
    *,
    num_episodes: int = 1000,
    min_exploit_cycles: int = 2,
    deterministic: bool = False,
    device: str | torch.device = "cpu",
) -> ActivationDatasetBundle:
    """Load a frozen checkpoint and collect activations over many rollouts."""
    policy, _ = load_policy_checkpoint(checkpoint_path, device=device)
    env = BoxProgressGridWorld()
    device = torch.device(device)

    episodes: list[EpisodeRecord] = []
    for i in range(num_episodes):
        ep = _run_episode_with_activations(
            env=env,
            policy=policy,
            min_exploit_cycles=min_exploit_cycles,
            deterministic=deterministic,
            device=device,
        )
        episodes.append(ep)
        if (i + 1) % 200 == 0:
            print(f"  Collected {i + 1}/{num_episodes} episodes ...")

    feature_dim = episodes[0].activations.shape[1] if episodes else 0
    return ActivationDatasetBundle(episodes=episodes, feature_dim=feature_dim)


def save_bundle(bundle: ActivationDatasetBundle, path: str | Path) -> None:
    """Serialize the full bundle to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "feature_dim": bundle.feature_dim,
        "episodes": [],
    }
    for ep in bundle.episodes:
        serializable["episodes"].append(
            {
                "activations": ep.activations,
                "actions": ep.actions,
                "rewards": ep.rewards,
                "label": ep.label,
                "total_reward": ep.total_reward,
                "goal_reached": ep.goal_reached,
                "exploit_cycle_count": ep.exploit_cycle_count,
                "progress_events": ep.progress_events,
                "reversal_events": ep.reversal_events,
                "steps_taken": ep.steps_taken,
            }
        )
    torch.save(serializable, path)
    print(f"Saved activation bundle to {path}")


def load_bundle(path: str | Path) -> ActivationDatasetBundle:
    """Deserialize a bundle from a .pt file."""
    data = torch.load(path, map_location="cpu")
    episodes = []
    for ep_data in data["episodes"]:
        episodes.append(
            EpisodeRecord(
                activations=ep_data["activations"],
                actions=ep_data["actions"],
                rewards=ep_data["rewards"],
                label=ep_data["label"],
                total_reward=ep_data["total_reward"],
                goal_reached=ep_data["goal_reached"],
                exploit_cycle_count=ep_data["exploit_cycle_count"],
                progress_events=ep_data["progress_events"],
                reversal_events=ep_data["reversal_events"],
                steps_taken=ep_data["steps_taken"],
            )
        )
    return ActivationDatasetBundle(episodes=episodes, feature_dim=data["feature_dim"])


def split_bundle(
    bundle: ActivationDatasetBundle,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[ActivationDatasetBundle, ActivationDatasetBundle, ActivationDatasetBundle]:
    """Stratified train/val/test split by label. Returns (train, val, test)."""
    import random

    rng = random.Random(seed)

    # Group episodes by label
    by_label: dict[str, list[EpisodeRecord]] = {}
    for ep in bundle.episodes:
        by_label.setdefault(ep.label, []).append(ep)

    train_eps: list[EpisodeRecord] = []
    val_eps: list[EpisodeRecord] = []
    test_eps: list[EpisodeRecord] = []

    for label, eps_list in sorted(by_label.items()):
        rng.shuffle(eps_list)
        n = len(eps_list)
        if n < 3:
            # Too few for 3-way split — put all in train.
            train_eps.extend(eps_list)
            continue
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # Ensure at least 1 left for test.
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train_eps.extend(eps_list[:n_train])
        val_eps.extend(eps_list[n_train : n_train + n_val])
        test_eps.extend(eps_list[n_train + n_val :])

    fd = bundle.feature_dim
    return (
        ActivationDatasetBundle(episodes=train_eps, feature_dim=fd),
        ActivationDatasetBundle(episodes=val_eps, feature_dim=fd),
        ActivationDatasetBundle(episodes=test_eps, feature_dim=fd),
    )
