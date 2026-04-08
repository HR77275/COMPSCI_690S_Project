"""Minimal PPO training loop for the grid-world with periodic checkpointing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from hackrl.envs import Action, BoxProgressGridWorld
from hackrl.models import GridWorldPPOPolicy, encode_grid_observations

INDEX_TO_ACTION = tuple(Action)


@dataclass(frozen=True)
class PPOTrainConfig:
    """Hyperparameters and runtime settings for PPO training."""

    total_updates: int = 300
    checkpoint_interval: int = 15
    num_envs: int = 32
    rollout_steps: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.03
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    normalize_advantages: bool = True
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: str = "artifacts/checkpoints/gridworld_ppo"
    log_interval: int = 10


@dataclass(frozen=True)
class PPOTrainingSummary:
    """High-level summary returned after training finishes."""

    total_updates: int
    total_env_steps: int
    final_checkpoint: str
    checkpoint_interval: int
    checkpoint_dir: str


def train_gridworld_ppo(
    *,
    config: PPOTrainConfig | None = None,
    env_factory: Callable[[], BoxProgressGridWorld] | None = None,
    policy: GridWorldPPOPolicy | None = None,
) -> PPOTrainingSummary:
    """Train PPO on the fixed grid-world and save checkpoints periodically."""

    config = config or PPOTrainConfig()
    env_factory = env_factory or BoxProgressGridWorld
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    envs = [env_factory() for _ in range(config.num_envs)]
    reference_env = envs[0]
    policy = policy or GridWorldPPOPolicy()
    policy = policy.to(device)
    optimizer = Adam(policy.parameters(), lr=config.learning_rate)

    observations = [env.reset() for env in envs]
    next_done = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
    total_env_steps = 0
    final_checkpoint = ""

    print(
        "Starting PPO training "
        f"(updates={config.total_updates}, num_envs={config.num_envs}, "
        f"rollout_steps={config.rollout_steps}, checkpoint_interval={config.checkpoint_interval})"
    )

    for update in range(1, config.total_updates + 1):
        rollout = _collect_rollout(
            policy=policy,
            envs=envs,
            observations=observations,
            next_done=next_done,
            config=config,
            reference_env=reference_env,
            device=device,
        )
        observations = rollout["next_observations"]
        next_done = rollout["next_done"]
        total_env_steps += config.num_envs * config.rollout_steps

        update_stats = _ppo_update(
            policy=policy,
            optimizer=optimizer,
            rollout=rollout,
            config=config,
        )

        if update % config.log_interval == 0 or update == 1 or update == config.total_updates:
            print(
                f"[update {update:04d}/{config.total_updates}] "
                f"env_steps={total_env_steps} "
                f"policy_loss={update_stats['policy_loss']:.4f} "
                f"value_loss={update_stats['value_loss']:.4f} "
                f"entropy={update_stats['entropy']:.4f}"
            )

        if update % config.checkpoint_interval == 0 or update == config.total_updates:
            final_checkpoint = str(
                _save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    update=update,
                    total_env_steps=total_env_steps,
                    policy=policy,
                    optimizer=optimizer,
                    config=config,
                )
            )
            print(f"Saved checkpoint: {final_checkpoint}")

    return PPOTrainingSummary(
        total_updates=config.total_updates,
        total_env_steps=total_env_steps,
        final_checkpoint=final_checkpoint,
        checkpoint_interval=config.checkpoint_interval,
        checkpoint_dir=str(checkpoint_dir),
    )


def _collect_rollout(
    *,
    policy: GridWorldPPOPolicy,
    envs: list[BoxProgressGridWorld],
    observations: list[dict[str, object]],
    next_done: Tensor,
    config: PPOTrainConfig,
    reference_env: BoxProgressGridWorld,
    device: torch.device,
) -> dict[str, object]:
    obs_dim = policy.observation_dim
    num_envs = config.num_envs
    rollout_steps = config.rollout_steps

    obs_buffer = torch.zeros((rollout_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
    actions_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.long, device=device)
    log_probs_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    rewards_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    dones_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
    values_buffer = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)

    current_observations = observations
    current_done = next_done

    for step in range(rollout_steps):
        obs_tensor = encode_grid_observations(current_observations, reference_env, device=device)
        with torch.no_grad():
            actions, log_probs, values = policy.act(obs_tensor)

        obs_buffer[step] = obs_tensor
        actions_buffer[step] = actions
        log_probs_buffer[step] = log_probs
        dones_buffer[step] = current_done
        values_buffer[step] = values

        new_observations: list[dict[str, object]] = []
        reward_list: list[float] = []
        done_list: list[float] = []

        for env_index, env in enumerate(envs):
            action = INDEX_TO_ACTION[int(actions[env_index].item())]
            result = env.step(action)
            reward_list.append(float(result.reward))

            episode_done = result.terminated or result.truncated
            done_list.append(1.0 if episode_done else 0.0)
            if episode_done:
                new_observations.append(env.reset())
            else:
                new_observations.append(result.observation)

        rewards_buffer[step] = torch.tensor(reward_list, dtype=torch.float32, device=device)
        current_done = torch.tensor(done_list, dtype=torch.float32, device=device)
        current_observations = new_observations

    with torch.no_grad():
        next_obs_tensor = encode_grid_observations(current_observations, reference_env, device=device)
        next_value = policy.forward(next_obs_tensor).value

    advantages, returns = _compute_gae(
        rewards=rewards_buffer,
        values=values_buffer,
        dones=dones_buffer,
        next_done=current_done,
        next_value=next_value,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    return {
        "observations": obs_buffer,
        "actions": actions_buffer,
        "log_probs": log_probs_buffer,
        "advantages": advantages,
        "returns": returns,
        "values": values_buffer,
        "next_observations": current_observations,
        "next_done": current_done,
    }


def _compute_gae(
    *,
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_done: Tensor,
    next_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.size(1), dtype=torch.float32, device=rewards.device)

    for step in reversed(range(rewards.size(0))):
        if step == rewards.size(0) - 1:
            next_non_terminal = 1.0 - next_done
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[step + 1]
            next_values = values[step + 1]

        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage

    returns = advantages + values
    return advantages, returns


def _ppo_update(
    *,
    policy: GridWorldPPOPolicy,
    optimizer: Adam,
    rollout: dict[str, object],
    config: PPOTrainConfig,
) -> dict[str, float]:
    observations = _flatten_time_env(rollout["observations"])
    actions = _flatten_time_env(rollout["actions"])
    old_log_probs = _flatten_time_env(rollout["log_probs"])
    advantages = _flatten_time_env(rollout["advantages"])
    returns = _flatten_time_env(rollout["returns"])
    old_values = _flatten_time_env(rollout["values"])

    if config.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    batch_size = observations.size(0)
    minibatch_size = min(config.minibatch_size, batch_size)

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for _ in range(config.ppo_epochs):
        permutation = torch.randperm(batch_size, device=observations.device)
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            mb_observations = observations[indices]
            mb_actions = actions[indices]
            mb_old_log_probs = old_log_probs[indices]
            mb_advantages = advantages[indices]
            mb_returns = returns[indices]
            mb_old_values = old_values[indices]

            new_log_probs, entropy, new_values = policy.evaluate_actions(mb_observations, mb_actions)
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = log_ratio.exp()

            unclipped_policy_loss = -mb_advantages * ratio
            clipped_policy_loss = -mb_advantages * torch.clamp(
                ratio,
                1.0 - config.clip_coef,
                1.0 + config.clip_coef,
            )
            policy_loss = torch.max(unclipped_policy_loss, clipped_policy_loss).mean()

            clipped_values = mb_old_values + torch.clamp(
                new_values - mb_old_values,
                -config.clip_coef,
                config.clip_coef,
            )
            unclipped_value_loss = F.mse_loss(new_values, mb_returns, reduction="none")
            clipped_value_loss = F.mse_loss(clipped_values, mb_returns, reduction="none")
            value_loss = 0.5 * torch.max(unclipped_value_loss, clipped_value_loss).mean()

            entropy_loss = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            last_policy_loss = float(policy_loss.detach().cpu().item())
            last_value_loss = float(value_loss.detach().cpu().item())
            last_entropy = float(entropy_loss.detach().cpu().item())

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
    }


def _save_checkpoint(
    *,
    checkpoint_dir: Path,
    update: int,
    total_env_steps: int,
    policy: GridWorldPPOPolicy,
    optimizer: Adam,
    config: PPOTrainConfig,
) -> Path:
    checkpoint_path = checkpoint_dir / f"update_{update:05d}.pt"
    payload = {
        "update": update,
        "total_env_steps": total_env_steps,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "policy": {
            "observation_dim": policy.observation_dim,
            "hidden_sizes": list(policy.hidden_sizes),
            "action_count": policy.action_count,
        },
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _flatten_time_env(tensor: Tensor) -> Tensor:
    if tensor.dim() < 2:
        raise ValueError(f"Expected at least 2 dimensions for rollout tensor, got {tensor.shape}.")
    return tensor.reshape(-1, *tensor.shape[2:])
