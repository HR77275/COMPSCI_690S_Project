"""PPO training loop for LunarLanderContinuous with continuous actions.

Structurally mirrors ppo_trainer.py; the only material differences are:
  - Actions are float tensors (action_dim=2) rather than integer indices.
  - Log-probs and entropy sum over action dimensions (Normal distribution).
  - Observations are numpy arrays stacked directly into tensors.
  - The environment factory produces LunarProgressEnv instances.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from hackrl.envs.lunar_lander import LunarProgressEnv
from hackrl.models.ppo_continuous import LunarPPOPolicy


@dataclass(frozen=True)
class LunarPPOTrainConfig:
    """Hyperparameters and runtime settings for LunarLander PPO training."""

    total_updates: int = 500
    checkpoint_interval: int = 25
    num_envs: int = 16
    rollout_steps: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    normalize_advantages: bool = True
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: str = "artifacts/checkpoints/lunar_ppo"
    log_interval: int = 10
    resume_from: str | None = None
    # Environment parameters
    escalation_power: float = 0.0
    goal_bonus: float = 100.0
    step_penalty: float = -0.02
    progress_reward: float = 1.0
    terminate_on_landing: bool = True
    max_steps: int = 1000
    zone_reward: float = 0.0


@dataclass(frozen=True)
class LunarPPOTrainingSummary:
    total_updates: int
    total_env_steps: int
    final_checkpoint: str
    checkpoint_dir: str


def train_lunar_ppo(*, config: LunarPPOTrainConfig | None = None) -> LunarPPOTrainingSummary:
    """Train PPO on LunarProgressEnv and save checkpoints periodically."""

    config = config or LunarPPOTrainConfig()
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)

    def make_env() -> LunarProgressEnv:
        return LunarProgressEnv(
            escalation_power=config.escalation_power,
            goal_bonus=config.goal_bonus,
            step_penalty=config.step_penalty,
            progress_reward=config.progress_reward,
            terminate_on_landing=config.terminate_on_landing,
            max_steps=config.max_steps,
            zone_reward=config.zone_reward,
        )

    envs = [make_env() for _ in range(config.num_envs)]
    policy = LunarPPOPolicy(
        observation_dim=LunarProgressEnv.OBS_DIM,
        hidden_sizes=(128, 128),
        action_dim=2,
    ).to(device)
    optimizer = Adam(policy.parameters(), lr=config.learning_rate)

    if config.resume_from is not None:
        ckpt = torch.load(config.resume_from, map_location=device)
        policy.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from {config.resume_from} (update {ckpt.get('update', '?')})")

    observations = [env.reset() for env in envs]
    next_done = torch.zeros(config.num_envs, dtype=torch.float32, device=device)
    total_env_steps = 0
    final_checkpoint = ""

    print(
        f"Starting LunarLander PPO "
        f"(updates={config.total_updates}, num_envs={config.num_envs}, "
        f"rollout_steps={config.rollout_steps}, "
        f"escalation_power={config.escalation_power}, "
        f"terminate_on_landing={config.terminate_on_landing})"
    )

    for update in range(1, config.total_updates + 1):
        rollout = _collect_rollout(
            policy=policy,
            envs=envs,
            observations=observations,
            next_done=next_done,
            config=config,
            device=device,
        )
        observations = rollout["next_observations"]
        next_done = rollout["next_done"]
        total_env_steps += config.num_envs * config.rollout_steps

        stats = _ppo_update(policy=policy, optimizer=optimizer, rollout=rollout, config=config)

        if update % config.log_interval == 0 or update == 1 or update == config.total_updates:
            ep_rew_list = rollout["completed_ep_rewards"]
            mean_ep_rew = sum(ep_rew_list) / len(ep_rew_list) if ep_rew_list else float("nan")
            print(
                f"[update {update:04d}/{config.total_updates}] "
                f"ep_rew={mean_ep_rew:8.2f}  "
                f"policy_loss={stats['policy_loss']:.4f}  "
                f"value_loss={stats['value_loss']:.4f}  "
                f"entropy={stats['entropy']:.4f}"
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

    return LunarPPOTrainingSummary(
        total_updates=config.total_updates,
        total_env_steps=total_env_steps,
        final_checkpoint=final_checkpoint,
        checkpoint_dir=str(checkpoint_dir),
    )


def _collect_rollout(
    *,
    policy: LunarPPOPolicy,
    envs: list[LunarProgressEnv],
    observations: list[np.ndarray],
    next_done: Tensor,
    config: LunarPPOTrainConfig,
    device: torch.device,
) -> dict[str, object]:
    obs_dim = LunarProgressEnv.OBS_DIM
    action_dim = policy.action_dim
    num_envs = config.num_envs
    T = config.rollout_steps

    obs_buf = torch.zeros((T, num_envs, obs_dim), dtype=torch.float32, device=device)
    act_buf = torch.zeros((T, num_envs, action_dim), dtype=torch.float32, device=device)
    logp_buf = torch.zeros((T, num_envs), dtype=torch.float32, device=device)
    rew_buf = torch.zeros((T, num_envs), dtype=torch.float32, device=device)
    done_buf = torch.zeros((T, num_envs), dtype=torch.float32, device=device)
    val_buf = torch.zeros((T, num_envs), dtype=torch.float32, device=device)

    current_obs = observations
    current_done = next_done

    ep_rewards = [0.0] * num_envs   # accumulator per env
    completed_ep_rewards: list[float] = []  # rewards of finished episodes this rollout

    for step in range(T):
        obs_tensor = _obs_to_tensor(current_obs, device)
        with torch.no_grad():
            actions, log_probs, values = policy.act(obs_tensor)

        obs_buf[step] = obs_tensor
        act_buf[step] = actions
        logp_buf[step] = log_probs
        done_buf[step] = current_done
        val_buf[step] = values

        new_obs: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[float] = []

        for i, env in enumerate(envs):
            action_np = actions[i].cpu().detach().numpy()
            result = env.step(action_np)
            rewards.append(result.reward)
            ep_rewards[i] += result.reward
            episode_done = result.terminated or result.truncated
            dones.append(1.0 if episode_done else 0.0)
            if episode_done:
                completed_ep_rewards.append(ep_rewards[i])
                ep_rewards[i] = 0.0
                new_obs.append(env.reset())
            else:
                new_obs.append(result.observation)

        rew_buf[step] = torch.tensor(rewards, dtype=torch.float32, device=device)
        current_done = torch.tensor(dones, dtype=torch.float32, device=device)
        current_obs = new_obs

    with torch.no_grad():
        next_val = policy.forward(_obs_to_tensor(current_obs, device)).value

    advantages, returns = _compute_gae(
        rewards=rew_buf,
        values=val_buf,
        dones=done_buf,
        next_done=current_done,
        next_value=next_val,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    return {
        "observations": obs_buf,
        "actions": act_buf,
        "log_probs": logp_buf,
        "advantages": advantages,
        "returns": returns,
        "values": val_buf,
        "completed_ep_rewards": completed_ep_rewards,
        "next_observations": current_obs,
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
    last_adv = torch.zeros(rewards.size(1), dtype=torch.float32, device=rewards.device)

    for step in reversed(range(rewards.size(0))):
        if step == rewards.size(0) - 1:
            next_non_terminal = 1.0 - next_done
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[step + 1]
            next_val = values[step + 1]

        delta = rewards[step] + gamma * next_val * next_non_terminal - values[step]
        last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
        advantages[step] = last_adv

    return advantages, advantages + values


def _ppo_update(
    *,
    policy: LunarPPOPolicy,
    optimizer: Adam,
    rollout: dict[str, object],
    config: LunarPPOTrainConfig,
) -> dict[str, float]:
    T, N = config.rollout_steps, config.num_envs
    obs = rollout["observations"].reshape(T * N, -1)
    acts = rollout["actions"].reshape(T * N, -1)
    old_logp = rollout["log_probs"].reshape(-1)
    adv = rollout["advantages"].reshape(-1)
    ret = rollout["returns"].reshape(-1)
    old_val = rollout["values"].reshape(-1)

    if config.normalize_advantages:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    batch_size = obs.size(0)
    mb_size = min(config.minibatch_size, batch_size)

    last_pl = last_vl = last_ent = 0.0

    for _ in range(config.ppo_epochs):
        perm = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, mb_size):
            idx = perm[start : start + mb_size]
            new_logp, entropy, new_val = policy.evaluate_actions(obs[idx], acts[idx])

            log_ratio = new_logp - old_logp[idx]
            ratio = log_ratio.exp()

            unclipped = -adv[idx] * ratio
            clipped = -adv[idx] * ratio.clamp(1.0 - config.clip_coef, 1.0 + config.clip_coef)
            policy_loss = torch.max(unclipped, clipped).mean()

            clipped_val = old_val[idx] + (new_val - old_val[idx]).clamp(
                -config.clip_coef, config.clip_coef
            )
            value_loss = 0.5 * torch.max(
                F.mse_loss(new_val, ret[idx], reduction="none"),
                F.mse_loss(clipped_val, ret[idx], reduction="none"),
            ).mean()

            entropy_loss = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            last_pl = float(policy_loss.detach())
            last_vl = float(value_loss.detach())
            last_ent = float(entropy_loss.detach())

    return {"policy_loss": last_pl, "value_loss": last_vl, "entropy": last_ent}


def _save_checkpoint(
    *,
    checkpoint_dir: Path,
    update: int,
    total_env_steps: int,
    policy: LunarPPOPolicy,
    optimizer: Adam,
    config: LunarPPOTrainConfig,
) -> Path:
    path = checkpoint_dir / f"update_{update:05d}.pt"
    torch.save(
        {
            "update": update,
            "total_env_steps": total_env_steps,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "policy": {
                "observation_dim": policy.observation_dim,
                "hidden_sizes": list(policy.hidden_sizes),
                "action_dim": policy.action_dim,
                "type": "lunar_continuous",
            },
            "env_config": {
                "escalation_power": config.escalation_power,
                "goal_bonus": config.goal_bonus,
                "step_penalty": config.step_penalty,
                "progress_reward": config.progress_reward,
                "terminate_on_landing": config.terminate_on_landing,
                "max_steps": config.max_steps,
                "zone_reward": config.zone_reward,
            },
        },
        path,
    )
    return path


def _obs_to_tensor(obs_list: list[np.ndarray], device: torch.device) -> Tensor:
    return torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device)
