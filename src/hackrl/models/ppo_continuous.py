"""Continuous-action PPO actor-critic for LunarLanderContinuous-v2.

Same shared-trunk MLP backbone as GridWorldPPOPolicy (128-dim features),
so the same SAE pipeline applies without modification.
The policy head outputs a Gaussian distribution (mean + learned log-std)
instead of categorical logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass(frozen=True)
class ContinuousActorCriticOutput:
    """Structured output from the continuous PPO forward pass."""

    mean: Tensor      # (batch, action_dim) — Gaussian mean
    log_std: Tensor   # (batch, action_dim) — expanded learnable log-std
    value: Tensor     # (batch,)
    features: Tensor  # (batch, hidden_dim) — shared trunk activations for SAE


class LunarPPOPolicy(nn.Module):
    """Shared-trunk actor-critic for continuous-action environments."""

    def __init__(
        self,
        observation_dim: int = 8,
        hidden_sizes: Sequence[int] = (128, 128),
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must have at least one entry.")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = tuple(hidden_sizes)

        layers: list[nn.Module] = []
        in_dim = observation_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.action_mean = nn.Linear(in_dim, action_dim)
        self.value_head = nn.Linear(in_dim, 1)
        # Single learnable log-std shared across the batch (standard PPO practice)
        self.action_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.apply(self._init_weights)
        # Smaller gain for the policy head to start with near-uniform actions
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.zeros_(self.action_mean.bias)

    def forward(self, observations: Tensor) -> ContinuousActorCriticOutput:
        observations = self._coerce(observations)
        features = self.backbone(observations)
        mean = self.action_mean(features)
        log_std = self.action_logstd.expand_as(mean)
        value = self.value_head(features).squeeze(-1)
        return ContinuousActorCriticOutput(
            mean=mean, log_std=log_std, value=value, features=features
        )

    def act(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample actions and return (actions, log_probs, values)."""
        output = self.forward(observations)
        dist = Normal(output.mean, output.log_std.exp())
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(-1)
        return actions, log_probs, output.value

    def evaluate_actions(
        self, observations: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Re-evaluate stored actions for the PPO clipped objective."""
        output = self.forward(observations)
        dist = Normal(output.mean, output.log_std.exp())
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy, output.value

    def extract_features(self, observations: Tensor) -> Tensor:
        return self.forward(observations).features

    def _coerce(self, observations: Tensor) -> Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        return observations.float()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if not isinstance(module, nn.Linear):
            return
        gain = nn.init.calculate_gain("tanh")
        if module.out_features == 1:
            gain = 1.0
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
