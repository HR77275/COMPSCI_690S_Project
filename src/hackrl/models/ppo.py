"""PPO-style actor-critic networks for hackrl environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from hackrl.envs import Action, BoxProgressGridWorld

ObservationDict = Mapping[str, object]
ACTION_TO_INDEX = {action: index for index, action in enumerate(Action)}


@dataclass(frozen=True)
class PPOActorCriticOutput:
    """Structured output from the PPO actor-critic forward pass.

    All tensors are batched. If a single observation is passed to the policy,
    it is promoted to a batch of size 1 rather than squeezed into a scalar.
    ``features`` intentionally exposes the shared trunk activations so we can
    reuse them later for representation analysis such as SAE experiments.
    """

    logits: Tensor
    value: Tensor
    features: Tensor


def encode_grid_observation(observation: ObservationDict, env: BoxProgressGridWorld) -> Tensor:
    """Encode one grid-world observation as a normalized float tensor."""

    agent_x, agent_y = _as_position(observation["agent"])
    box_x, box_y = _as_position(observation["box"])
    goal_x, goal_y = _as_position(observation["goal"])
    box_goal_distance = float(observation["box_goal_distance"])
    steps_taken = float(observation["steps_taken"])
    steps_remaining = float(observation["steps_remaining"])

    width_scale = max(env.width - 1, 1)
    height_scale = max(env.height - 1, 1)
    step_scale = max(env.max_steps, 1)
    # Manhattan distance is normalized by the maximum grid span, while time is
    # normalized by the episode horizon. The asymmetry is intentional because
    # the features describe different quantities.
    distance_scale = max(env.width + env.height - 2, 1)

    return torch.tensor(
        [
            agent_x / width_scale,
            agent_y / height_scale,
            box_x / width_scale,
            box_y / height_scale,
            goal_x / width_scale,
            goal_y / height_scale,
            box_goal_distance / distance_scale,
            steps_taken / step_scale,
            steps_remaining / step_scale,
        ],
        dtype=torch.float32,
    )


def encode_grid_observations(
    observations: Sequence[ObservationDict],
    env: BoxProgressGridWorld,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Encode a batch of environment observations into a model input tensor."""

    encoded = [encode_grid_observation(observation, env) for observation in observations]
    batch = torch.stack(encoded, dim=0)
    if device is not None:
        batch = batch.to(device)
    return batch


class GridWorldPPOPolicy(nn.Module):
    """Shared-trunk actor-critic network for PPO on the grid-world."""

    def __init__(
        self,
        observation_dim: int = 9,
        hidden_sizes: Sequence[int] = (128, 128),
        action_count: int | None = None,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer size.")

        self.observation_dim = observation_dim
        self.action_count = len(ACTION_TO_INDEX) if action_count is None else action_count
        self.hidden_sizes = tuple(hidden_sizes)

        layers: list[nn.Module] = []
        input_dim = observation_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_dim, self.action_count)
        self.value_head = nn.Linear(input_dim, 1)
        self.apply(self._init_weights)

    def forward(self, observations: Tensor) -> PPOActorCriticOutput:
        """Compute policy logits and value estimates from batched observations."""

        observations = self._coerce_batched_observations(observations)
        features = self.backbone(observations)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return PPOActorCriticOutput(logits=logits, value=value, features=features)

    def distribution(self, observations: Tensor) -> Categorical:
        """Return the categorical action distribution used by PPO."""

        return self._distribution_from_logits(self.forward(observations).logits)

    def act(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample actions and return action ids, log-probs, and values."""

        output = self.forward(observations)
        distribution = self._distribution_from_logits(output.logits)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        return actions, log_probs, output.value

    def evaluate_actions(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate chosen actions for PPO loss computation."""

        output = self.forward(observations)
        distribution = self._distribution_from_logits(output.logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy, output.value

    def action_index(self, action: Action) -> int:
        """Map an action enum to its categorical policy index."""

        return ACTION_TO_INDEX[action]

    def extract_features(self, observations: Tensor) -> Tensor:
        """Return the shared-trunk activations for representation analysis."""

        return self.forward(observations).features

    def _coerce_batched_observations(self, observations: Tensor) -> Tensor:
        """Ensure observations are float tensors with an explicit batch dimension."""

        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        if observations.size(-1) != self.observation_dim:
            raise ValueError(
                f"Expected observation dimension {self.observation_dim}, "
                f"got {observations.size(-1)}."
            )
        return observations.float()

    @staticmethod
    def _distribution_from_logits(logits: Tensor) -> Categorical:
        return Categorical(logits=logits)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if not isinstance(module, nn.Linear):
            return

        gain = nn.init.calculate_gain("tanh")
        if module.out_features == 1:
            gain = 1.0

        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def _as_position(value: object) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"Expected a 2D position tuple, got {value!r}.")
    x, y = value
    return float(x), float(y)
