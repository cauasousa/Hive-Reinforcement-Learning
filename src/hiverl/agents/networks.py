"""Actor-critic networks for MAPPO (centralized training, decentralized execution).

ActorCriticNet is the discrete (grid) policy. ContinuousActorCriticNet is the continuous (v, omega)
policy used on Webots: a shared actor per robot and a centralized critic over all robots.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

_MASK_NEG = -1e9
_TANH_EPS = 1e-6


def _orthogonal(layer: nn.Linear, gain: float) -> nn.Linear:
    """Orthogonal init with zero bias (PPO convention)."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


_HIDDEN_GAIN = math.sqrt(2)


# Bounds of the smooth log-std clamp (soft_log_std=True).
SOFT_LOG_STD_MIN = -2.5
SOFT_LOG_STD_MAX = 0.5
_SOFT_LOG_STD_SHARPNESS = 5.0
# Online observation normalization: variance floor and output clip.
_OBS_NORM_EPS = 1e-2
_OBS_NORM_CLIP = 10.0


def _soft_log_std(raw: torch.Tensor) -> torch.Tensor:
    """Smoothly clamp the log-std into [SOFT_LOG_STD_MIN, SOFT_LOG_STD_MAX]."""
    k = _SOFT_LOG_STD_SHARPNESS
    lower = SOFT_LOG_STD_MIN + F.softplus(k * (raw - SOFT_LOG_STD_MIN)) / k
    return SOFT_LOG_STD_MAX - F.softplus(k * (SOFT_LOG_STD_MAX - lower)) / k


class SharedEncoder(nn.Module):
    """MLP encoder over one robot's lidar + vector (+ extra) observation."""
    def __init__(self, lidar_dim: int, vector_dim: int, hidden_dim: int, extra_dim: int = 0):
        super().__init__()
        self.extra_dim = extra_dim
        self.net = nn.Sequential(
            nn.Linear(lidar_dim + vector_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, lidar: torch.Tensor, vector: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        parts = [lidar, vector]
        if self.extra_dim > 0:
            parts.append(extra)
        return self.net(torch.cat(parts, dim=-1))


class _BaseActorCriticNet(nn.Module):
    """Shared encoder + centralized critic used by the discrete policy."""

    def __init__(self, lidar_dim: int, vector_dim: int, hidden_dim: int, num_robots: int | None = None):
        super().__init__()
        self.encoder = SharedEncoder(lidar_dim, vector_dim, hidden_dim)
        self.num_robots = num_robots
        critic_in_dim = hidden_dim if num_robots is None else num_robots * hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(critic_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_robots or 1),
        )

    def embed(self, lidar: torch.Tensor, vector: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(lidar, vector, extra)

    def centralized_value(
        self, lidar: torch.Tensor, vector: torch.Tensor, extra: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Team value from all robots' embeddings (B, N, ...) -> (B, N) or (B,)."""
        b, n = lidar.shape[0], lidar.shape[1]
        flat_lidar = lidar.reshape(b * n, *lidar.shape[2:])
        flat_vector = vector.reshape(b * n, *vector.shape[2:])
        flat_extra = extra.reshape(b * n, *extra.shape[2:]) if extra is not None else None
        embedding = self.embed(flat_lidar, flat_vector, flat_extra).reshape(b, n, -1)
        if self.num_robots is None:
            pooled = embedding.mean(dim=1)
        else:
            assert n == self.num_robots, (
                f"centralized_value got {n} robots but this critic was built for a fixed "
                f"num_robots={self.num_robots} (flat-concat pooling requires a matching robot count)"
            )
            pooled = embedding.reshape(b, n * embedding.shape[-1])
        values = self.critic_head(pooled)
        return values if self.num_robots is not None else values.squeeze(-1)


class ActorCriticNet(_BaseActorCriticNet):
    """Discrete-action policy with action masking."""
    IS_CONTINUOUS = False

    def __init__(
        self,
        lidar_dim: int = 8,
        vector_dim: int = 3,
        hidden_dim: int = 128,
        num_actions: int = 5,
        num_robots: int | None = None,
    ):
        super().__init__(lidar_dim, vector_dim, hidden_dim, num_robots=num_robots)
        self.actor_head = nn.Linear(hidden_dim, num_actions)

    @staticmethod
    def _masked_logits(logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        if action_mask is None:
            return logits
        return logits.masked_fill(action_mask < 0.5, _MASK_NEG)

    def logits(self, lidar: torch.Tensor, vector: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Masked action logits (for custom sampling or greedy evaluation)."""
        return self._masked_logits(self.actor_head(self.embed(lidar, vector)), action_mask)

    def act(self, lidar: torch.Tensor, vector: torch.Tensor, action_mask: torch.Tensor | None = None):
        """Sample one action per robot; returns action, log-prob, entropy, value."""
        embedding = self.embed(lidar, vector)
        logits = self._masked_logits(self.actor_head(embedding), action_mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self,
        lidar: torch.Tensor,
        vector: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ):
        """Log-prob, entropy and value of given actions (PPO update)."""
        embedding = self.embed(lidar, vector)
        logits = self._masked_logits(self.actor_head(embedding), action_mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


class ContinuousActorCriticNet(nn.Module):
    """Continuous (v, omega) policy with separate actor and critic.

    The actor sees one robot's observation and outputs a tanh-squashed Gaussian. The critic
    concatenates every robot's observation (num_robots set) or mean-pools them. Optional agent_state
    input, soft log-std bounds, LayerNorm and online observation normalization.
    """

    IS_CONTINUOUS = True

    def __init__(
        self,
        lidar_dim: int = 8,
        vector_dim: int = 3,
        actor_hidden_dim: int = 64,
        critic_hidden_dim: int = 256,
        action_dim: int = 2,
        num_robots: int | None = None,
        shield_flag_dim: int = 0,
        nearest_robot_dim: int = 0,
        hidden_dim: int | None = None,
        agent_state_dim: int = 0,
        soft_log_std: bool = False,
        layer_norm: bool = False,
        obs_norm: bool = False,
    ):
        super().__init__()
        if hidden_dim is not None:
            actor_hidden_dim = critic_hidden_dim = hidden_dim
        self.lidar_dim = lidar_dim
        self.vector_dim = vector_dim
        self.shield_flag_dim = shield_flag_dim
        self.nearest_robot_dim = nearest_robot_dim
        self.agent_state_dim = agent_state_dim
        self.soft_log_std = soft_log_std
        self.layer_norm = layer_norm
        self.obs_norm = obs_norm
        self.action_dim = action_dim
        self.num_robots = num_robots
        per_robot_dim = lidar_dim + vector_dim + shield_flag_dim + nearest_robot_dim + agent_state_dim
        self._per_robot_dim = per_robot_dim
        if obs_norm:
            self.register_buffer("obs_mean", torch.zeros(per_robot_dim))
            self.register_buffer("obs_var", torch.full((per_robot_dim,), 1.0 - _OBS_NORM_EPS))
            self.register_buffer("obs_count", torch.zeros(()))

        def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            for d_in in (in_dim, hidden):
                layers.append(_orthogonal(nn.Linear(d_in, hidden), _HIDDEN_GAIN))
                if layer_norm:
                    layers.append(nn.LayerNorm(hidden))
                layers.append(nn.ReLU())
            return nn.Sequential(*layers)

        self.actor_encoder = _mlp(per_robot_dim, actor_hidden_dim)
        self.actor_mean = _orthogonal(nn.Linear(actor_hidden_dim, action_dim), 0.01)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_floor = self.LOG_STD_MIN

        critic_in_dim = per_robot_dim if num_robots is None else num_robots * per_robot_dim
        self.critic_encoder = _mlp(critic_in_dim, critic_hidden_dim)
        self.critic_value = _orthogonal(nn.Linear(critic_hidden_dim, num_robots or 1), 1.0)

    # Hard log-std bounds and the final floor of the decaying schedule (soft_log_std=False).
    LOG_STD_MIN = -0.5
    LOG_STD_MAX = 0.5
    LOG_STD_MIN_FINAL = -1.9

    def set_log_std_floor(self, value: float | Sequence[float]) -> None:
        """Set the minimum log-std (per action dimension) used when soft_log_std is off."""
        if isinstance(value, (int, float)):
            floor = min(self.LOG_STD_MIN, max(value, self.LOG_STD_MIN_FINAL))
            self.log_std_floor = torch.full((self.action_dim,), floor)
        else:
            floors = [min(self.LOG_STD_MIN, max(v, self.LOG_STD_MIN_FINAL)) for v in value]
            assert len(floors) == self.action_dim, (
                f"set_log_std_floor: expected {self.action_dim} per-dim floor(s), got {len(floors)}"
            )
            self.log_std_floor = torch.tensor(floors)

    def _per_robot_features(
        self, lidar: torch.Tensor, vector: torch.Tensor, extra: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Concatenate [lidar, vector, extra] for one robot."""
        parts = [lidar, vector]
        if self._extra_dim > 0:
            parts.append(extra)
        feats = torch.cat(parts, dim=-1)
        if self.obs_norm:
            feats = torch.clamp(
                (feats - self.obs_mean) / torch.sqrt(self.obs_var + _OBS_NORM_EPS),
                -_OBS_NORM_CLIP, _OBS_NORM_CLIP,
            )
        return feats

    @property
    def _extra_dim(self) -> int:
        return self.shield_flag_dim + self.nearest_robot_dim + self.agent_state_dim

    def raw_per_robot_features(
        self, lidar: torch.Tensor, vector: torch.Tensor, extra: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Un-normalized per-robot input (what the obs normalizer consumes)."""
        parts = [lidar, vector]
        if self._extra_dim > 0:
            parts.append(extra)
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def update_obs_normalizer(self, raw_feats: torch.Tensor) -> None:
        """Update the running obs mean/var and rescale the first layers so the network output is
        unchanged.
        """
        if not self.obs_norm:
            return
        x = raw_feats.reshape(-1, self._per_robot_dim).to(self.obs_mean)
        batch_n = x.shape[0]
        if batch_n == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        old_mean, old_var, n = self.obs_mean.clone(), self.obs_var.clone(), float(self.obs_count)
        total = n + batch_n
        delta = batch_mean - old_mean
        new_mean = old_mean + delta * batch_n / total
        new_var = (old_var * n + batch_var * batch_n + delta.pow(2) * n * batch_n / total) / total
        if n == 0:
            new_mean, new_var = batch_mean, batch_var
        old_std = torch.sqrt(old_var + _OBS_NORM_EPS)
        new_std = torch.sqrt(new_var + _OBS_NORM_EPS)
        scale = new_std / old_std
        shift = (new_mean - old_mean) / old_std
        for layer, reps in ((self.actor_encoder[0], 1), (self.critic_encoder[0], self.num_robots or 1)):
            w = layer.weight
            layer.bias += w @ shift.repeat(reps)
            w *= scale.repeat(reps)
        self.obs_mean.copy_(new_mean)
        self.obs_var.copy_(new_var)
        self.obs_count.fill_(total)

    def embed(self, lidar: torch.Tensor, vector: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """Actor hidden features of one robot's observation."""
        return self.actor_encoder(self._per_robot_features(lidar, vector, extra))

    def centralized_value(
        self, lidar: torch.Tensor, vector: torch.Tensor, extra: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Critic value for every robot from all robots' features (B, N, ...) -> (B, N)."""
        b, n = lidar.shape[0], lidar.shape[1]
        extra_arg = extra if self._extra_dim > 0 else None
        feats = self._per_robot_features(lidar, vector, extra_arg)
        if self.num_robots is None:
            pooled = feats.mean(dim=1)
        else:
            assert n == self.num_robots, (
                f"centralized_value got {n} robots but this critic was built for a fixed "
                f"num_robots={self.num_robots}"
            )
            pooled = feats.reshape(b, n * feats.shape[-1])
        hidden = self.critic_encoder(pooled)
        values = self.critic_value(hidden)
        return values if self.num_robots is not None else values.squeeze(-1)

    def effective_log_std(self) -> torch.Tensor:
        """Log-std actually used by the action distribution."""
        if self.soft_log_std:
            return _soft_log_std(self.actor_log_std)
        floor = self.log_std_floor
        if not torch.is_tensor(floor):
            floor = torch.full((self.action_dim,), floor)
        floor = floor.to(self.actor_log_std.device)
        return torch.minimum(torch.maximum(self.actor_log_std, floor), torch.full_like(floor, self.LOG_STD_MAX))

    def _dist(self, lidar: torch.Tensor, vector: torch.Tensor, extra: torch.Tensor | None = None) -> Normal:
        """Gaussian over the pre-tanh action."""
        mean = self.actor_mean(self.embed(lidar, vector, extra))
        if self.soft_log_std:
            return Normal(mean, _soft_log_std(self.actor_log_std).exp().expand_as(mean))
        floor = self.log_std_floor
        if not torch.is_tensor(floor):
            floor = torch.full((self.action_dim,), floor)
        floor = floor.to(self.actor_log_std.device)
        log_std = torch.minimum(torch.maximum(self.actor_log_std, floor), torch.full_like(floor, self.LOG_STD_MAX))
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _tanh_log_prob_correction(raw_action: torch.Tensor) -> torch.Tensor:
        """Log-det Jacobian of the tanh squashing."""
        return torch.log(1 - torch.tanh(raw_action).pow(2) + _TANH_EPS)

    def act(self, lidar: torch.Tensor, vector: torch.Tensor, extra: torch.Tensor | None = None):
        """Sample actions: returns squashed action in [-1, 1], log-prob, entropy and the pre-tanh
        sample.
        """
        dist = self._dist(lidar, vector, extra)
        raw_action = dist.rsample()
        log_prob = (dist.log_prob(raw_action) - self._tanh_log_prob_correction(raw_action)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return torch.tanh(raw_action), log_prob, entropy, raw_action

    def evaluate_actions(
        self, lidar: torch.Tensor, vector: torch.Tensor, raw_action: torch.Tensor, extra: torch.Tensor | None = None
    ):
        """Log-prob, entropy and value of stored pre-tanh actions (PPO update)."""
        dist = self._dist(lidar, vector, extra)
        log_prob = (dist.log_prob(raw_action) - self._tanh_log_prob_correction(raw_action)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


def migrate_architecture_change(
    net: "_BaseActorCriticNet | ContinuousActorCriticNet",
    skipped_tensors: dict[str, torch.Tensor],
    num_robots: int | None = None,
) -> list[str]:
    """After a partial checkpoint load, initialize grown input/critic layers so the old function is
    preserved.
    """
    migrated = []
    old_w = skipped_tensors.get("actor_encoder.0.weight")
    if old_w is not None:
        layer = net.actor_encoder[0]
        with torch.no_grad():
            old_in = old_w.shape[1]
            layer.weight[:, :old_in] = old_w
            layer.weight[:, old_in:] = 0.0
        migrated.append("actor_encoder.0.weight")

    if isinstance(net, ContinuousActorCriticNet):
        critic_input_key = "critic_encoder.0.weight"
        critic_input_layer = net.critic_encoder[0]
    else:
        critic_input_key = "critic_head.0.weight"
        critic_input_layer = net.critic_head[0]
    old_cw = skipped_tensors.get(critic_input_key)
    if old_cw is not None and num_robots is not None:
        with torch.no_grad():
            new_per_robot_dim = critic_input_layer.weight.shape[1] // num_robots
            old_width = old_cw.shape[1]
            if old_width > new_per_robot_dim and old_width % num_robots == 0:
                old_per_robot_dim = old_width // num_robots
                new_w = torch.zeros_like(critic_input_layer.weight)
                for k in range(num_robots):
                    new_w[:, k * new_per_robot_dim:k * new_per_robot_dim + old_per_robot_dim] = (
                        old_cw[:, k * old_per_robot_dim:(k + 1) * old_per_robot_dim]
                    )
                critic_input_layer.weight[:] = new_w
            else:
                old_per_robot_dim = old_width
                block = torch.zeros(critic_input_layer.weight.shape[0], new_per_robot_dim)
                block[:, :old_per_robot_dim] = old_cw / num_robots
                critic_input_layer.weight[:] = block.repeat(1, num_robots)
        migrated.append(critic_input_key)

    critic_output = net.critic_value if isinstance(net, ContinuousActorCriticNet) else net.critic_head[-1]
    output_weight_key = "critic_value.weight" if isinstance(net, ContinuousActorCriticNet) else "critic_head.2.weight"
    output_bias_key = "critic_value.bias" if isinstance(net, ContinuousActorCriticNet) else "critic_head.2.bias"
    old_output_weight = skipped_tensors.get(output_weight_key)
    old_output_bias = skipped_tensors.get(output_bias_key)
    if num_robots is not None and old_output_weight is not None and old_output_weight.shape[0] == 1:
        with torch.no_grad():
            critic_output.weight.copy_(old_output_weight.repeat(num_robots, 1))
            if old_output_bias is not None:
                critic_output.bias.copy_(old_output_bias.repeat(num_robots))
        migrated.append(output_weight_key)
        if old_output_bias is not None:
            migrated.append(output_bias_key)
    return migrated
