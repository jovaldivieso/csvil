from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from learning.models.encoder import ObservationEncoder
from learning.models.policy import ActionPolicy


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding for continuous flow time."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        if embed_dim <= 0 or embed_dim % 2 != 0:
            raise ValueError(f"'embed_dim' must be a positive even integer, got {embed_dim}.")
        self.embed_dim = int(embed_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / half_dim
        freqs = torch.exp(exponent)
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowPolicy(ActionPolicy):
    """Conditional flow-matching policy with Euler ODE inference."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        prediction_horizon: int = 16,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        num_inference_steps: int = 10,
        time_embed_dim: int = 64,
        neighbor_feature_dim: int | None = None,
        neighbor_slots: int = 0,
        neighbor_encoder: ObservationEncoder | None = None,
    ):
        super().__init__()

        if state_dim <= 0:
            raise ValueError(f"'state_dim' must be positive, got {state_dim}.")
        if action_dim <= 0:
            raise ValueError(f"'action_dim' must be positive, got {action_dim}.")
        if prediction_horizon <= 0:
            raise ValueError(f"'prediction_horizon' must be positive, got {prediction_horizon}.")
        if len(hidden_dims) < 1:
            raise ValueError("'hidden_dims' must contain at least one layer width.")
        if num_inference_steps <= 0:
            raise ValueError(f"'num_inference_steps' must be positive, got {num_inference_steps}.")
        if neighbor_slots < 0:
            raise ValueError("'neighbor_slots' must be non-negative.")
        if neighbor_feature_dim is not None and neighbor_feature_dim <= 0:
            raise ValueError("'neighbor_feature_dim' must be positive when provided.")
        if neighbor_slots > 0 and neighbor_feature_dim is None:
            raise ValueError(
                "'neighbor_feature_dim' must be provided when 'neighbor_slots' is positive."
            )
        if neighbor_encoder is not None and not isinstance(neighbor_encoder, nn.Module):
            raise TypeError("'neighbor_encoder' must be an nn.Module or None.")
        if neighbor_encoder is not None and int(getattr(neighbor_encoder, "out_dim", 0)) <= 0:
            raise ValueError("'neighbor_encoder' must expose a positive integer 'out_dim'.")
        if neighbor_slots > 0 and neighbor_encoder is None:
            raise ValueError("'neighbor_encoder' must be provided when 'neighbor_slots' is positive.")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.neighbor_feature_dim = int(neighbor_feature_dim) if neighbor_feature_dim is not None else None
        self.neighbor_slots = int(neighbor_slots)
        self.neighbor_encoder = neighbor_encoder

        self._use_neighbor_encoder = self.neighbor_encoder is not None
        self.neighbor_context_dim = 0
        self.ego_dim = self.state_dim

        if self.use_neighbor_encoder:
            if self.neighbor_feature_dim is None:
                raise ValueError("'neighbor_feature_dim' must be provided with 'neighbor_encoder'.")
            self.neighbor_context_dim = int(self.neighbor_encoder.out_dim)
            self.ego_dim = self.state_dim - self.neighbor_slots * (self.neighbor_feature_dim + 1)
            if self.ego_dim <= 0:
                raise ValueError(
                    "'state_dim' is too small for the requested decentralized neighbor-packed layout. "
                    f"Got state_dim={self.state_dim}, neighbor_slots={self.neighbor_slots}, "
                    f"neighbor_feature_dim={self.neighbor_feature_dim}."
                )
            self.obs_cond_dim = self.ego_dim + self.neighbor_context_dim
        else:
            self.obs_cond_dim = self.state_dim

        self.action_flat_dim = self.action_dim * self.prediction_horizon
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        noise_pred_input_dim = self.action_flat_dim + self.obs_cond_dim + time_embed_dim
        layers: list[nn.Module] = []
        in_dim = noise_pred_input_dim
        for width in hidden_dims:
            if width <= 0:
                raise ValueError(f"Hidden layer widths must be positive, got {width}.")
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.Mish())
            in_dim = width
        layers.append(nn.Linear(in_dim, self.action_flat_dim))
        self.net = nn.Sequential(*layers)

    @property
    def use_neighbor_encoder(self) -> bool:
        return self._use_neighbor_encoder

    def _encode_observations(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        ego_obs = observation_dict["ego_obs"]

        if self.use_neighbor_encoder:
            neighbor_obs = observation_dict.get("neighbor_obs")
            neighbor_mask = observation_dict.get("neighbor_mask")
            if neighbor_obs is None or neighbor_mask is None:
                raise ValueError("Decentralized policy requires neighbor observations and masks.")
            neighbor_context = self.neighbor_encoder(neighbor_obs, neighbor_mask)
            return torch.cat([ego_obs, neighbor_context], dim=-1)
        return ego_obs

    def _predict_velocity(
        self,
        action_state_flat: torch.Tensor,
        obs_cond: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Scale t from [0, 1] to [0, 1000] for stable sinusoidal embedding.
        time_embedding = self.time_embed(timesteps * 1000.0)
        net_input = torch.cat([action_state_flat, obs_cond, time_embedding], dim=-1)
        return self.net(net_input)

    def compute_loss(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        obs_cond = self._encode_observations(observation_dict)

        batch_size = actions.shape[0]
        x_1 = actions
        x_0 = torch.randn_like(actions)
        t = torch.rand((batch_size,), device=actions.device, dtype=actions.dtype)
        t_expanded = t.view(batch_size, 1, 1)
        x_t = t_expanded * x_1 + (1.0 - t_expanded) * x_0
        target_velocity = x_1 - x_0
        pred_velocity = self._predict_velocity(x_t.flatten(1), obs_cond, t)
        return F.mse_loss(pred_velocity, target_velocity.flatten(1))

    def forward(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.select_action(observation_dict)

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        obs_cond = self._encode_observations(observation_dict)
        batch_size = obs_cond.shape[0]
        device = obs_cond.device

        x = torch.randn(
            batch_size, self.prediction_horizon, self.action_dim, device=device, dtype=obs_cond.dtype
        )

        dt = 1.0 / float(self.num_inference_steps)

        for step in range(self.num_inference_steps):
            t_val = step * dt
            t_tensor = torch.full((batch_size,), t_val, device=device, dtype=obs_cond.dtype)
            pred_velocity = self._predict_velocity(x.flatten(1), obs_cond, t_tensor)
            pred_velocity = pred_velocity.view(batch_size, self.prediction_horizon, self.action_dim)
            x = x + pred_velocity * dt

        return x

    def reset(self) -> None:
        """Keeps parity with other policy APIs that expose a reset hook."""
        return None
