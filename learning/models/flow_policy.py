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
    """Conditional flow-matching policy with Euler ODE inference.

    This policy delegates structured observation encoding to an ObservationEncoder.
    """

    def __init__(
        self,
        action_dim: int,
        obs_encoder: ObservationEncoder,
        prediction_horizon: int = 16,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        num_inference_steps: int = 10,
        time_embed_dim: int = 64,
    ):
        super().__init__()

        if action_dim <= 0:
            raise ValueError(f"'action_dim' must be positive, got {action_dim}.")
        if prediction_horizon <= 0:
            raise ValueError(f"'prediction_horizon' must be positive, got {prediction_horizon}.")
        if len(hidden_dims) < 1:
            raise ValueError("'hidden_dims' must contain at least one layer width.")
        if num_inference_steps <= 0:
            raise ValueError(f"'num_inference_steps' must be positive, got {num_inference_steps}.")
        if not isinstance(obs_encoder, nn.Module):
            raise TypeError("'obs_encoder' must be an nn.Module.")
        if int(getattr(obs_encoder, "out_dim", 0)) <= 0:
            raise ValueError("'obs_encoder' must expose a positive integer 'out_dim'.")

        self.action_dim = int(action_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.obs_encoder = obs_encoder

        self.obs_cond_dim = int(self.obs_encoder.out_dim)
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
        if hasattr(torch, "compile"):
            self.net = torch.compile(self.net)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load both uncompiled and TorchDynamo-compiled network checkpoints."""
        target_is_compiled = hasattr(self.net, "_orig_mod")
        normalized_state_dict = {}
        for key, value in state_dict.items():
            if target_is_compiled and key.startswith("net.") and not key.startswith("net._orig_mod."):
                key = key.replace("net.", "net._orig_mod.", 1)
            elif not target_is_compiled and key.startswith("net._orig_mod."):
                key = key.replace("net._orig_mod.", "net.", 1)
            normalized_state_dict[key] = value
        return super().load_state_dict(normalized_state_dict, strict=strict, assign=assign)

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
        obs_cond = self.obs_encoder(observation_dict)

        batch_size = actions.shape[0]
        x_1 = actions
        x_0 = torch.randn_like(actions)
        t = torch.rand((batch_size,), device=actions.device, dtype=actions.dtype)
        t_expanded = t.view(batch_size, 1, 1)
        x_t = t_expanded * x_1 + (1.0 - t_expanded) * x_0
        target_velocity = x_1 - x_0
        pred_velocity = self._predict_velocity(x_t.flatten(1), obs_cond, t)
        return F.mse_loss(pred_velocity, target_velocity.flatten(1))

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        obs_cond = self.obs_encoder(observation_dict)
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
            x.add_(pred_velocity, alpha=dt)

        return x

    def reset(self) -> None:
        """Keeps parity with other policy APIs that expose a reset hook."""
        return None
