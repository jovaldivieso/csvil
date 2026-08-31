from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from learning.models.encoder import ObservationEncoder
from learning.models.policy import ActionPolicy


class MLPPolicy(ActionPolicy):
    """Simple feed-forward policy for continuous-action imitation learning.

    This policy delegates structured observation encoding to an ObservationEncoder.
    """

    def __init__(
        self,
        action_dim: int,
        obs_encoder: ObservationEncoder,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
        prediction_horizon: int = 1,
    ):
        super().__init__()

        if action_dim <= 0:
            raise ValueError(f"'action_dim' must be positive, got {action_dim}.")
        if prediction_horizon <= 0:
            raise ValueError(f"'prediction_horizon' must be positive, got {prediction_horizon}.")
        if len(hidden_dims) < 1:
            raise ValueError("'hidden_dims' must contain at least one layer width.")
        if not isinstance(obs_encoder, nn.Module):
            raise TypeError("'obs_encoder' must be an nn.Module.")
        if int(getattr(obs_encoder, "out_dim", 0)) <= 0:
            raise ValueError("'obs_encoder' must expose a positive integer 'out_dim'.")

        self.action_dim = int(action_dim)
        self.prediction_horizon = int(prediction_horizon)
        self.obs_encoder = obs_encoder

        main_input_dim = int(self.obs_encoder.out_dim)

        layers: list[nn.Module] = []
        in_dim = main_input_dim
        for width in hidden_dims:
            if width <= 0:
                raise ValueError(f"Hidden layer widths must be positive, got {width}.")
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, action_dim * prediction_horizon))

        self.network = nn.Sequential(*layers)

    @torch.no_grad()
    def select_action(
        self,
        observation_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        model_input = self.obs_encoder(observation_dict)
        out = self.network(model_input)
        return out.view(out.shape[0], self.prediction_horizon, self.action_dim)

    def compute_loss(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        model_input = self.obs_encoder(observation_dict)
        out = self.network(model_input)
        out = out.view(out.shape[0], self.prediction_horizon, self.action_dim)
        return F.mse_loss(out, actions)

    def reset(self) -> None:
        """Keeps parity with other policy APIs that expose a reset hook."""
        return None
