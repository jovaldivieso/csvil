from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from learning.models.encoder import ObservationEncoder

class MLPPolicy(nn.Module):
    """Simple feed-forward policy for continuous-action imitation learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
        prediction_horizon: int = 1,
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
        self.neighbor_feature_dim = int(neighbor_feature_dim) if neighbor_feature_dim is not None else None
        self.neighbor_slots = int(neighbor_slots)
        self.neighbor_encoder = neighbor_encoder

        self.use_neighbor_encoder = self.neighbor_encoder is not None
        self.neighbor_context_dim = 0
        self.ego_dim = self.state_dim
        self.neighbor_input_dim = 0

        if self.use_neighbor_encoder:
            if self.neighbor_feature_dim is None:
                raise ValueError("'neighbor_feature_dim' must be provided with 'neighbor_encoder'.")
            self.neighbor_input_dim = int(self.neighbor_feature_dim)
            self.neighbor_context_dim = int(self.neighbor_encoder.out_dim)
            self.ego_dim = self.state_dim - self.neighbor_slots * (self.neighbor_input_dim + 1)
            if self.ego_dim <= 0:
                raise ValueError(
                    "'state_dim' is too small for the requested decentralized neighbor-packed layout. "
                    f"Got state_dim={self.state_dim}, neighbor_slots={self.neighbor_slots}, "
                    f"neighbor_feature_dim={self.neighbor_input_dim}."
                )

            main_input_dim = self.ego_dim + self.neighbor_context_dim
        else:
            main_input_dim = self.state_dim

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

    def forward(
        self,
        observation_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        ego_obs = observation_dict["ego_obs"]

        if self.use_neighbor_encoder:
            neighbor_obs = observation_dict.get("neighbor_obs")
            neighbor_mask = observation_dict.get("neighbor_mask")
            if neighbor_obs is None or neighbor_mask is None:
                raise ValueError("Decentralized policy requires neighbor observations and masks.")
            neighbor_context = self.neighbor_encoder(neighbor_obs, neighbor_mask)
            model_input = torch.cat([ego_obs, neighbor_context], dim=-1)
        else:
            model_input = ego_obs

        out = self.network(model_input)
        return out.view(out.shape[0], self.prediction_horizon, self.action_dim)

    def select_action(
        self,
        observation_dict: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.forward(observation_dict)

    def reset(self) -> None:
        """Keeps parity with other policy APIs that expose a reset hook."""
        return None
