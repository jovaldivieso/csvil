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
        neighbor_feature_dim: int | None = None,
        neighbor_slots: int = 0,
        neighbor_encoder: ObservationEncoder | None = None,
    ):
        super().__init__()

        if state_dim <= 0:
            raise ValueError(f"'state_dim' must be positive, got {state_dim}.")
        if action_dim <= 0:
            raise ValueError(f"'action_dim' must be positive, got {action_dim}.")
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
                    "'state_dim' is too small for the requested deep-set layout. "
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
        layers.append(nn.Linear(in_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def _split_structured_observation(
        self,
        observation_tensor: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(observation_tensor, Mapping):
            if "ego_obs" in observation_tensor:
                ego = observation_tensor["ego_obs"]
            else:
                env = observation_tensor.get("observation.environment_state")
                state = observation_tensor.get("observation.state")
                if env is None or state is None:
                    raise KeyError(
                        "Structured observations must provide either 'ego_obs' or both "
                        "'observation.environment_state' and 'observation.state'."
                    )
                ego = torch.cat([env, state], dim=-1)

            neighbor_obs = observation_tensor.get("neighbor_obs")
            neighbor_mask = observation_tensor.get("neighbor_mask")
            if neighbor_obs is None and "observation.neighbor_state" in observation_tensor:
                neighbor_obs = observation_tensor["observation.neighbor_state"]
            if neighbor_mask is None and "observation.neighbor_mask" in observation_tensor:
                neighbor_mask = observation_tensor["observation.neighbor_mask"]

            if ego.ndim == 1:
                ego = ego.unsqueeze(0)
            if self.use_neighbor_encoder and neighbor_obs is not None and neighbor_mask is not None:
                if neighbor_obs.ndim == 1:
                    neighbor_obs = neighbor_obs.unsqueeze(0)
                if neighbor_mask.ndim == 1:
                    neighbor_mask = neighbor_mask.unsqueeze(0)

                batch_size = ego.shape[0]
                if self.neighbor_slots == 0:
                    neighbor_obs = neighbor_obs.reshape(batch_size, 0, self.neighbor_input_dim)
                    neighbor_mask = neighbor_mask.reshape(batch_size, 0, 1)
                else:
                    if neighbor_obs.ndim == 2:
                        neighbor_obs = neighbor_obs.reshape(
                            batch_size,
                            self.neighbor_slots,
                            self.neighbor_input_dim,
                        )
                    if neighbor_mask.ndim == 2:
                        neighbor_mask = neighbor_mask.reshape(batch_size, self.neighbor_slots, 1)

            return ego, neighbor_obs, neighbor_mask

        if observation_tensor.ndim == 1:
            observation_tensor = observation_tensor.unsqueeze(0)

        if not self.use_neighbor_encoder:
            return observation_tensor, None, None

        neighbor_total = self.neighbor_slots * (self.neighbor_input_dim + 1)
        ego = observation_tensor[:, : self.ego_dim]
        neighbor_flat = observation_tensor[:, self.ego_dim : self.ego_dim + neighbor_total]

        if self.neighbor_slots == 0:
            neighbor_obs = observation_tensor.new_zeros((observation_tensor.shape[0], 0, self.neighbor_input_dim))
            neighbor_mask = observation_tensor.new_zeros((observation_tensor.shape[0], 0, 1))
        else:
            neighbor_obs = neighbor_flat[:, : self.neighbor_slots * self.neighbor_input_dim].reshape(
                observation_tensor.shape[0],
                self.neighbor_slots,
                self.neighbor_input_dim,
            )
            neighbor_mask = neighbor_flat[:, self.neighbor_slots * self.neighbor_input_dim :].reshape(
                observation_tensor.shape[0],
                self.neighbor_slots,
                1,
            )

        return ego, neighbor_obs, neighbor_mask

    def forward(
        self,
        observation_tensor: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        ego_obs, neighbor_obs, neighbor_mask = self._split_structured_observation(observation_tensor)

        if self.use_neighbor_encoder:
            if neighbor_obs is None or neighbor_mask is None:
                raise ValueError("Deep-set policy mode requires neighbor observations and masks.")
            neighbor_context = self.neighbor_encoder(neighbor_obs, neighbor_mask)
            model_input = torch.cat([ego_obs, neighbor_context], dim=-1)
        else:
            model_input = ego_obs

        return self.network(model_input)

    def select_action(
        self,
        observation_tensor: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Match policy API style by returning a tensor action for a batch.

                Accepts one of the following observation formats:

                - A pre-flattened torch.Tensor.
                - A structured mapping containing ``ego_obs``, ``neighbor_obs``, and
                    ``neighbor_mask`` (or ``observation.neighbor_state`` and its mask).
                - A general mapping of observation tensors concatenated in insertion order.
        """
        if isinstance(observation_tensor, Mapping):
            if any(
                key in observation_tensor
                for key in ("ego_obs", "neighbor_obs", "neighbor_mask", "observation.neighbor_state")
            ):
                return self.forward(observation_tensor)

            chunks: list[torch.Tensor] = []
            for value in observation_tensor.values():
                tensor = value
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                chunks.append(tensor)
            if not chunks:
                raise ValueError("Observation mapping is empty.")
            model_input = torch.cat(chunks, dim=-1)
            return self.forward(model_input)

        return self.forward(observation_tensor)

    def reset(self) -> None:
        """Keeps parity with other policy APIs that expose a reset hook."""
        return None
