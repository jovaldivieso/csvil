from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


class CustomMLPPolicy(nn.Module):
    """Simple feed-forward policy for continuous-action imitation learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
    ):
        super().__init__()

        if state_dim <= 0:
            raise ValueError(f"'state_dim' must be positive, got {state_dim}.")
        if action_dim <= 0:
            raise ValueError(f"'action_dim' must be positive, got {action_dim}.")
        if len(hidden_dims) < 1:
            raise ValueError("'hidden_dims' must contain at least one layer width.")

        layers: list[nn.Module] = []
        in_dim = state_dim
        for width in hidden_dims:
            if width <= 0:
                raise ValueError(f"Hidden layer widths must be positive, got {width}.")
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, observation_tensor: torch.Tensor) -> torch.Tensor:
        if observation_tensor.ndim == 1:
            observation_tensor = observation_tensor.unsqueeze(0)
        return self.network(observation_tensor)

    def select_action(
        self,
        observation_tensor: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Match policy API style by returning a tensor action for a batch.

        Accepts either a pre-flattened tensor or a dict of observation tensors
        that will be concatenated in insertion order.
        """
        if isinstance(observation_tensor, Mapping):
            chunks: list[torch.Tensor] = []
            for value in observation_tensor.values():
                tensor = value
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                chunks.append(tensor)
            if not chunks:
                raise ValueError("Observation mapping is empty.")
            model_input = torch.cat(chunks, dim=-1)
        else:
            model_input = observation_tensor

        return self.forward(model_input)

    def reset(self) -> None:
        """Keeps parity with other policy APIs that expose a reset hook."""
        return None
