from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class ObservationEncoder(nn.Module, ABC):
    """Interface for encoders that turn a masked neighbor observation set into context."""

    @property
    @abstractmethod
    def out_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, neighbor_obs: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EncoderFactory:
    @staticmethod
    def create(encoder_type: str, in_features: int, **kwargs: object) -> ObservationEncoder:
        normalized_type = encoder_type.strip().lower()
        if normalized_type == "deepset":
            from learning.models.deep_set_encoder import DeepSetEncoder

            return DeepSetEncoder(in_features=in_features, **kwargs)
        raise ValueError(f"Unknown observation encoder '{encoder_type}'. Supported encoders: 'deepset'.")
