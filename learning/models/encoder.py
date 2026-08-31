from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn

DEFAULT_ENCODER_TYPE = "deepset"


class ObservationEncoder(nn.Module, ABC):
    """Interface for encoders that turn structured observations into flat context."""

    @property
    @abstractmethod
    def out_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class EncoderFactory:
    @staticmethod
    def create(
        encoder_type: str,
        state_dim: int,
        neighbor_feature_dim: int,
        neighbor_slots: int,
        **kwargs: object,
    ) -> ObservationEncoder:
        normalized_type = encoder_type.strip().lower()
        if normalized_type == DEFAULT_ENCODER_TYPE:
            from learning.models.deepset_encoder import DeepSetEncoder

            return DeepSetEncoder(
                state_dim=state_dim,
                neighbor_feature_dim=neighbor_feature_dim,
                neighbor_slots=neighbor_slots,
                **kwargs,
            )
        if normalized_type == "transformer":
            from learning.models.transformer_encoder import TransformerEncoder

            return TransformerEncoder(
                state_dim=state_dim,
                neighbor_feature_dim=neighbor_feature_dim,
                neighbor_slots=neighbor_slots,
                **kwargs,
            )

        if normalized_type == "gnn":
            from learning.models.gnn_encoder import GNNEncoder

            return GNNEncoder(
                state_dim=state_dim,
                neighbor_feature_dim=neighbor_feature_dim,
                neighbor_slots=neighbor_slots,
                **kwargs,
            )

        raise ValueError(
            f"Unknown observation encoder '{encoder_type}'. "
            f"Supported observation encoders: '{DEFAULT_ENCODER_TYPE}', 'transformer', 'gnn'."
        )
