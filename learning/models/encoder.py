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

    @staticmethod
    def _compute_ego_dim(state_dim: int, neighbor_slots: int, neighbor_feature_dim: int) -> int:
        ego_dim = state_dim - neighbor_slots * neighbor_feature_dim
        if ego_dim <= 0:
            raise ValueError("'state_dim' is too small for the packed observation layout.")
        return ego_dim

    @staticmethod
    def _split_neighbor_tensors(
        raw_neighbors: torch.Tensor,
        raw_mask: torch.Tensor,
        neighbor_feature_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Un-flatten packed per-neighbor tensors into (B, neighbor_count, feature_dim)/(B, neighbor_count).

        Each observation packs its neighbor features neighbor-major,
        feature-minor. The neighbor count is inferred from the flat size (not
        taken from the encoder's configured ``neighbor_slots``) so encoders
        stay agnostic to the exact runtime fleet size.
        """
        batch_size = raw_neighbors.shape[0]
        if raw_neighbors.shape[1] == 0 and raw_mask.shape[1] == 0:
            return (
                raw_neighbors.new_empty((batch_size, 0, neighbor_feature_dim)),
                raw_mask.new_empty((batch_size, 0)),
            )
        try:
            neighbor_obs = raw_neighbors.view(batch_size, -1, neighbor_feature_dim)
        except RuntimeError as exc:
            raise ValueError(
                "Flat neighbor tensors do not match the encoder's configured feature dimensions."
            ) from exc
        if neighbor_obs.shape[1] != raw_mask.shape[1]:
            raise ValueError("Neighbor state and mask tensors must contain the same number of slots.")
        return neighbor_obs, raw_mask


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
