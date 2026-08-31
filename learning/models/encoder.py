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
    def _compute_ego_dim(
        state_dim: int,
        neighbor_slots: int,
        neighbor_feature_dim: int,
        observation_horizon: int,
    ) -> int:
        ego_dim = state_dim - neighbor_slots * (neighbor_feature_dim + observation_horizon)
        if ego_dim <= 0:
            raise ValueError("'state_dim' is too small for the packed observation layout.")
        return ego_dim

    @staticmethod
    def _split_neighbor_tensors(
        raw_neighbors: torch.Tensor,
        raw_mask: torch.Tensor,
        neighbor_feature_dim: int,
        observation_horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Un-flatten packed per-neighbor tensors into (B, neighbor_count, ...).

        The dataset pipeline concatenates history frames time-major (oldest to
        newest), and each frame packs its neighbor features neighbor-major,
        feature-minor. The flat layout is therefore ``(time, neighbor, feature)``
        for ``raw_neighbors`` and ``(time, neighbor)`` for ``raw_mask`` -- this
        permutes both into a neighbor-major view so each neighbor slot carries
        its own feature/mask history. The neighbor count is inferred from the
        flat size (not taken from the encoder's configured ``neighbor_slots``)
        so encoders stay agnostic to the exact runtime fleet size.
        """
        batch_size = raw_neighbors.shape[0]
        per_frame_dim = neighbor_feature_dim // observation_horizon
        try:
            neighbor_obs = raw_neighbors.view(batch_size, observation_horizon, -1, per_frame_dim)
            neighbor_mask = raw_mask.view(batch_size, observation_horizon, -1)
        except RuntimeError as exc:
            raise ValueError(
                "Flat neighbor tensors do not match the encoder's configured feature dimensions."
            ) from exc
        if neighbor_obs.shape[2] != neighbor_mask.shape[2]:
            raise ValueError("Neighbor state and mask tensors must contain the same number of slots.")
        neighbor_count = neighbor_obs.shape[2]
        neighbor_obs = neighbor_obs.permute(0, 2, 1, 3).reshape(batch_size, neighbor_count, neighbor_feature_dim)
        neighbor_mask = neighbor_mask.permute(0, 2, 1)
        return neighbor_obs, neighbor_mask


class EncoderFactory:
    @staticmethod
    def create(
        encoder_type: str,
        state_dim: int,
        neighbor_feature_dim: int,
        neighbor_slots: int,
        observation_horizon: int = 1,
        **kwargs: object,
    ) -> ObservationEncoder:
        normalized_type = encoder_type.strip().lower()
        if normalized_type == DEFAULT_ENCODER_TYPE:
            from learning.models.deepset_encoder import DeepSetEncoder

            return DeepSetEncoder(
                state_dim=state_dim,
                neighbor_feature_dim=neighbor_feature_dim,
                neighbor_slots=neighbor_slots,
                observation_horizon=observation_horizon,
                **kwargs,
            )
        if normalized_type == "transformer":
            from learning.models.transformer_encoder import TransformerEncoder

            return TransformerEncoder(
                state_dim=state_dim,
                neighbor_feature_dim=neighbor_feature_dim,
                neighbor_slots=neighbor_slots,
                observation_horizon=observation_horizon,
                **kwargs,
            )

        if normalized_type == "gnn":
            from learning.models.gnn_encoder import GNNEncoder

            return GNNEncoder(
                state_dim=state_dim,
                neighbor_feature_dim=neighbor_feature_dim,
                neighbor_slots=neighbor_slots,
                observation_horizon=observation_horizon,
                **kwargs,
            )

        raise ValueError(
            f"Unknown observation encoder '{encoder_type}'. "
            f"Supported observation encoders: '{DEFAULT_ENCODER_TYPE}', 'transformer', 'gnn'."
        )
