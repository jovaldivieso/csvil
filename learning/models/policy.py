from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn

DEFAULT_POLICY_TYPE = "mlp"


class ActionPolicy(nn.Module, ABC):
    """Abstract base class for all continuous-action imitation learning policies."""

    def forward(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Default PyTorch module forward pass delegates to select_action."""
        return self.select_action(observation_dict)

    @abstractmethod
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def compute_loss(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Optional loss computation hook used by policies with custom training objectives."""
        raise NotImplementedError(f"{type(self).__name__} does not implement 'compute_loss'.")


class PolicyFactory:
    @staticmethod
    def create(policy_type: str, **kwargs: object) -> ActionPolicy:
        normalized_type = policy_type.strip().lower()
        if normalized_type == "mlp":
            from learning.models.mlp_policy import MLPPolicy

            return MLPPolicy(**kwargs)
        if normalized_type == "flow":
            from learning.models.flow_policy import FlowPolicy

            return FlowPolicy(**kwargs)
        raise ValueError(
            f"Unknown policy type '{policy_type}'. "
            f"Supported policies: 'mlp', 'flow'."
        )
