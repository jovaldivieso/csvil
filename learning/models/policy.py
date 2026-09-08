from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn

DEFAULT_POLICY_TYPE = "mlp"


class ActionPolicy(nn.Module, ABC):
    """Abstract base class for all continuous-action imitation learning policies."""

    def forward(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError(
            "Do not call forward() directly on ActionPolicy. "
            "Use select_action() for inference or compute_loss() for training."
        )

    @abstractmethod
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


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
        if normalized_type == "safeflow":
            from learning.models.flow_policy import FlowPolicy
            from learning.models.safe_flow_policy import SafeFlowMPCPolicy
            from planning.casadi_projector import CasadiTrajectoryProjector

            flow_kwargs = dict(kwargs)
            simulator = flow_kwargs.pop("simulator", None)
            planner_config = flow_kwargs.pop("planner_config", None)
            if simulator is None or planner_config is None:
                raise ValueError(
                    "policy_type 'safeflow' requires 'simulator' and 'planner_config' kwargs "
                    "to build its CasadiTrajectoryProjector."
                )
            inner_policy = FlowPolicy(**flow_kwargs)
            num_robots = int(getattr(simulator, "num_robots", 1))
            local_sims = list(simulator.simulators) if num_robots > 1 else [simulator]
            neighbor_slots = max(0, num_robots - 1)
            # One independent Opti problem per robot, each bound to *that*
            # robot's own sim object -- not a single one shared across all of
            # them. Dynamics parameters (dt, nx, bounds, ...) are identical
            # across a homogeneous fleet so sharing those would be harmless,
            # but each sim object also carries its own mutable goal (set via
            # MultiRobotSimulator.set_goal), and the projector's invert_obs()
            # and goal_state reads both depend on it -- sharing simulators[0]
            # for every projector would silently make every robot but the
            # first track robot 0's goal instead of its own. Concurrent
            # select_action solves also need distinct Opti instances anyway:
            # a shared Opti's mutable per-call state (set_value/set_initial)
            # isn't safe to touch from multiple threads at once.
            projectors = [
                CasadiTrajectoryProjector(
                    local_sims[i], planner_config, neighbor_slots, horizon=inner_policy.prediction_horizon
                )
                for i in range(num_robots)
            ]
            return SafeFlowMPCPolicy(inner_policy, projectors, simulator)
        raise ValueError(
            f"Unknown policy type '{policy_type}'. "
            f"Supported policies: 'mlp', 'flow', 'safeflow'."
        )
