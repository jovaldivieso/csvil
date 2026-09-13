from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
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
            # Whether to unwrap turns on whether `simulator` *is* a
            # MultiRobotSimulator (it always exposes `.simulators`, holding
            # 1 robot or many), not on whether it holds more than one robot:
            # a one-robot multi_robot config is valid, and MultiRobotSimulator
            # itself doesn't expose the per-robot dynamics metadata
            # (velocity_state_indices, etc.) this policy and its projector
            # need -- only its sub-simulators do. A bare, non-multi_robot
            # simulator (no `.simulators` at all) is used as-is.
            live_sims = list(getattr(simulator, "simulators", [simulator]))
            neighbor_slots = max(0, num_robots - 1)
            # One independent Opti problem per robot, each bound to *that*
            # robot's own sim object -- not a single one shared across all of
            # them. Dynamics parameters (dt, nx, bounds, ...) are identical
            # across a homogeneous fleet so sharing those would be harmless,
            # but each sim object also carries a goal attribute, and
            # invert_obs() reads it to reconstruct absolute state from the
            # ego-relative observation -- sharing simulators[0] for every
            # projector would silently make every robot but the first
            # invert against robot 0's goal instead of its own.
            #
            # These are deep-copied with the goal zeroed, once, here, rather
            # than kept as live references into `simulator`: every
            # constraint/cost this projector pipeline touches (dynamics,
            # action/velocity bounds, pairwise neighbor distances, terminal
            # velocity) is equivariant under a per-robot rigid transform of
            # invert_obs's reconstructed frame, and neighbor trajectories are
            # built from that same reconstructed x0 rather than any absolute
            # world position -- so the goal used here is never anything more
            # than a fixed, self-consistent local anchor, not the live
            # episode's actual goal. That means this policy never needs to
            # track -- or be resynced to -- whatever simulator instance
            # actually drives the live rollout (DAgger collection and
            # evaluation each construct their own, separate from this one,
            # every round/call). Deep-copying also protects a caller that
            # passes the same simulator object for both policy construction
            # and its own rollout from having that simulator's goal
            # overwritten to zero out from under it.
            #
            # This equivalence would break if this projector ever gained a
            # genuinely world-frame-referencing term (finite absolute
            # position bounds, static obstacles, a goal-tracking cost) --
            # anything added like that must take the real goal as an
            # explicit input rather than reading it off these local_sims.
            local_sims = []
            for sim in live_sims:
                local_sim = copy.deepcopy(sim)
                local_sim.set_goal(np.zeros_like(local_sim.goal))
                local_sims.append(local_sim)

            # d_safe/d_collision are fleet-level (MultiRobotSim) attributes,
            # not attributes of the single-robot `sim` each projector holds
            # (local_sims[i]) -- CasadiTrajectoryProjector's own
            # config.get("d_safe", getattr(self.sim, "d_safe", 0.0)) fallback
            # therefore can never actually reach the fleet's real value on
            # its own, and silently settles on 0.0 (no collision avoidance
            # at all) whenever planner_config doesn't separately repeat it.
            # Every current caller's planner_config happens to already carry
            # both (it's the full validated system config), but nothing
            # enforces that, so a future caller passing a narrower
            # planner_config would silently get an unsafe policy with no
            # error. Explicit planner_config overrides are still respected
            # via setdefault; only a missing key falls back to the fleet's
            # own value.
            projector_config = dict(planner_config)
            projector_config.setdefault("d_safe", float(getattr(simulator, "d_safe", 0.0)))
            projector_config.setdefault(
                "d_collision",
                float(getattr(simulator, "d_collision", projector_config["d_safe"])),
            )
            projectors = [
                CasadiTrajectoryProjector(
                    local_sims[i],
                    projector_config,
                    neighbor_slots,
                    horizon=inner_policy.prediction_horizon,
                    robot_index=i,
                )
                for i in range(num_robots)
            ]
            return SafeFlowMPCPolicy(inner_policy, projectors, local_sims)
        raise ValueError(
            f"Unknown policy type '{policy_type}'. "
            f"Supported policies: 'mlp', 'flow', 'safeflow'."
        )
