from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from learning.models.flow_policy import FlowPolicy
from learning.models.policy import ActionPolicy
from planning.casadi_projector import CasadiTrajectoryProjector
from systems.dynamics import DynamicsProtocol

# Sentinel position (far outside any real workspace) for a masked-out (absent)
# neighbor slot. No collision constraint is meaningful for an unseen neighbor.
_NO_NEIGHBOR_SENTINEL = 1.0e6


class SafeFlowMPCPolicy(ActionPolicy):
    """Wraps a FlowPolicy with a CasADi safety projection at every inference step.

    Composition over inheritance: encoding, loss, and reset all delegate to
    the inner FlowPolicy unchanged. ``select_action`` re-implements the Euler
    ODE integration loop, extrapolates other robots' positions from the
    decentralized neighbor observation, and projects the intermediate action
    sequence onto the safety manifold via ``CasadiTrajectoryProjector`` after
    every flow step (SafeFlowMPC, Oelerich et al., 2026).
    """

    def __init__(
        self,
        inner_policy: FlowPolicy,
        projectors: Sequence[CasadiTrajectoryProjector],
        local_sims: Sequence[DynamicsProtocol],
    ) -> None:
        super().__init__()
        self.inner_policy = inner_policy
        # One sim object per robot, not a single one shared across all of
        # them: invert_obs() reads each one's own goal attribute to
        # reconstruct absolute state from the ego-relative observation --
        # sharing simulators[0] for every robot would silently make every
        # robot but the first invert against robot 0's goal instead of its
        # own. Dynamics parameters (dt, nx, index tuples, ...) ARE identical
        # across a homogeneous fleet, so local_sims[0] is used deliberately
        # (and safely) wherever only those are needed, e.g. in
        # _build_neighbor_trajectories.
        #
        # These are PolicyFactory.create's own private, goal-zeroed copies
        # (see its comment for why a fixed, arbitrary goal anchor is exactly
        # as correct here as the live episode's actual one) -- never a live
        # simulator reference, so this policy has nothing that needs to be
        # kept in sync with whatever simulator instance actually drives a
        # given rollout/evaluation call.
        self.local_sims = list(local_sims)
        num_robots = len(self.local_sims)
        self.neighbor_slots = max(0, num_robots - 1)

        self.projectors = list(projectors)
        if len(self.projectors) != num_robots:
            raise ValueError(
                f"'projectors' must supply exactly one CasadiTrajectoryProjector per robot "
                f"({num_robots}), got {len(self.projectors)}."
            )
        # Each robot's projection is decentralized (fully independent of the
        # others; neighbors enter only as parameters), but each Opti instance
        # holds mutable per-call state (set_value/set_initial), so concurrent
        # calls need one dedicated projector each -- never share a single
        # Opti across threads. Only worth pooling threads for >1 robot.
        self._pool = ThreadPoolExecutor(max_workers=num_robots) if num_robots > 1 else None

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load either a SafeFlowMPCPolicy checkpoint or a bare FlowPolicy checkpoint.

        Strips a leading ``inner_policy.`` prefix if present (a checkpoint
        saved from a SafeFlowMPCPolicy), otherwise takes keys as-is (a plain
        FlowPolicy checkpoint) -- ``projector`` never appears in a state dict
        since ``CasadiTrajectoryProjector`` isn't an nn.Module. Delegates to
        ``self.inner_policy.load_state_dict`` directly rather than
        ``super().load_state_dict``: PyTorch's recursive module loading calls
        each submodule's private ``_load_from_state_dict`` hook, not its
        public ``load_state_dict`` override, so going through ``super()``
        would silently skip FlowPolicy's own torch.compile key remapping.
        """
        inner_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("inner_policy."):
                inner_state_dict[key[len("inner_policy.") :]] = value
            elif not key.startswith("projector."):
                inner_state_dict[key] = value
        return self.inner_policy.load_state_dict(inner_state_dict, strict=strict, assign=assign)

    def reset(self) -> None:
        self.inner_policy.reset()
        # Each projector carries its own persistent primal/dual warm-start
        # state across calls (see CasadiTrajectoryProjector.project()) --
        # without clearing it here, a new episode's first solve would warm
        # start from the previous episode's unrelated final trajectory.
        for projector in self.projectors:
            projector.reset()

    def compute_loss(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.inner_policy.compute_loss(observation_dict, actions)

    def _extract_ego_observation(self, observation_dict: Mapping[str, torch.Tensor]) -> np.ndarray:
        """Reconstruct the flat ego observation ``local_sim.invert_obs`` expects."""
        parts = [
            observation_dict["observation.environment_state"],
            observation_dict["observation.state"],
        ]
        return torch.cat(parts, dim=-1).detach().cpu().numpy()

    def _build_neighbor_trajectories(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        x0_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extrapolate each neighbor's absolute position over the horizon.

        Returns ``(neighbor_trajs, neighbor_active)``: ``neighbor_active``,
        shape ``(batch_size, neighbor_slots)``, is this tick's visibility
        mask (1 = a real, currently-visible neighbor; 0 = masked-out/absent)
        -- CasadiTrajectoryProjector gates its collision constraints on this
        directly rather than relying solely on a masked slot's
        ``neighbor_trajs`` entry being a numerically-far-away sentinel
        position.

        Neighbor velocity/turn-rate are direct observation features
        (``systems/multi_robot.py``'s ``observe()`` computes them as an
        exact finite difference against the previous joint state, zero when
        unavailable -- episode's first tick, or a neighbor that just entered
        visibility), not something this method has to reconstruct from
        stacked history. It only needs to rotate the ego-frame-relative
        velocity back into the global frame -- the inverse of
        ``global_vector_to_ego``'s rotation, the same formula used below for
        position -- before extrapolating.

        That absolute position/velocity is then extrapolated over the
        horizon. When this system has both a heading
        (``angular_state_indices``) and proprioception
        (``velocity_state_indices``) -- i.e. a unicycle-shaped model -- the
        *neighbor* is extrapolated the same way, not with a straight line:
        homogeneous fleet means the neighbor runs the exact same dynamics, so
        its own heading (this robot's heading + the observed relative
        heading) and twist (the observed turn-rate feature directly; speed
        from projecting the rotated velocity estimate onto the neighbor's
        *previous* heading -- reconstructed as ``theta_n_now - omega_n*dt``,
        matching ``predict_next_state``'s own convention of advancing
        position using the heading at the *start* of the step that produced
        this velocity estimate -- which also discards any lateral component
        estimation noise would otherwise leak in, since a unicycle can't
        move sideways) seed a forward simulation through
        ``local_sim.predict_next_state`` itself, tracing a proper arc under
        sustained turning instead of a chord. Systems without a
        heading/velocity state (e.g. single_integrator, double_integrator)
        keep the straight-line extrapolation, which is exact for them (no
        heading to curve around) or the best information available
        otherwise.
        """
        batch_size = x0_batch.shape[0]
        horizon = self.projectors[0].N
        neighbor_trajs = np.full(
            (batch_size, self.neighbor_slots, 2, horizon + 1), _NO_NEIGHBOR_SENTINEL, dtype=float
        )
        if self.neighbor_slots == 0:
            return neighbor_trajs, np.zeros((batch_size, 0), dtype=float)

        dt = float(self.local_sims[0].dt)
        pos_idx = tuple(getattr(self.local_sims[0], "position_indices", (0, 1)))
        angular_idx = tuple(getattr(self.local_sims[0], "angular_state_indices", ()))
        theta_idx = angular_idx[0] if angular_idx else None
        velocity_idx = tuple(getattr(self.local_sims[0], "velocity_state_indices", ()))
        position_dim = len(pos_idx)
        num_orientation = len(angular_idx)

        obs_encoder = self.inner_policy.obs_encoder
        neighbor_feature_dim = int(getattr(obs_encoder, "neighbor_feature_dim", 0))

        neighbor_state = observation_dict["observation.neighbor_state"].detach().cpu().numpy()
        neighbor_mask = observation_dict["observation.neighbor_mask"].detach().cpu().numpy()
        # Neighbor-major, feature-minor packing: per neighbor, [position...,
        # (sin, cos) per angular state..., velocity..., turn-rate per angular
        # state...] -- see MultiRobotSimulator._relative_feature_names.
        feat = neighbor_state.reshape(batch_size, self.neighbor_slots, neighbor_feature_dim)
        mask_now = neighbor_mask.reshape(batch_size, self.neighbor_slots)

        rel_pos_now = feat[:, :, 0:position_dim]
        vel_start = position_dim + 2 * num_orientation
        rel_vel_now = feat[:, :, vel_start : vel_start + position_dim]
        steps = np.arange(horizon + 1, dtype=float) * dt  # (N+1,)
        use_dynamics_model = theta_idx is not None and len(velocity_idx) == 2
        zero_action = np.zeros(self.local_sims[0].nu)

        for b in range(batch_size):
            theta_now = float(x0_batch[b, theta_idx]) if theta_idx is not None else 0.0
            cos_now, sin_now = np.cos(theta_now), np.sin(theta_now)
            pos_now = x0_batch[b, list(pos_idx)]

            for j in range(self.neighbor_slots):
                # Neighbor slot j is fleet index j if it comes before this
                # robot (b) in fleet order, else j + 1 (skipping over b's own
                # index) -- e.g. robot 2's slot 0 is fleet robot 0, but its
                # slot 2 is fleet robot 3, not 2. The fleet contract only
                # requires equal dimensions/types across robots, not equal
                # numeric parameters (systems/multi_robot.py's per-robot
                # config), so a neighbor can have a different max_linear_vel
                # than this robot -- forward-simulating it below with the wrong
                # sim object would clip its forecast to *this* robot's
                # limits instead of its own.
                neighbor_fleet_idx = j if j < b else j + 1
                if mask_now[b, j] <= 0.5:
                    # Masked-out neighbor: leave the far-away sentinel, i.e.
                    # treat it as absent rather than as a bounded-speed
                    # reachable set. The hard d_collision floor is therefore
                    # enforced only against neighbors represented by the
                    # current observation and its forecast. This policy does
                    # not provide a formal fleet-level collision guarantee for
                    # unseen neighbors, and independent per-robot projections
                    # do not provide a guarantee against uncoordinated future
                    # neighbor plans. Widening visibility or bounding relative
                    # speed can reduce this gap, but does not remove the need
                    # for conservative reachable sets or coordinated planning
                    # if a formal fleet-level guarantee is required.
                    continue
                rx, ry = rel_pos_now[b, j]
                abs_now = np.array(
                    [pos_now[0] + cos_now * rx - sin_now * ry, pos_now[1] + sin_now * rx + cos_now * ry]
                )
                # Same rotation as above, applied to the velocity feature
                # instead of the position feature -- observe() already
                # zeroes this when no previous reading exists (episode's
                # first tick, or this neighbor just entered visibility).
                vx_rel, vy_rel = rel_vel_now[b, j]
                abs_vel = np.array(
                    [cos_now * vx_rel - sin_now * vy_rel, sin_now * vx_rel + cos_now * vy_rel]
                )

                if use_dynamics_model:
                    rel_theta_now = np.arctan2(feat[b, j, position_dim], feat[b, j, position_dim + 1])
                    theta_n_now = np.arctan2(np.sin(theta_now + rel_theta_now), np.cos(theta_now + rel_theta_now))
                    omega_n = float(feat[b, j, vel_start + position_dim])
                    theta_n_prev = theta_n_now - omega_n * dt
                    v_n = abs_vel[0] * np.cos(theta_n_prev) + abs_vel[1] * np.sin(theta_n_prev)

                    neighbor_state_est = np.zeros(self.local_sims[0].nx)
                    neighbor_state_est[list(pos_idx)] = abs_now
                    neighbor_state_est[theta_idx] = theta_n_now
                    neighbor_state_est[velocity_idx[0]] = v_n
                    neighbor_state_est[velocity_idx[1]] = omega_n

                    traj_x, traj_y = [abs_now[0]], [abs_now[1]]
                    state_k = neighbor_state_est
                    for _ in range(horizon):
                        state_k = self.local_sims[neighbor_fleet_idx].predict_next_state(
                            state_k, zero_action, validate=False
                        )
                        traj_x.append(state_k[pos_idx[0]])
                        traj_y.append(state_k[pos_idx[1]])
                    neighbor_trajs[b, j, 0, :] = traj_x
                    neighbor_trajs[b, j, 1, :] = traj_y
                    continue

                neighbor_trajs[b, j, 0, :] = abs_now[0] + abs_vel[0] * steps
                neighbor_trajs[b, j, 1, :] = abs_now[1] + abs_vel[1] * steps

        return neighbor_trajs, mask_now

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        inner = self.inner_policy
        obs_cond = inner.obs_encoder(observation_dict)
        batch_size = obs_cond.shape[0]
        # self.local_sims[b]/self.projectors[b] are indexed directly by batch
        # position under the assumption that it *is* the fleet's own robot
        # ordering (see build_decentralized_joint_action, the only caller,
        # which always builds exactly one row per robot in that order) --
        # not checking this would let a batch of the wrong size silently
        # apply the wrong robot's projector/local_sim (too small) or crash
        # with an opaque IndexError (too large) instead of failing clearly.
        if batch_size != len(self.projectors):
            raise ValueError(
                f"SafeFlowMPCPolicy.select_action received a batch of size {batch_size}, but "
                f"this policy was constructed for a fleet of {len(self.projectors)} robots. Each "
                "batch row must correspond to exactly one robot, in fleet order."
            )
        device = obs_cond.device

        x = torch.randn(
            batch_size, inner.prediction_horizon, inner.action_dim, device=device, dtype=obs_cond.dtype
        )

        ego_obs_np = self._extract_ego_observation(observation_dict)
        # invert_obs is goal-dependent (systems/unicycle2.py reconstructs
        # absolute state from a goal-relative observation), so this must use
        # each robot's own sim object, not a shared one.
        x0_batch = np.stack([self.local_sims[b].invert_obs(ego_obs_np[b]) for b in range(batch_size)])
        neighbor_trajs_batch = None
        neighbor_active_batch = None
        if self.neighbor_slots > 0:
            neighbor_trajs_batch, neighbor_active_batch = self._build_neighbor_trajectories(
                observation_dict, x0_batch
            )

        # inner._predict_velocity operates in FlowPolicy's normalized action
        # space (see FlowPolicy.compute_loss/select_action), but the CasADi
        # projector -- like every other physical consumer -- compares u_ref
        # against the robot's real max_action, so it needs physical units.
        # x itself stays in normalized space throughout (that's what the
        # network was trained to integrate); only the values handed to/read
        # back from the projector are rescaled at that boundary.
        action_scale_np = inner.action_scale.detach().cpu().numpy()

        dt = 1.0 / float(inner.num_inference_steps)
        for step in range(inner.num_inference_steps):
            t_val = step * dt
            t_tensor = torch.full((batch_size,), t_val, device=device, dtype=obs_cond.dtype)
            pred_velocity = inner._predict_velocity(x.flatten(1), obs_cond, t_tensor)
            pred_velocity = pred_velocity.view(batch_size, inner.prediction_horizon, inner.action_dim)
            x = x + dt * pred_velocity

            # --- SafeFlowMPC projection step ---
            x_np = x.detach().cpu().numpy()
            x_physical_np = x_np * action_scale_np

            def _project_one(i: int) -> np.ndarray:
                u_ref = x_physical_np[i].T  # (action_dim, horizon) -> (nu, N) for CasADi
                neighbor_trajs = neighbor_trajs_batch[i] if neighbor_trajs_batch is not None else None
                neighbor_active = neighbor_active_batch[i] if neighbor_active_batch is not None else None
                return self.projectors[i].project(ego_obs_np[i], u_ref, neighbor_trajs, neighbor_active)

            # Each robot's projection is fully independent (decentralized;
            # neighbors enter only as parameters), so they're dispatched
            # concurrently across the per-robot projector pool rather than
            # solved one at a time.
            if self._pool is not None:
                # Not self._pool.map(...): map()'s iterator raises as soon as
                # it reaches a failed future, which can be before later
                # futures have even started -- list(...) would then abort
                # and leave those still running in the background while the
                # caller handles the exception (e.g. retries with a fresh
                # select_action() call), racing a new solve against a
                # stale one on the same mutable per-robot Opti instance.
                # Submitting everything first and waiting out every
                # future's .exception() before any .result() guarantees the
                # whole batch has finished before an exception can propagate.
                futures = [self._pool.submit(_project_one, i) for i in range(batch_size)]
                for future in futures:
                    future.exception()
                results = [future.result() for future in futures]
            else:
                results = [_project_one(i) for i in range(batch_size)]
            for i, u_safe in enumerate(results):
                # u_safe is physical units (same convention as u_ref above);
                # back to normalized space before it re-enters the Euler loop.
                x_np[i] = u_safe.T / action_scale_np
            x = torch.tensor(x_np, device=device, dtype=x.dtype)

        # Rescale the final normalized-space trajectory to physical units,
        # matching FlowPolicy.select_action's own return convention -- every
        # caller (apply_execution_noise, simulator.step, ...) expects that.
        return x * inner.action_scale
