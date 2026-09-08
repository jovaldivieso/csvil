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
# neighbor slot, so its d_safe constraint is always trivially satisfied.
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
        simulator: DynamicsProtocol,
    ) -> None:
        super().__init__()
        self.inner_policy = inner_policy
        self.simulator = simulator
        num_robots = int(getattr(simulator, "num_robots", 1))
        # One sim object per robot, not a single one shared across all of
        # them: each carries its own mutable goal (systems/multi_robot.py's
        # set_goal), and invert_obs() depends on it -- sharing simulators[0]
        # for every robot would silently make every robot but the first
        # invert its observation against robot 0's goal instead of its own.
        # Dynamics parameters (dt, nx, index tuples, ...) ARE identical
        # across a homogeneous fleet, so local_sims[0] is used deliberately
        # (and safely) wherever only those are needed, e.g. in
        # _build_neighbor_trajectories.
        self.local_sims = list(simulator.simulators) if num_robots > 1 else [simulator]
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

    def compute_loss(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.inner_policy.compute_loss(observation_dict, actions)

    def _extract_ego_observation(self, observation_dict: Mapping[str, torch.Tensor]) -> np.ndarray:
        """Reconstruct the flat, single-frame ego observation ``local_sim.invert_obs`` expects.

        ``observation.environment_state`` is always single-frame.
        ``observation.state`` (this robot's own proprioception) is now
        stacked across ``observation_horizon`` -- so the policy sees its own
        recent motion history -- but ``invert_obs`` still needs exactly one
        frame, so this slices out the most recent one: the last
        ``per_frame_state_dim`` columns, matching the oldest-to-newest
        stacking convention shared with the neighbor tensors. Degenerates to
        the whole tensor when ``observation_horizon == 1``.
        """
        observation_horizon = int(getattr(self.inner_policy.obs_encoder, "observation_horizon", 1))
        state = observation_dict["observation.state"]
        per_frame_state_dim = state.shape[-1] // observation_horizon
        latest_state = state[:, -per_frame_state_dim:]
        parts = [
            observation_dict["observation.environment_state"],
            latest_state,
        ]
        return torch.cat(parts, dim=-1).detach().cpu().numpy()

    def _build_neighbor_trajectories(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        x0_batch: np.ndarray,
    ) -> np.ndarray:
        """Extrapolate each neighbor's absolute position over the horizon.

        The neighbor observation stores each neighbor's relative
        ``[rel_x, rel_y, sin(rel_theta), cos(rel_theta)]`` in *this robot's
        own ego frame at that instant*
        (``systems/multi_robot.py``'s ``global_vector_to_ego`` rotates by
        this robot's *current* ``-theta`` every frame) -- there is no
        velocity channel, and the observing robot's own frame is itself
        rotating (and translating) between history frames. So naively
        differencing ``rel_pos(t) - rel_pos(t-1)`` folds this robot's own
        motion into the estimate instead of isolating the neighbor's: e.g. if
        the neighbor is stationary and only this robot spins in place, that
        difference is nonzero anyway, purely from the frame rotating under
        it -- and even without rotation, differencing raw relative vectors
        gives the *relative* velocity (v_neighbor - v_self), not the
        neighbor's own, which is what's needed to extrapolate its *absolute*
        future position.

        To avoid both, this reconstructs each neighbor's absolute position at
        both history instants before differencing: "now" from this robot's
        current pose (always known exactly), and "previous" by
        back-propagating this robot's *own* pose one step using its own
        observed proprioception (``velocity_state_indices``, e.g. unicycle2's
        ``(v, omega)``) under the same constant-twist assumption already used
        elsewhere here -- the same order of approximation as assuming the
        neighbor's own velocity is roughly constant over one step. Systems
        without proprioception (``velocity_state_indices`` empty, e.g.
        single_integrator) fall back to treating this robot's own pose as
        unchanged between frames, which is the pre-existing behavior for
        them.

        That reconstructed absolute position/velocity is then extrapolated
        over the horizon. When this system has both a heading
        (``angular_state_indices``) and proprioception
        (``velocity_state_indices``) -- i.e. a unicycle-shaped model -- the
        *neighbor* is extrapolated the same way, not with a straight line:
        homogeneous fleet means the neighbor runs the exact same dynamics, so
        its own heading (this robot's heading + the observed relative
        heading) and twist (turn rate from two consecutive heading estimates;
        speed from projecting the Cartesian velocity estimate onto that
        heading, which also discards any lateral component finite-difference
        noise would otherwise leak in -- a unicycle can't move sideways) seed
        a forward simulation through ``local_sim.predict_next_state`` itself,
        tracing a proper arc under sustained turning instead of a chord.
        Systems without a heading/velocity state (e.g. single_integrator,
        double_integrator) keep the straight-line extrapolation, which is
        exact for them (no heading to curve around) or the best information
        available otherwise.
        """
        batch_size = x0_batch.shape[0]
        horizon = self.projectors[0].N
        neighbor_trajs = np.full(
            (batch_size, self.neighbor_slots, 2, horizon + 1), _NO_NEIGHBOR_SENTINEL, dtype=float
        )
        if self.neighbor_slots == 0:
            return neighbor_trajs

        dt = float(self.local_sims[0].dt)
        pos_idx = tuple(getattr(self.local_sims[0], "position_indices", (0, 1)))
        angular_idx = tuple(getattr(self.local_sims[0], "angular_state_indices", ()))
        theta_idx = angular_idx[0] if angular_idx else None
        velocity_idx = tuple(getattr(self.local_sims[0], "velocity_state_indices", ()))

        obs_encoder = self.inner_policy.obs_encoder
        observation_horizon = int(getattr(obs_encoder, "observation_horizon", 1))
        neighbor_feature_dim = int(getattr(obs_encoder, "neighbor_feature_dim", 0))
        per_frame_dim = neighbor_feature_dim // observation_horizon if observation_horizon > 0 else neighbor_feature_dim

        neighbor_state = observation_dict["observation.neighbor_state"].detach().cpu().numpy()
        neighbor_mask = observation_dict["observation.neighbor_mask"].detach().cpu().numpy()
        # Time-major, neighbor-major, feature-minor packing (ObservationHistoryBuffer).
        feat = neighbor_state.reshape(batch_size, observation_horizon, self.neighbor_slots, per_frame_dim)
        mask = neighbor_mask.reshape(batch_size, observation_horizon, self.neighbor_slots)

        rel_pos_now = feat[:, -1, :, 0:2]
        mask_now = mask[:, -1, :]
        steps = np.arange(horizon + 1, dtype=float) * dt  # (N+1,)
        use_dynamics_model = theta_idx is not None and len(velocity_idx) == 2
        zero_action = np.zeros(self.local_sims[0].nu)

        for b in range(batch_size):
            theta_now = float(x0_batch[b, theta_idx]) if theta_idx is not None else 0.0
            cos_now, sin_now = np.cos(theta_now), np.sin(theta_now)
            pos_now = x0_batch[b, list(pos_idx)]

            if len(velocity_idx) == 2:
                v_now = float(x0_batch[b, velocity_idx[0]])
                omega_now = float(x0_batch[b, velocity_idx[1]])
            else:
                v_now, omega_now = 0.0, 0.0
            theta_prev = theta_now - omega_now * dt
            cos_prev, sin_prev = np.cos(theta_prev), np.sin(theta_prev)
            pos_prev = pos_now - v_now * dt * np.array([cos_prev, sin_prev])

            for j in range(self.neighbor_slots):
                if mask_now[b, j] <= 0.5:
                    continue  # masked-out neighbor: leave the far-away sentinel
                rx, ry = rel_pos_now[b, j]
                abs_now = np.array(
                    [pos_now[0] + cos_now * rx - sin_now * ry, pos_now[1] + sin_now * rx + cos_now * ry]
                )

                if observation_horizon >= 2:
                    prx, pry = feat[b, -2, j, 0:2]
                    abs_prev = np.array(
                        [pos_prev[0] + cos_prev * prx - sin_prev * pry, pos_prev[1] + sin_prev * prx + cos_prev * pry]
                    )
                    abs_vel = (abs_now - abs_prev) / dt
                else:
                    abs_vel = np.zeros(2)

                if use_dynamics_model:
                    rel_theta_now = np.arctan2(feat[b, -1, j, 2], feat[b, -1, j, 3])
                    theta_n_now = np.arctan2(np.sin(theta_now + rel_theta_now), np.cos(theta_now + rel_theta_now))
                    if observation_horizon >= 2:
                        rel_theta_prev = np.arctan2(feat[b, -2, j, 2], feat[b, -2, j, 3])
                        theta_n_prev = np.arctan2(
                            np.sin(theta_prev + rel_theta_prev), np.cos(theta_prev + rel_theta_prev)
                        )
                        omega_n = np.arctan2(np.sin(theta_n_now - theta_n_prev), np.cos(theta_n_now - theta_n_prev)) / dt
                        # Project onto the *previous* heading, not the
                        # current one: predict_next_state advances position
                        # using the heading at the *start* of each step, so
                        # abs_vel (a finite difference over [t-1, t]) points
                        # along theta_n_prev, not theta_n_now -- projecting
                        # onto the wrong one costs a cos(omega*dt) factor,
                        # a small but needless O(dt) speed underestimate.
                        v_n = abs_vel[0] * np.cos(theta_n_prev) + abs_vel[1] * np.sin(theta_n_prev)
                    else:
                        v_n, omega_n = 0.0, 0.0

                    neighbor_state_est = np.zeros(self.local_sims[0].nx)
                    neighbor_state_est[list(pos_idx)] = abs_now
                    neighbor_state_est[theta_idx] = theta_n_now
                    neighbor_state_est[velocity_idx[0]] = v_n
                    neighbor_state_est[velocity_idx[1]] = omega_n

                    traj_x, traj_y = [abs_now[0]], [abs_now[1]]
                    state_k = neighbor_state_est
                    for _ in range(horizon):
                        state_k = self.local_sims[0].predict_next_state(state_k, zero_action, validate=False)
                        traj_x.append(state_k[pos_idx[0]])
                        traj_y.append(state_k[pos_idx[1]])
                    neighbor_trajs[b, j, 0, :] = traj_x
                    neighbor_trajs[b, j, 1, :] = traj_y
                    continue

                neighbor_trajs[b, j, 0, :] = abs_now[0] + abs_vel[0] * steps
                neighbor_trajs[b, j, 1, :] = abs_now[1] + abs_vel[1] * steps

        return neighbor_trajs

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        inner = self.inner_policy
        obs_cond = inner.obs_encoder(observation_dict)
        batch_size = obs_cond.shape[0]
        device = obs_cond.device

        x = torch.randn(
            batch_size, inner.prediction_horizon, inner.action_dim, device=device, dtype=obs_cond.dtype
        )

        ego_obs_np = self._extract_ego_observation(observation_dict)
        # invert_obs is goal-dependent (systems/unicycle2.py reconstructs
        # absolute state from a goal-relative observation), so this must use
        # each robot's own sim object, not a shared one.
        x0_batch = np.stack([self.local_sims[b].invert_obs(ego_obs_np[b]) for b in range(batch_size)])
        neighbor_trajs_batch = (
            self._build_neighbor_trajectories(observation_dict, x0_batch) if self.neighbor_slots > 0 else None
        )

        dt = 1.0 / float(inner.num_inference_steps)
        for step in range(inner.num_inference_steps):
            t_val = step * dt
            t_tensor = torch.full((batch_size,), t_val, device=device, dtype=obs_cond.dtype)
            pred_velocity = inner._predict_velocity(x.flatten(1), obs_cond, t_tensor)
            pred_velocity = pred_velocity.view(batch_size, inner.prediction_horizon, inner.action_dim)
            x = x + dt * pred_velocity

            # --- SafeFlowMPC projection step ---
            x_np = x.detach().cpu().numpy()

            def _project_one(i: int) -> np.ndarray:
                u_ref = x_np[i].T  # (action_dim, horizon) -> (nu, N) for CasADi
                neighbor_trajs = neighbor_trajs_batch[i] if neighbor_trajs_batch is not None else None
                return self.projectors[i].project(ego_obs_np[i], u_ref, neighbor_trajs)

            # Each robot's projection is fully independent (decentralized;
            # neighbors enter only as parameters), so they're dispatched
            # concurrently across the per-robot projector pool rather than
            # solved one at a time.
            if self._pool is not None:
                results = list(self._pool.map(_project_one, range(batch_size)))
            else:
                results = [_project_one(i) for i in range(batch_size)]
            for i, u_safe in enumerate(results):
                x_np[i] = u_safe.T
            x = torch.tensor(x_np, device=device, dtype=x.dtype)

        return x
