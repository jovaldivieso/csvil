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

        # _build_neighbor_trajectories can only isolate a neighbor's absolute
        # velocity by differencing its reconstructed position against the
        # *previous* observation frame -- a single frame has no history to
        # difference against at all, regardless of whether this robot itself
        # has velocity proprioception. observation_horizon == 1 therefore
        # hits the exact same "every neighbor forecast as stationary for the
        # whole episode" gap the velocity-state check below closes for
        # velocity-less systems -- it just does it through the *encoder*
        # config instead of the *system* type, so that check alone would
        # miss it entirely for a velocity-having fleet stacking only one
        # frame.
        if self.neighbor_slots > 0 and int(getattr(inner_policy.obs_encoder, "observation_horizon", 1)) < 2:
            raise ValueError(
                "SafeFlowMPCPolicy requires observation_horizon >= 2 for a multi-robot fleet "
                "(neighbor_slots > 0): a moving neighbor's velocity can only be estimated by "
                "differencing two consecutive observation frames, so a single-frame encoder "
                "would forecast every neighbor as stationary for the entire episode regardless "
                "of its actual motion."
            )

        # _build_neighbor_trajectories reconstructs this robot's own previous
        # pose from its own observed proprioception (velocity_state_indices)
        # to isolate a neighbor's absolute velocity from the raw relative-
        # offset history (see its docstring). A system with no velocity
        # state at all (single_integrator, unicycle1) has no way to recover
        # that previous pose from a single current-frame observation, so
        # every visible neighbor is reported as momentarily stationary for
        # the entire episode, not just a one-tick warm-up artifact -- an
        # approaching neighbor could then satisfy the hard d_collision
        # forecast while closing distance for real. This is a genuine
        # incompatibility between decentralized multi-robot SafeFlow
        # coordination and velocity-less systems (unlike the earlier,
        # narrower terminal-rest constraint, which such systems satisfy
        # trivially and are NOT rejected for) -- fail closed here rather
        # than silently shipping a policy whose neighbor forecast cannot
        # bound real neighbor motion. A single first-order robot alone
        # (neighbor_slots == 0, nothing to forecast) is unaffected.
        if self.neighbor_slots > 0 and not getattr(self.local_sims[0], "velocity_state_indices", ()):
            raise ValueError(
                f"SafeFlowMPCPolicy cannot support a multi-robot fleet of "
                f"{type(self.local_sims[0]).__name__}: this system has no velocity state, so a "
                "neighbor's absolute velocity can never be recovered from observation history, "
                "and every neighbor would be forecast as stationary for the whole prediction "
                "horizon regardless of its actual motion. Use a velocity-having system (e.g. "
                "unicycle2, double_integrator) for multi-robot SafeFlow, or drop to a single "
                "robot (no neighbors to forecast)."
            )

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extrapolate each neighbor's absolute position over the horizon.

        Returns ``(neighbor_trajs, neighbor_active)``: ``neighbor_active``,
        shape ``(batch_size, neighbor_slots)``, is this tick's visibility
        mask (1 = a real, currently-visible neighbor; 0 = masked-out/absent)
        -- CasadiTrajectoryProjector gates its collision constraints on this
        directly rather than relying solely on a masked slot's
        ``neighbor_trajs`` entry being a numerically-far-away sentinel
        position.

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
        current pose (always known exactly), and "previous" from the actual
        previous frame of this robot's own observed proprioception
        (``velocity_state_indices``, e.g. unicycle2's ``(v, omega)``),
        already available via the stacked ``observation.state`` history --
        exact for this robot's own contribution to the estimate, leaving only
        the neighbor's own velocity changing between frames as a source of
        error, the same order of approximation as assuming that velocity is
        roughly constant over one step. Systems without proprioception
        (``velocity_state_indices`` empty, e.g. single_integrator,
        unicycle1) have no state from which to recover this robot's own
        previous pose at all, so this could only ever report every neighbor
        as momentarily stationary regardless of its real motion --
        ``SafeFlowMPCPolicy.__init__`` therefore rejects a velocity-less
        system outright as soon as it has any neighbor slots at all, so this
        method (only ever called when ``neighbor_slots > 0``, see
        ``select_action``) can assume ``velocity_state_indices`` is always
        populated.

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
            return neighbor_trajs, np.zeros((batch_size, 0), dtype=float)

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

        # observation.state stacks this robot's own proprioception across
        # observation_horizon frames (ObservationHistoryBuffer), so the
        # *actual* previous-frame velocity/turn-rate is already available --
        # no need to approximate it from the current frame as if unchanged.
        # predict_next_state advances position/heading using the velocity at
        # the *start* of each step, so using the real previous reading below
        # makes the back-propagated pos_prev/theta_prev exact instead of
        # biased by however much the ego itself accelerated between frames.
        ego_proprio_dim = len(velocity_idx)
        ego_proprio_prev_known = observation_horizon >= 2 and ego_proprio_dim > 0
        if ego_proprio_prev_known:
            ego_proprio_hist = observation_dict["observation.state"].detach().cpu().numpy()
            ego_state_mask_hist = observation_dict["observation.state_mask"].detach().cpu().numpy()
            ego_mask_dim = ego_state_mask_hist.shape[-1] // observation_horizon
            ego_proprio_prev_all = ego_proprio_hist[:, -2 * ego_proprio_dim : -ego_proprio_dim]
            ego_state_mask_prev_all = ego_state_mask_hist[:, -2 * ego_mask_dim : -ego_mask_dim]

        for b in range(batch_size):
            theta_now = float(x0_batch[b, theta_idx]) if theta_idx is not None else 0.0
            cos_now, sin_now = np.cos(theta_now), np.sin(theta_now)
            pos_now = x0_batch[b, list(pos_idx)]

            has_prev_proprio = (
                ego_proprio_prev_known and float(np.min(ego_state_mask_prev_all[b])) > 0.5
            )

            if theta_idx is not None and len(velocity_idx) == 2:
                # Unicycle-style (v, omega) proprioception: back-propagate
                # this robot's own heading and position using the *previous*
                # frame's v/omega (predict_next_state's own convention),
                # falling back to the current frame's when no real previous
                # one exists yet (the episode's first tick).
                v_now = float(x0_batch[b, velocity_idx[0]])
                omega_now = float(x0_batch[b, velocity_idx[1]])
                if has_prev_proprio:
                    v_prev, omega_prev = (float(value) for value in ego_proprio_prev_all[b])
                else:
                    v_prev, omega_prev = v_now, omega_now
                theta_prev = theta_now - omega_prev * dt
                cos_prev, sin_prev = np.cos(theta_prev), np.sin(theta_prev)
                pos_prev = pos_now - v_prev * dt * np.array([cos_prev, sin_prev])
            else:
                # Cartesian (vx, vy) proprioception with no heading to rotate
                # by (e.g. double_integrator) -- the only other case reachable
                # here, since SafeFlowMPCPolicy.__init__ rejects a velocity-
                # less system outright whenever neighbor_slots > 0 (this
                # method is never even called otherwise: see select_action).
                # Back-propagate each axis independently via
                # predict_next_state's own exact identity pos_now = pos_prev
                # + 0.5*dt*(v_prev + v_now) -- equivalent to its next_pos =
                # pos + v*dt + 0.5*a*dt**2 update with the action eliminated
                # using next_vel = v + a*dt, so this needs no knowledge of
                # the actual action, only the previous frame's velocity
                # (falling back to the current frame's, as above, on the
                # episode's first tick). Reusing (cos_now, sin_now) rather
                # than re-deriving them from velocity_idx[1] matters here --
                # that component is vy, not an angular rate, and treating it
                # as one (as a single len(velocity_idx)==2 check used to)
                # fabricates a nonexistent rotation from the ego's own
                # y-velocity.
                cos_prev, sin_prev = cos_now, sin_now
                v_now_vec = x0_batch[b, list(velocity_idx)]
                v_prev_vec = ego_proprio_prev_all[b] if has_prev_proprio else v_now_vec
                pos_prev = pos_now - 0.5 * dt * (v_prev_vec + v_now_vec)

            for j in range(self.neighbor_slots):
                # Neighbor slot j is fleet index j if it comes before this
                # robot (b) in fleet order, else j + 1 (skipping over b's own
                # index) -- e.g. robot 2's slot 0 is fleet robot 0, but its
                # slot 2 is fleet robot 3, not 2. The fleet contract only
                # requires equal dimensions/types across robots, not equal
                # numeric parameters (systems/multi_robot.py's per-robot
                # config), so a neighbor can have a different max_speed than
                # this robot -- forward-simulating it below with the wrong
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

                # mask[b, -2, j] guards against the previous frame being
                # zero-padding (either pre-episode warm-up or this neighbor
                # simply being invisible then) -- differencing against a
                # fabricated [0, 0] "previous position" would otherwise
                # produce a large fictitious velocity the instant a neighbor
                # first becomes visible.
                if observation_horizon >= 2 and mask[b, -2, j] > 0.5:
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
                    if observation_horizon >= 2 and mask[b, -2, j] > 0.5:
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
                neighbor_active = neighbor_active_batch[i] if neighbor_active_batch is not None else None
                return self.projectors[i].project(ego_obs_np[i], u_ref, neighbor_trajs, neighbor_active)

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
