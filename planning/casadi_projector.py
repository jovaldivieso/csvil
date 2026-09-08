import warnings
from typing import Any, Mapping

import casadi as ca
import numpy as np

from systems.dynamics import DynamicsProtocol


class CasadiTrajectoryProjector:
    """
    CasADi-based projection of a single robot's proposed action sequence onto
    its safety manifold, treating other robots as time-varying parametric
    obstacles rather than jointly-optimized fleet members.

    Modeled after ``planning.casadi_planner.CasadiPlanner``, restricted to one
    robot's own dynamics (decentralized: this is built once per ego robot,
    not once per fleet). The running cost pulls the optimized action sequence
    ``U`` toward a reference action sequence ``u_ref`` supplied by a learned
    policy (e.g. flow matching), instead of tracking a goal state throughout
    the horizon. Dynamics, action/state bounds are retained from
    ``CasadiPlanner``; pairwise fleet collision constraints are replaced by a
    soft ``d_safe`` constraint against externally-supplied neighbor position
    trajectories (``neighbor_traj_param``). The terminal cost is the
    control-invariant safety condition itself (SafeFlowMPC, Oelerich et al.,
    2026, Eq. 12): driving terminal velocity/angular-velocity toward zero so
    the horizon ends in a safe, controllable rest state -- not tracking the
    task goal, which the paper's own formulation never asks the projector to
    do (task-directed progress comes entirely from the flow policy's own
    proposal, ``u_ref``).
    """

    def _resolve_r_diag(self, config: Mapping[str, Any]) -> np.ndarray:
        """Per-robot action weight diagonal, sliced from a fleet-sized 'R_diag' if needed."""
        raw_r_diag = config.get("R_diag")
        if raw_r_diag is not None:
            if not isinstance(raw_r_diag, list):
                raise ValueError("'R_diag' must be a list.")
            # A fleet-level config's R_diag is laid out as consecutive per-robot
            # blocks in robot order; this robot's own block is the leading slice.
            diag_values = raw_r_diag[: self.sim.nu] if len(raw_r_diag) != self.sim.nu else raw_r_diag
            if len(diag_values) != self.sim.nu:
                raise ValueError(
                    f"'R_diag' must supply at least nu={self.sim.nu} values for this robot, got {len(raw_r_diag)}."
                )
            values: list[float] = []
            for idx, value in enumerate(diag_values):
                if not isinstance(value, (int, float)):
                    raise ValueError(f"'R_diag[{idx}]' must be numeric.")
                value_f = float(value)
                if value_f <= 0:
                    raise ValueError(f"'R_diag[{idx}]' must be positive.")
                values.append(value_f)
            return np.asarray(values, dtype=float)

        r_weight = config.get("R_weight", 0.1)
        if not isinstance(r_weight, (int, float)):
            raise ValueError("'R_weight' must be numeric.")
        if float(r_weight) <= 0:
            raise ValueError("'R_weight' must be positive.")
        return np.asarray([float(r_weight)] * self.sim.nu, dtype=float)

    def __init__(
        self,
        local_sim: DynamicsProtocol,
        config: Mapping[str, Any],
        neighbor_slots: int = 0,
        *,
        horizon: int,
    ):
        self.sim = local_sim
        # The *tracked* portion must match the flow policy's own
        # prediction_horizon exactly: project() takes the entire
        # flow-proposed action sequence and replaces it wholesale every
        # denoising step. config['horizon'] is a different, unrelated
        # quantity -- the expert CasADi MPC's own planning horizon used to
        # generate DAgger demonstrations -- even though it lives in the same
        # shared planner config, so it must never be read here.
        self.tracked_horizon = int(horizon)
        if self.tracked_horizon <= 0:
            raise ValueError("'horizon' must be a positive integer.")

        # The *internal* planning horizon extends beyond the tracked portion
        # with a free "coast to a stop" tail, long enough to actually
        # decelerate from this system's own max speed at its own max
        # action -- otherwise the terminal "come to rest" condition below
        # would be geometrically unreachable from any meaningful cruising
        # speed within just the short tracked_horizon, forcing the optimizer
        # to keep velocity far below what's actually safe just to stay
        # feasible (confirmed: this measurably capped speed regardless of
        # the terminal cost's weight, since it's a kinematic feasibility
        # limit, not a cost-tuning one). The tail isn't tracked against any
        # reference and isn't returned by project() -- it only needs to
        # exist so a genuine stopping maneuver is reachable.
        max_speed = getattr(self.sim, "max_speed", None)
        max_action_for_stopping = getattr(self.sim, "max_action", None)
        if max_speed is not None and max_action_for_stopping and float(max_action_for_stopping) > 0:
            coast_steps = int(np.ceil(float(max_speed) / float(max_action_for_stopping) / float(self.sim.dt)))
        else:
            coast_steps = 0
        self.N = self.tracked_horizon + coast_steps

        self.collision_slack_penalty_weight = float(config.get("collision_slack_penalty_weight", 10000.0))
        # Independent of Q_diag/terminal_cost_multiplier: those are tuned for
        # the expert planner's own goal-distance terminal cost (a much
        # longer horizon, tracking a possibly-distant position), which is a
        # different quantity at a different scale than "how much does a
        # small residual terminal velocity cost." Reusing them here made
        # even a tiny terminal velocity extremely expensive, so the cheapest
        # way to satisfy it was to barely accelerate in the first place.
        self.terminal_velocity_weight = float(config.get("terminal_velocity_weight", 1.0))

        if self.terminal_velocity_weight <= 0:
            raise ValueError("'terminal_velocity_weight' must be positive.")
        if self.collision_slack_penalty_weight <= 0:
            raise ValueError("'collision_slack_penalty_weight' must be positive.")

        # RTI-style bounded compute budget (SafeFlowMPC, Oelerich et al. 2026,
        # Sec. III): the paper never runs its projection to full non-convex
        # optimality either -- it solves exactly one QP per call via the
        # real-time-iteration scheme in acados, explicitly because doing so
        # at every one of the Ns flow steps is "computationally infeasible."
        # Their own "Ours w/ NL-Opt" ablation (solving to full optimality
        # instead) is both slower and far less predictable (Table I:
        # 93+-178 ms vs. 62+-5 ms) -- fatal for a hard real-time loop. IPOPT
        # has no single-QP mode, so this caps its iteration budget instead;
        # see project()'s handling of "Maximum_Iterations_Exceeded", which
        # accepts the truncated iterate rather than treating it as a failure,
        # relying on warm-starting across successive calls (every flow step,
        # every control tick) to keep refining it -- never on any single call
        # converging to completion.
        self.max_iter = int(config.get("projector_max_iter", 10))
        if self.max_iter <= 0:
            raise ValueError("'projector_max_iter' must be a positive integer.")

        self.neighbor_slots = int(neighbor_slots)
        if self.neighbor_slots < 0:
            raise ValueError("'neighbor_slots' must be non-negative.")

        self.d_safe = float(config.get("d_safe", getattr(self.sim, "d_safe", 0.0)))
        if self.d_safe < 0:
            raise ValueError("'d_safe' must be non-negative.")

        self.R = np.diag(self._resolve_r_diag(config))

        self.opti = ca.Opti()

        self.X = self.opti.variable(self.sim.nx, self.N + 1)
        self.U = self.opti.variable(self.sim.nu, self.N)

        self.x0_param = self.opti.parameter(self.sim.nx)
        self.u_ref_param = self.opti.parameter(self.sim.nu, self.tracked_horizon)

        cost = 0

        x_sym = ca.SX.sym("x", self.sim.nx)
        u_sym = ca.SX.sym("u", self.sim.nu)
        x_next_sym = self.sim.casadi_dynamics(x_sym, u_sym)
        step_fn = ca.Function("step", [x_sym, u_sym], [x_next_sym])
        F_map = step_fn.map(self.N)
        X_next = F_map(self.X[:, :-1], self.U)
        self.opti.subject_to(self.X[:, 1:] == X_next)

        robot_max_action = float(getattr(self.sim, "max_action"))
        self.opti.subject_to(ca.vec(self.U) >= -robot_max_action)
        self.opti.subject_to(ca.vec(self.U) <= robot_max_action)

        sub_lower = getattr(self.sim, "state_lower_bounds", None)
        sub_upper = getattr(self.sim, "state_upper_bounds", None)
        if sub_lower is not None:
            lower_bounds = np.asarray(sub_lower, dtype=float)
            finite_lower = np.isfinite(lower_bounds)
            if np.any(finite_lower):
                lower_indices = np.flatnonzero(finite_lower).tolist()
                lower_values = ca.repmat(
                    ca.reshape(ca.DM(lower_bounds[finite_lower]), len(lower_indices), 1), 1, self.N
                )
                self.opti.subject_to(ca.vec(self.X[lower_indices, 1:]) >= ca.vec(lower_values))
        if sub_upper is not None:
            upper_bounds = np.asarray(sub_upper, dtype=float)
            finite_upper = np.isfinite(upper_bounds)
            if np.any(finite_upper):
                upper_indices = np.flatnonzero(finite_upper).tolist()
                upper_values = ca.repmat(
                    ca.reshape(ca.DM(upper_bounds[finite_upper]), len(upper_indices), 1), 1, self.N
                )
                self.opti.subject_to(ca.vec(self.X[upper_indices, 1:]) <= ca.vec(upper_values))

        # Running cost: track the flow-proposed action sequence over the
        # tracked prefix, not a goal state. The coast tail (columns
        # tracked_horizon:N) has no reference to track -- it's free, subject
        # only to dynamics/bounds/collision-avoidance and the terminal
        # condition below, existing purely so a real stopping maneuver is
        # reachable.
        action_diff = self.U[:, : self.tracked_horizon] - self.u_ref_param
        cost += ca.sum2(ca.sum1(ca.mtimes(self.R, action_diff) * action_diff))

        # Dynamic obstacles: other robots as time-varying parametric constraints,
        # not jointly-optimized fleet members. CasADi parameters/variables are
        # strictly 2D, so the (neighbor_slots, 2, N+1) neighbor trajectory is
        # packed as (2*neighbor_slots, N+1): row 2j is neighbor j's x, row
        # 2j+1 its y (matches a C-order reshape of a (neighbor_slots, 2, N+1)
        # numpy array).
        self.neighbor_traj_param = None
        if self.neighbor_slots > 0 and self.d_safe > 0.0:
            self.neighbor_traj_param = self.opti.parameter(2 * self.neighbor_slots, self.N + 1)
            collision_slack = self.opti.variable(self.neighbor_slots, self.N + 1)
            self.opti.subject_to(ca.vec(collision_slack) >= 0)

            pos_idx = tuple(getattr(self.sim, "position_indices", (0, 1)))
            for j in range(self.neighbor_slots):
                diff_x = self.X[pos_idx[0], :] - self.neighbor_traj_param[2 * j, :]
                diff_y = self.X[pos_idx[1], :] - self.neighbor_traj_param[2 * j + 1, :]
                squared_distance = diff_x ** 2 + diff_y ** 2
                self.opti.subject_to(squared_distance + collision_slack[j, :] >= self.d_safe ** 2)

            cost += self.collision_slack_penalty_weight * ca.sum2(ca.sum1(collision_slack))

        # Terminal condition: come to a safe, controllable rest -- the
        # projector's actual terminal safety condition (SafeFlowMPC, Oelerich
        # et al. 2026, Eq. 12: zero velocity/acceleration/jerk at the
        # horizon's end), not the task goal. All task-directed progress
        # already comes from the flow network's own learned proposal
        # (u_ref, tracked by the running cost above); re-adding goal-tracking
        # here creates a competing incentive to race toward a possibly-
        # distant goal within just this short horizon, which can override
        # the network's own learned pacing and saturate the action bounds --
        # confirmed to cause collisions a plain, unprojected flow rollout did
        # not have. Systems with no velocity state at all (single_integrator,
        # unicycle1) have no momentum to carry them unsafely past the
        # horizon, so no terminal condition is needed for them.
        velocity_idx = tuple(getattr(self.sim, "velocity_state_indices", ()))
        if velocity_idx:
            terminal_velocity = self.X[list(velocity_idx), self.N]
            cost += self.terminal_velocity_weight * ca.sumsqr(terminal_velocity)

        self.opti.minimize(cost)
        self.opti.subject_to(self.X[:, 0] == self.x0_param)

        # Consecutive project() calls (across flow denoising steps, and across
        # control ticks) re-solve this *same* Opti problem with only slightly
        # perturbed parameters, so dual-warm-starting the multipliers in
        # project() is far more effective than a cold start every time. A
        # bare warm_start_init_point=yes alone barely helps -- IPOPT still
        # pushes iterates away from bounds and restarts the barrier parameter
        # high by default -- so the push/frac options and a small mu_init are
        # part of the standard recipe too.
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": self.max_iter,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1e-9,
            "ipopt.warm_start_bound_frac": 1e-9,
            "ipopt.warm_start_slack_bound_push": 1e-9,
            "ipopt.warm_start_slack_bound_frac": 1e-9,
            "ipopt.warm_start_mult_bound_push": 1e-9,
            "ipopt.mu_init": 1e-6,
        }
        self.opti.solver("ipopt", opts)
        self._prev_lam_g: np.ndarray | None = None

    def project(self, obs: np.ndarray, u_ref: np.ndarray, neighbor_trajs: np.ndarray | None = None) -> np.ndarray:
        """Project a reference action sequence ``u_ref`` (nu, tracked_horizon) onto the safety manifold.

        Returns the projected action sequence for just the tracked prefix
        (shape matching ``u_ref``); the internal coast tail used to make the
        terminal "come to rest" condition reachable is solved for but never
        returned or tracked against any reference.

        ``neighbor_trajs``, if this robot has neighbor slots and ``d_safe`` >
        0, must be shaped ``(neighbor_slots, 2, N + 1)`` in this robot's own
        *absolute* frame, where ``N`` is the full internal horizon
        (``self.N``, including the coast tail) -- not just ``tracked_horizon``.

        With ``self.max_iter`` capping IPOPT's budget (RTI-style), hitting
        that cap is the *expected* steady-state outcome, not a failure: the
        truncated iterate is accepted and returned, warm-starting the next
        call. Falls back to the unprojected ``u_ref`` with a warning only for
        a genuine solver failure (e.g. infeasibility), so that degrades
        gracefully instead of halting the caller's iterative flow-matching
        loop.
        """
        x0 = self.sim.invert_obs(obs)
        # Full-length (nu, N) warm-start guess: u_ref for the tracked prefix,
        # then a zero-action coast for the tail (a simple, adequate initial
        # guess -- the solver still has to find an actual deceleration
        # maneuver to satisfy the terminal condition; this just seeds it).
        u_guess = np.zeros((self.sim.nu, self.N), dtype=float)
        u_guess[:, : self.tracked_horizon] = u_ref
        state_guess = np.zeros((self.sim.nx, self.N + 1), dtype=float)
        state_guess[:, 0] = x0
        for step in range(self.N):
            state_guess[:, step + 1] = self.sim.predict_next_state(
                state_guess[:, step], u_guess[:, step], validate=False
            )

        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.u_ref_param, u_ref)
        if self.neighbor_traj_param is not None:
            if neighbor_trajs is None:
                raise ValueError("'neighbor_trajs' is required: this projector has neighbor_slots > 0 and d_safe > 0.")
            self.opti.set_value(
                self.neighbor_traj_param,
                np.asarray(neighbor_trajs, dtype=float).reshape(2 * self.neighbor_slots, self.N + 1),
            )
        self.opti.set_initial(self.X, state_guess)
        self.opti.set_initial(self.U, u_guess)
        if self._prev_lam_g is not None:
            self.opti.set_initial(self.opti.lam_g, self._prev_lam_g)

        try:
            sol = self.opti.solve()
            self._prev_lam_g = sol.value(self.opti.lam_g)
            return sol.value(self.U)[:, : self.tracked_horizon]
        except RuntimeError as exc:
            # IPOPT raises even when it merely exhausted its (deliberately
            # small) iteration budget, not just on genuine failures --
            # opti.stats() still reports which one just happened.
            if self.opti.stats().get("return_status") == "Maximum_Iterations_Exceeded":
                u_result = self.opti.debug.value(self.U)[:, : self.tracked_horizon]
                lam_g_result = np.asarray(self.opti.debug.value(self.opti.lam_g))
                if np.all(np.isfinite(u_result)) and np.all(np.isfinite(lam_g_result)):
                    # This is the normal RTI operating mode, not a degraded
                    # one -- no warning, and warm-start from it like any
                    # other solve so the next call keeps refining it.
                    self._prev_lam_g = lam_g_result
                    return u_result
            warnings.warn(
                f"CasadiTrajectoryProjector solve failed, falling back to the unprojected reference "
                f"action sequence: {exc}",
                stacklevel=2,
            )
            return u_ref
