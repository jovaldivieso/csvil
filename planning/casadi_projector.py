import warnings
from typing import Any, Mapping

import casadi as ca
import numpy as np

from planning.casadi_planner import PlannerSolveError
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
    ``CasadiPlanner``; pairwise fleet collision avoidance against externally-
    supplied neighbor position trajectories (``neighbor_traj_param``) is a
    soft ``d_safe`` buffer distance (relaxable via slack, cheaply penalized)
    with a hard ``d_collision`` floor beneath it (never relaxable -- see
    ``__init__``) so slack can eat into the safety margin but never into a
    modeled physical collision. This floor applies only to the neighbor
    trajectories supplied to the projector. The terminal condition is the control-invariant safety
    condition itself (SafeFlowMPC, Oelerich et al., 2026, Eq. 12), enforced
    as the hard equality constraint the paper states: terminal
    velocity/angular-velocity must be exactly zero so the horizon ends in a
    safe, controllable rest state -- not tracking the task goal, which the
    paper's own formulation never asks the projector to do (task-directed
    progress comes entirely from the flow policy's own proposal, ``u_ref``).
    First-order systems (no velocity state) need no terminal constraint at
    all: Assumption 1's control-invariant-safety-set requirement is
    trivially satisfied everywhere for a driftless first-order system (u=0
    is a fixed point at any state), not skipped as a deviation from it.
    """

    def _resolve_r_diag(self, config: Mapping[str, Any], robot_index: int) -> np.ndarray:
        """Per-robot action weight diagonal, sliced from a fleet-sized 'R_diag' if needed."""
        raw_r_diag = config.get("R_diag")
        if raw_r_diag is not None:
            if not isinstance(raw_r_diag, list):
                raise ValueError("'R_diag' must be a list.")
            if len(raw_r_diag) == self.sim.nu:
                # Already sized for exactly one robot (e.g. a fleet-of-one) --
                # nothing to slice.
                diag_values = raw_r_diag
            else:
                # A fleet-level config's R_diag is laid out as consecutive
                # per-robot blocks in robot order; take this robot's own
                # block, not always the leading one.
                start = robot_index * self.sim.nu
                diag_values = raw_r_diag[start : start + self.sim.nu]
            if len(diag_values) != self.sim.nu:
                raise ValueError(
                    f"'R_diag' must supply at least nu={self.sim.nu} values for robot {robot_index} "
                    f"(fleet-wide R_diag has {len(raw_r_diag)} values total)."
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
        robot_index: int = 0,
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

        # The internal planning horizon matches the tracked portion exactly
        # (SafeFlowMPC, Oelerich et al. 2026, Eq. 12): the terminal "come to
        # rest" condition is enforced at the end of this same N-step horizon,
        # not on an appended tail beyond it, as the hard equality constraint
        # the paper states (see the terminal-velocity block below). A
        # too-short horizon can therefore make the solve genuinely
        # infeasible -- not just weaken a soft penalty -- if it cannot
        # decelerate from this system's own worst-case velocity to zero at
        # its own max_action; the construction-time check below raises
        # before that can happen silently at runtime.
        self.N = self.tracked_horizon

        self.collision_slack_penalty_weight = float(config.get("collision_slack_penalty_weight", 10000.0))
        if self.collision_slack_penalty_weight <= 0:
            raise ValueError("'collision_slack_penalty_weight' must be positive.")

        self.neighbor_slots = int(neighbor_slots)
        if self.neighbor_slots < 0:
            raise ValueError("'neighbor_slots' must be non-negative.")

        self.d_safe = float(config.get("d_safe", getattr(self.sim, "d_safe", 0.0)))
        if self.d_safe < 0:
            raise ValueError("'d_safe' must be non-negative.")

        # d_safe is the soft planning buffer (may be relaxed via slack below
        # when unavoidable); d_collision is the hard physical floor beneath
        # it that must never be crossed regardless of slack.
        self.d_collision = float(config.get("d_collision", getattr(self.sim, "d_collision", self.d_safe)))
        if self.d_collision < 0:
            raise ValueError("'d_collision' must be non-negative.")
        if self.d_collision > self.d_safe:
            raise ValueError(
                "'d_collision' must not exceed 'd_safe': d_safe is the soft planning buffer "
                "distance and d_collision is the hard physical floor beneath it."
            )

        # How close to exactly zero a re-simulated fallback trajectory's
        # terminal velocity must be to still count as "coasting to a safe
        # rest" (see _revalidate_fallback) -- not the NLP's own solver
        # tolerance, which the hard terminal_velocity == 0 constraint below
        # already enforces for an actual solve.
        self.fallback_terminal_velocity_tol = float(config.get("fallback_terminal_velocity_tol", 1e-2))
        if self.fallback_terminal_velocity_tol < 0:
            raise ValueError("'fallback_terminal_velocity_tol' must be non-negative.")

        self.R = np.diag(self._resolve_r_diag(config, robot_index))

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

        # Running cost: track the flow-proposed action sequence against
        # u_ref over the entire horizon, not a goal state.
        action_diff = self.U[:, : self.tracked_horizon] - self.u_ref_param
        cost += ca.sum2(ca.sum1(ca.mtimes(self.R, action_diff) * action_diff))

        # Dynamic obstacles: other robots as time-varying parametric constraints,
        # not jointly-optimized fleet members. CasADi parameters/variables are
        # strictly 2D, so the (neighbor_slots, 2, N+1) neighbor trajectory is
        # packed as (2*neighbor_slots, N+1): row 2j is neighbor j's x, row
        # 2j+1 its y (matches a C-order reshape of a (neighbor_slots, 2, N+1)
        # numpy array).
        self.pos_idx = tuple(getattr(self.sim, "position_indices", (0, 1)))

        self.neighbor_traj_param = None
        if self.neighbor_slots > 0 and self.d_safe > 0.0:
            self.neighbor_traj_param = self.opti.parameter(2 * self.neighbor_slots, self.N + 1)
            collision_slack = self.opti.variable(self.neighbor_slots, self.N + 1)
            self.opti.subject_to(ca.vec(collision_slack) >= 0)

            for j in range(self.neighbor_slots):
                diff_x = self.X[self.pos_idx[0], :] - self.neighbor_traj_param[2 * j, :]
                diff_y = self.X[self.pos_idx[1], :] - self.neighbor_traj_param[2 * j + 1, :]
                squared_distance = diff_x ** 2 + diff_y ** 2
                # Soft target: stay at or beyond the d_safe planning buffer
                # whenever possible, relaxable via collision_slack when it
                # isn't (mirrors CasadiPlanner's own expert-side handling).
                self.opti.subject_to(squared_distance + collision_slack[j, :] >= self.d_safe ** 2)
                # Hard floor: collision_slack is unbounded above, so the soft
                # term alone never actually guarantees separation -- an
                # expensive-but-feasible solve could accept slack past the
                # buffer and into real contact. d_collision (<= d_safe,
                # enforced in __init__) is the actual physical contact
                # threshold, so it stays a hard constraint regardless of how
                # much slack the soft d_safe term takes on.
                self.opti.subject_to(squared_distance >= self.d_collision ** 2)

            cost += self.collision_slack_penalty_weight * ca.sum2(ca.sum1(collision_slack))

        # Terminal condition: come to a safe, controllable rest -- the
        # projector's actual terminal safety condition (SafeFlowMPC, Oelerich
        # et al. 2026, Eq. 12: zero velocity/acceleration/jerk at the
        # horizon's end), enforced as the hard equality constraint the paper
        # states rather than a soft cost: a soft penalty lets the optimizer
        # trade residual terminal motion for tracking u_ref, which does not
        # actually establish the invariant resting trajectory the fallback
        # in project() relies on (Theorem 2). Not tracking the task goal:
        # all task-directed progress already comes from the flow network's
        # own learned proposal (u_ref, tracked by the running cost above);
        # re-adding goal-tracking here creates a competing incentive to race
        # toward a possibly-distant goal within just this short horizon,
        # which can override the network's own learned pacing and saturate
        # the action bounds -- confirmed to cause collisions a plain,
        # unprojected flow rollout did not have.
        #
        # Eq. 12 zeros every state-derivative level except position -- for
        # the paper's own manipulator (state = [q, q_dot, q_ddot, jerk]),
        # that's velocity, acceleration, and jerk, leaving the control (the
        # *next* derivative, "snap") unconstrained at the boundary. This
        # system's state chain is only [position, velocity]; control
        # (acceleration) already fills that same "next derivative" slot, so
        # the correct analog is to zero velocity alone and leave the control
        # free -- constraining the control too would be *more* restrictive
        # than the paper's own condition, not more faithful to it.
        #
        # Systems with no velocity state at all (single_integrator,
        # unicycle1) genuinely need no constraint here -- not a deviation
        # from the paper, but Assumption 1 ("a controller exists with a
        # control-invariant safety set encompassing the terminal set")
        # being trivially satisfied everywhere for a driftless first-order
        # system: state_{k+1} = state_k + u*dt, so u=0 is a fixed point at
        # *any* state, not just one specially reached. There is nothing to
        # "come to rest" from -- a first-order system is always already
        # instantaneously stoppable.
        self.velocity_idx = tuple(getattr(self.sim, "velocity_state_indices", ()))
        if self.velocity_idx:
            terminal_velocity = self.X[list(self.velocity_idx), self.N]
            self.opti.subject_to(terminal_velocity == 0)

            # Feasibility sanity check at construction time rather than only
            # discovering it via opaque runtime solve failures: this hard
            # constraint is only satisfiable if the horizon is long enough
            # to decelerate from this system's own worst-case velocity to
            # zero at its own max_action.
            dt = float(getattr(self.sim, "dt"))
            robot_max_action = float(getattr(self.sim, "max_action"))
            state_upper = getattr(self.sim, "state_upper_bounds", None)
            state_lower = getattr(self.sim, "state_lower_bounds", None)
            if state_upper is not None and state_lower is not None and robot_max_action > 0:
                state_upper = np.asarray(state_upper, dtype=float)
                state_lower = np.asarray(state_lower, dtype=float)
                worst_case_speeds = [
                    max(abs(state_upper[idx]), abs(state_lower[idx]))
                    for idx in self.velocity_idx
                    if np.isfinite(state_upper[idx]) and np.isfinite(state_lower[idx])
                ]
                if worst_case_speeds:
                    min_feasible_horizon = max(worst_case_speeds) / (robot_max_action * dt)
                    if self.N < min_feasible_horizon:
                        raise ValueError(
                            f"'horizon' ({self.N}) is too short for {type(self.sim).__name__} to "
                            "always satisfy the hard terminal-velocity constraint: decelerating from "
                            f"its worst-case velocity ({max(worst_case_speeds)}) at max_action "
                            f"({robot_max_action}) needs at least {min_feasible_horizon:.1f} steps at "
                            f"dt={dt}. Increase 'horizon' (equivalently, the flow policy's own "
                            "prediction_horizon) -- otherwise this constraint can be infeasible from a "
                            "reachable state."
                        )

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
        # Primal warm-start state, mirroring CasadiPlanner's own MPC
        # recipe (see reset()/project() below for how these are used).
        self._prev_X_sol: np.ndarray | None = None
        self._prev_U_sol: np.ndarray | None = None
        self._prev_x0: np.ndarray | None = None

    def reset(self) -> None:
        """Signals the start of a new episode.

        Without this, a stale solution from the end of the *previous*
        episode would otherwise be shifted and reused as this episode's
        first warm start in project() -- a worse guess than the cold-start
        fallback (forward-simulating u_ref), since it reflects a completely
        unrelated trajectory rather than nothing at all.
        """
        self._prev_lam_g = None
        self._prev_X_sol = None
        self._prev_U_sol = None
        self._prev_x0 = None
        # set_initial(..., lam_g, ...) is sticky on the Opti object -- clearing
        # only the Python-side _prev_lam_g cache above stops project() from
        # re-applying it, but whatever value the last pre-reset call handed
        # Opti stays active internally until something overwrites it. Without
        # this, the new episode's first solve would still warm-start from the
        # previous (unrelated) episode's final dual multipliers.
        self.opti.set_initial(self.opti.lam_g, 0.0)

    def _revalidate_fallback(
        self, x0: np.ndarray, u_guess: np.ndarray, neighbor_trajs: np.ndarray | None
    ) -> np.ndarray | None:
        """Re-derive and re-check a candidate fallback trajectory against the *current* tick.

        The cached ``_prev_X_sol`` this is shifted from was only ever
        verified against the state/neighbor forecast at the time it was
        solved -- the actual current ``x0`` can have drifted from what that
        solve predicted (execution noise, model mismatch), and
        ``neighbor_trajs`` for *this* call can differ from what the cached
        plan assumed. Accepting the stale trajectory on faith would let a
        run of solve failures return actions that no longer actually respect
        bounds, the current neighbor forecast, or the terminal rest
        condition.

        Re-simulates ``u_guess`` (already shifted/padded by the caller) from
        the real ``x0`` via ``predict_next_state`` -- this is the only
        dynamics info available without another NLP solve -- and returns the
        resulting state trajectory if it still respects state bounds, the
        current neighbor forecast's hard ``d_collision`` floor, and ends
        acceptably close to the terminal rest condition; ``None`` if any of
        those no longer hold.
        """
        state_guess = np.zeros((self.sim.nx, self.N + 1), dtype=float)
        state_guess[:, 0] = x0
        for step in range(self.N):
            state_guess[:, step + 1] = self.sim.predict_next_state(
                state_guess[:, step], u_guess[:, step], validate=False
            )

        tolerance = 1e-6
        lower_bounds = getattr(self.sim, "state_lower_bounds", None)
        if lower_bounds is not None:
            lower_bounds = np.asarray(lower_bounds, dtype=float)
            finite = np.isfinite(lower_bounds)
            if np.any(finite) and np.any(
                state_guess[finite][:, 1:] < lower_bounds[finite][:, None] - tolerance
            ):
                return None
        upper_bounds = getattr(self.sim, "state_upper_bounds", None)
        if upper_bounds is not None:
            upper_bounds = np.asarray(upper_bounds, dtype=float)
            finite = np.isfinite(upper_bounds)
            if np.any(finite) and np.any(
                state_guess[finite][:, 1:] > upper_bounds[finite][:, None] + tolerance
            ):
                return None

        if self.neighbor_traj_param is not None and neighbor_trajs is not None:
            neighbor_trajs = np.asarray(neighbor_trajs, dtype=float)
            robot_pos = state_guess[list(self.pos_idx), :]
            for j in range(self.neighbor_slots):
                diff = robot_pos - neighbor_trajs[j]
                squared_distance = np.sum(diff ** 2, axis=0)
                if np.any(squared_distance < self.d_collision ** 2 - tolerance):
                    return None

        terminal_velocity = state_guess[list(self.velocity_idx), self.N]
        if np.any(np.abs(terminal_velocity) > self.fallback_terminal_velocity_tol):
            return None

        return state_guess

    def project(self, obs: np.ndarray, u_ref: np.ndarray, neighbor_trajs: np.ndarray | None = None) -> np.ndarray:
        """Project a reference action sequence ``u_ref`` (nu, tracked_horizon) onto the safety manifold.

        Returns the projected action sequence, shape matching ``u_ref``. The
        internal optimization horizon ``self.N`` equals ``tracked_horizon``
        exactly (SafeFlowMPC, Oelerich et al. 2026, Eq. 12: the terminal
        "come to rest" condition is enforced at the end of this same
        N-step horizon, not on an appended tail beyond it).

        ``neighbor_trajs``, if this robot has neighbor slots and ``d_safe`` >
        0, must be shaped ``(neighbor_slots, 2, N + 1)``, i.e.
        ``(neighbor_slots, 2, tracked_horizon + 1)``, in this robot's own
        *absolute* frame. If the NLP solve fails and a previously-solved safe
        trajectory exists, falls back to that trajectory (SafeFlowMPC,
        Oelerich et al. 2026, Theorem 2) only after ``_revalidate_fallback``
        confirms it still respects bounds, the hard ``d_collision`` floor
        against *this call's* neighbor forecast, and the terminal rest
        condition from the *actual current* state -- never blindly, since
        the cached trajectory was only ever verified against the state/
        neighbor forecast at the time it was solved. Never falls back to the
        unprojected ``u_ref`` either way -- returning an unprojected action
        on a transient solver hiccup would bypass every constraint this
        projector exists to enforce. On a cold-start failure (first call
        ever, or right after ``reset()``) or when the cached trajectory no
        longer re-validates, there is no established safe trajectory to fall
        back on -- exactly the situation SafeFlowMPC's Assumption 2 assumes
        away by requiring one to already exist -- so this raises
        ``planning.casadi_planner.PlannerSolveError`` instead of fabricating
        a fallback with no safety basis. Callers should handle it the same
        way they already handle an expert ``PlannerSolveError`` (discard or
        retry the episode), not attempt to recover an action from it.

        IPOPT is always run to full convergence here (SafeFlowMPC, Oelerich
        et al. 2026, Sec. III uses this exact "Ours w/ NL-Opt" mode when
        using IPOPT rather than acados/RTI): an early-terminated interior-
        point iterate is not a QP solved to its own optimum the way one RTI
        step is, so it carries no guarantee of satisfying dynamics, bounds,
        or collision constraints -- accepting it would silently break the
        safety guarantee this projector exists to provide. Do not cap
        ``ipopt.max_iter``; the only faithful way to trade off compute here
        is fewer flow steps (``num_inference_steps``), exactly as the paper's
        own IPOPT ablation reduces ``Ns`` from 7 to 4 rather than truncating
        any individual solve.
        """
        x0 = self.sim.invert_obs(obs)
        # Primal warm-start guess, mirroring CasadiPlanner's own MPC recipe
        # but adapted to project()'s two distinct call patterns:
        #  - Consecutive flow denoising steps within the same control tick
        #    share the exact same x0 (the environment hasn't stepped yet;
        #    only u_ref, the flow network's reference, has been refined) --
        #    detected via exact equality, since obs/invert_obs are
        #    deterministic and the caller passes the same array every flow
        #    step. No time has elapsed, so the previous converged solution
        #    is reused as-is, unshifted -- a *better*-justified warm start
        #    than CasadiPlanner's shift, precisely because nothing needs
        #    shifting here.
        #  - A genuinely new control tick (x0 has actually moved) gets
        #    CasadiPlanner's shift-by-one-step treatment: the state/action
        #    previously planned for index 1 is now approximately index 0.
        # Falls back to forward-simulating u_ref (the only information
        # available yet) on a true cold start -- first call ever, or right
        # after reset().
        if self._prev_X_sol is not None and self._prev_U_sol is not None:
            if self._prev_x0 is not None and np.array_equal(x0, self._prev_x0):
                state_guess = self._prev_X_sol
                u_guess = self._prev_U_sol
            else:
                # Padding beyond the previous solve's own horizon represents
                # time *after* it already reached the certified at-rest
                # terminal state -- per Assumption 1 / Eq. 10 (Oelerich et
                # al. 2026), staying there is what "keeps the robot in the
                # terminal safety set" means. For this dynamics class that
                # controller is exactly zero action: with terminal_velocity
                # == 0 (hard constraint above), applying u=0 from that state
                # is a fixed point of predict_next_state. Padding with the
                # *last applied* action instead (whatever nonzero
                # deceleration drove velocity to zero) would reapply that
                # deceleration to an already-at-rest state and accelerate it
                # straight back away from rest -- exactly the "failure
                # recovery does not establish an invariant resting
                # trajectory" gap a stale plan could otherwise fall into.
                state_guess = np.hstack([self._prev_X_sol[:, 1:], self._prev_X_sol[:, -1:]])
                u_guess = np.hstack(
                    [self._prev_U_sol[:, 1:], np.zeros((self.sim.nu, 1), dtype=float)]
                )
        else:
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
            try:
                sol = self.opti.solve()
            except RuntimeError:
                # The shifted warm start can occasionally leave IPOPT stuck
                # at a locally infeasible point for the hard, non-convex
                # d_collision keep-out constraint -- e.g. once the actual
                # state, often at near-saturated velocity, has drifted from
                # what the previous solve predicted -- even though a
                # feasible trajectory exists (confirmed empirically against
                # the expert CasadiPlanner, which shares this exact
                # constraint pattern: a cold restart from the same x0/goal
                # resolved every reproduced failure). A cold restart here is
                # strictly better than reaching for the fallback trajectory
                # below when it works: continued genuine progress instead of
                # coasting on an old plan, with no new safety risk (the
                # solution still has to satisfy every hard constraint this
                # Opti problem enforces, cold-started or not).
                self.opti.set_initial(self.X, 0.0)
                self.opti.set_initial(self.U, 0.0)
                self.opti.set_initial(self.opti.lam_g, 0.0)
                sol = self.opti.solve()
            self._prev_lam_g = sol.value(self.opti.lam_g)
            self._prev_X_sol = sol.value(self.X)
            self._prev_U_sol = sol.value(self.U)
            self._prev_x0 = x0
            return self._prev_U_sol[:, : self.tracked_horizon]
        except RuntimeError as exc:
            if self._prev_U_sol is not None:
                # SafeFlowMPC, Oelerich et al. 2026, Theorem 2: on a
                # projection failure, the safe fallback is to keep executing
                # the current/last trajectory, not an arbitrary unprojected
                # action -- it stays safe for all t > t0+T by Theorem 1. But
                # u_guess/state_guess above were only ever verified against
                # the state/neighbor forecast *at the time they were solved*
                # -- the actual current x0 can have drifted from what was
                # predicted, and neighbor_trajs for *this* call can differ
                # from what the cached plan assumed. _revalidate_fallback
                # re-simulates from the real x0 and re-checks bounds,
                # d_collision against the current neighbor forecast, and the
                # terminal rest condition before trusting it.
                revalidated_state = self._revalidate_fallback(x0, u_guess, neighbor_trajs)
                if revalidated_state is not None:
                    warnings.warn(
                        "CasadiTrajectoryProjector solve failed, falling back to the last "
                        "known-safe projected trajectory (re-validated against the current "
                        f"state and neighbor forecast) instead of the unprojected reference: {exc}",
                        stacklevel=2,
                    )
                    self._prev_X_sol = revalidated_state
                    self._prev_U_sol = u_guess
                    self._prev_x0 = x0
                    return u_guess[:, : self.tracked_horizon]
                # The cached plan no longer holds from here -- clear it so a
                # subsequent call doesn't keep trying to reuse/shift a
                # trajectory already known to no longer be safe.
                self._prev_lam_g = None
                self._prev_X_sol = None
                self._prev_U_sol = None
                self._prev_x0 = None
            # SafeFlowMPC's safety guarantee (Theorem 2) rests entirely on
            # Assumption 2 -- "a safe trajectory q^0(t) exists at the start
            # of the robot movement" -- and Algorithm 1 takes that initial
            # safe trajectory as a hard *precondition*, never something the
            # algorithm derives from nothing. Neither a true cold start nor a
            # cached plan that no longer re-validates has a paper-faithful
            # safe action to fall back on here. Silently returning the
            # unprojected u_ref, or a stale plan that no longer verifiably
            # holds, would misrepresent an unverified action as a handled
            # failure, so this fails loudly instead -- callers are expected
            # to treat it like an expert PlannerSolveError (discard/retry
            # this episode), not attempt to recover an action from it.
            raise PlannerSolveError(
                "CasadiTrajectoryProjector solve failed with no revalidated safe trajectory to "
                f"fall back on: {exc}"
            ) from exc
