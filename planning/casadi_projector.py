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
        # not on an appended tail beyond it. This is a soft cost here (not a
        # hard equality constraint as in the paper), so a short horizon does
        # not make the solve infeasible -- it just means the achievable
        # terminal velocity may be far from zero whenever the tracked
        # horizon is too short to decelerate from cruising speed at this
        # system's own max action, weakening the safety margin the terminal
        # condition is meant to provide.
        self.N = self.tracked_horizon

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

        self.neighbor_slots = int(neighbor_slots)
        if self.neighbor_slots < 0:
            raise ValueError("'neighbor_slots' must be non-negative.")

        self.d_safe = float(config.get("d_safe", getattr(self.sim, "d_safe", 0.0)))
        if self.d_safe < 0:
            raise ValueError("'d_safe' must be non-negative.")

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
        Oelerich et al. 2026, Theorem 2), never to the unprojected ``u_ref`` --
        returning an unprojected action on a transient solver hiccup would
        bypass every constraint this projector exists to enforce. On a
        cold-start failure (first call ever, or right after ``reset()``),
        there is no established safe trajectory to fall back on at all --
        exactly the situation SafeFlowMPC's Assumption 2 assumes away by
        requiring one to already exist -- so this raises
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
                state_guess = np.hstack([self._prev_X_sol[:, 1:], self._prev_X_sol[:, -1:]])
                u_guess = np.hstack([self._prev_U_sol[:, 1:], self._prev_U_sol[:, -1:]])
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
                # action -- it stays safe for all t > t0+T by Theorem 1.
                # u_guess/state_guess above are exactly that trajectory
                # (reused unshifted for a same-tick retry, or shifted by one
                # step for a new tick), so accept them as though they had
                # been solved: this both returns a certified-safe action now
                # and lets a run of consecutive failures keep coasting
                # further along the same safe trajectory instead of
                # re-returning an identical fallback every time.
                warnings.warn(
                    "CasadiTrajectoryProjector solve failed, falling back to the last "
                    f"known-safe projected trajectory instead of the unprojected reference: {exc}",
                    stacklevel=2,
                )
                self._prev_X_sol = state_guess
                self._prev_U_sol = u_guess
                self._prev_x0 = x0
                return u_guess[:, : self.tracked_horizon]
            # SafeFlowMPC's safety guarantee (Theorem 2) rests entirely on
            # Assumption 2 -- "a safe trajectory q^0(t) exists at the start
            # of the robot movement" -- and Algorithm 1 takes that initial
            # safe trajectory as a hard *precondition*, never something the
            # algorithm derives from nothing. A cold start with no prior
            # solve to fall back on is exactly the situation the paper
            # assumes away; there is no paper-faithful safe action to
            # return here. Silently returning the unprojected u_ref would
            # misrepresent an unverified action as a handled failure, so
            # this fails loudly instead -- callers are expected to treat it
            # like an expert PlannerSolveError (discard/retry this
            # episode), not attempt to recover an action from it.
            raise PlannerSolveError(
                "CasadiTrajectoryProjector solve failed with no previous safe trajectory to "
                f"fall back on (cold start -- first call, or right after reset()): {exc}"
            ) from exc
