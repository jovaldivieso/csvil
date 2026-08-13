import casadi as ca
import numpy as np
from typing import Any, Mapping

from .planner import Planner
from systems.dynamics import DynamicsProtocol


class PlannerSolveError(RuntimeError):
    """Raised when the CasADi optimization problem cannot be solved."""


class CasadiPlanner(Planner):
    """
    CasADi-based finite-horizon optimal control for single-robot and multi-robot systems.

    Modes:
        - "mpc" (Receding Horizon): Solves the NLP at every step and executes
          only the first action. Stateless between steps.
        - "open_loop" (Trajectory Optimization): Solves the NLP once at the
          start of the episode, caches the trajectory, and executes it blindly.

    Mathematical Formulation
    ------------------------
    The planner treats any simulator as a fleet of robots. A standard simulator
    is interpreted as a fleet of one. For each robot $i$ with state $x_i$ and
    action $u_i$, and horizon $N$, the optimizer solves:

    J(X, U) = sum over k = 0..N-1, sum over robots i of:
        state_cost(i, k) = (x_i[k] - x_goal_i)^T Q_i (x_i[k] - x_goal_i)
        control_cost(i, k) = u_i[k]^T R_i u_i[k]

    plus terminal_cost = sum over robots i of:
        terminal_cost(i) = (x_i[N] - x_goal_i)^T (alpha * Q_i) (x_i[N] - x_goal_i)

    where alpha is ``terminal_cost_multiplier``.

    Subject to, for every k = 0..N-1:

        x[k + 1] = f(x[k], u[k])

    together with per-robot action limits, optional per-robot state bounds, and
    the pairwise collision-avoidance constraints:

        (x_i[k] - x_j[k])^2 + (y_i[k] - y_j[k])^2 >= d_safe^2   for all i < j

    ``d_safe`` is read from planner config, falling back to simulator config.
    For single-robot systems the pairwise collision constraints are inactive.

    Decision variables:
        X : Global state trajectory over the horizon.
        U : Global control trajectory over the horizon.

    Parameters:
        x_init : Current observed global state.
        x_goal : Global goal state assembled from all robot goals.
        Q_i : Per-robot state penalty blocks extracted from ``Q_diag``.
        R_i : Per-robot control penalty blocks.
        d_safe : Minimum allowed planar distance between any robot pair.
        collision_slack_penalty_weight : Positive weight for collision slack penalties.

    The solver keeps the fleet jointly feasible by applying the shared dynamics,
    per-robot action bounds, optional per-robot state bounds, and the pairwise
    collision constraints above at every step in the horizon.

    Control penalty configuration
    -----------------------------
    The action penalty matrix ``R`` is diagonal and can be configured in three
    equivalent ways (highest priority first):

    1. ``R_weight_per_robot`` (multi-robot only): list with one entry per robot.
       Each entry can be a scalar (isotropic per robot) or a full per-action list.
    2. ``R_diag``: full global diagonal list of length ``nu``.
    3. ``R_weight``: positive scalar fallback applied to all action dimensions.
    """

    def _resolve_r_diag(self, config: Mapping[str, Any]) -> np.ndarray:
        raw_r_per_robot = config.get("R_weight_per_robot")
        raw_r_diag = config.get("R_diag")

        per_robot_diag: np.ndarray | None = None
        explicit_diag: np.ndarray | None = None

        if raw_r_per_robot is not None:
            if not isinstance(raw_r_per_robot, list):
                raise ValueError("'R_weight_per_robot' must be a list.")
            if len(raw_r_per_robot) != len(self.robot_action_slices):
                raise ValueError(
                    "'R_weight_per_robot' length must match number of robots "
                    f"({len(self.robot_action_slices)}), got {len(raw_r_per_robot)}."
                )

            diag_values: list[float] = []
            for robot_idx, (entry, action_slice) in enumerate(zip(raw_r_per_robot, self.robot_action_slices)):
                action_dim = int(action_slice.stop - action_slice.start)
                if isinstance(entry, (int, float)):
                    value = float(entry)
                    if value <= 0:
                        raise ValueError(
                            f"'R_weight_per_robot[{robot_idx}]' must be positive."
                        )
                    diag_values.extend([value] * action_dim)
                elif isinstance(entry, list):
                    if len(entry) != action_dim:
                        raise ValueError(
                            f"'R_weight_per_robot[{robot_idx}]' must have length "
                            f"{action_dim}, got {len(entry)}."
                        )
                    for weight_idx, value in enumerate(entry):
                        if not isinstance(value, (int, float)):
                            raise ValueError(
                                f"'R_weight_per_robot[{robot_idx}][{weight_idx}]' must be numeric."
                            )
                        value_f = float(value)
                        if value_f <= 0:
                            raise ValueError(
                                f"'R_weight_per_robot[{robot_idx}][{weight_idx}]' must be positive."
                            )
                        diag_values.append(value_f)
                else:
                    raise ValueError(
                        f"'R_weight_per_robot[{robot_idx}]' must be numeric or a list of length {action_dim}."
                    )

            if len(diag_values) != self.sim.nu:
                raise ValueError(
                    "Expanded 'R_weight_per_robot' length must equal action dimension "
                    f"nu={self.sim.nu}, got {len(diag_values)}."
                )
            per_robot_diag = np.asarray(diag_values, dtype=float)

        if raw_r_diag is not None:
            if not isinstance(raw_r_diag, list):
                raise ValueError("'R_diag' must be a list.")
            if len(raw_r_diag) != self.sim.nu:
                raise ValueError(
                    f"'R_diag' length must equal simulator action dimension nu={self.sim.nu}, got {len(raw_r_diag)}."
                )
            diag_values = []
            for idx, value in enumerate(raw_r_diag):
                if not isinstance(value, (int, float)):
                    raise ValueError(f"'R_diag[{idx}]' must be numeric.")
                value_f = float(value)
                if value_f <= 0:
                    raise ValueError(f"'R_diag[{idx}]' must be positive.")
                diag_values.append(value_f)
            explicit_diag = np.asarray(diag_values, dtype=float)

        if per_robot_diag is not None and explicit_diag is not None:
            if not np.allclose(per_robot_diag, explicit_diag, rtol=0.0, atol=0.0):
                raise ValueError(
                    "'R_weight_per_robot' and 'R_diag' are both provided but inconsistent."
                )
            return per_robot_diag

        if per_robot_diag is not None:
            return per_robot_diag

        if explicit_diag is not None:
            return explicit_diag

        r_weight = config.get("R_weight", 0.1)
        if not isinstance(r_weight, (int, float)):
            raise ValueError("'R_weight' must be numeric.")
        if float(r_weight) <= 0:
            raise ValueError("'R_weight' must be positive.")
        return np.asarray([float(r_weight)] * self.sim.nu, dtype=float)

    def __init__(self, simulator: DynamicsProtocol, config: Mapping[str, Any]):
        self.sim = simulator
        self.N = config.get("horizon", 20)
        self.mode = config.get("mode", "mpc")  # Default to MPC
        self.terminal_cost_multiplier = float(config.get("terminal_cost_multiplier", 10.0))
        self.collision_slack_penalty_weight = float(
            config.get("collision_slack_penalty_weight", 10000.0)
        )

        if self.N <= 0:
            raise ValueError("'horizon' must be a positive integer.")
        if self.mode not in {"mpc", "open_loop"}:
            raise ValueError("'mode' must be one of {'mpc', 'open_loop'}.")
        if self.terminal_cost_multiplier <= 0:
            raise ValueError("'terminal_cost_multiplier' must be positive.")
        if self.collision_slack_penalty_weight <= 0:
            raise ValueError("'collision_slack_penalty_weight' must be positive.")

        self.sub_simulators = self.sim.simulators
        self.robot_state_slices = self.sim.robot_state_slices
        self.robot_action_slices = self.sim.robot_action_slices
        self.d_safe = float(config.get("d_safe", getattr(self.sim, "d_safe", 0.0)))
        if self.d_safe < 0:
            raise ValueError("'d_safe' must be non-negative.")

        # State variables for open-loop planning
        self.cached_plan = None
        self.step_idx = 0
        self.last_X_sol = None
        self.last_U_sol = None

        # Cost matrices depending on system state
        q_diag = config.get("Q_diag", [10.0] * self.sim.nx)
        if len(q_diag) != self.sim.nx:
            raise ValueError(
                f"'Q_diag' length must equal simulator state dimension nx={self.sim.nx}, got {len(q_diag)}."
            )
        self.Q = np.diag(q_diag)
        self.R = np.diag(self._resolve_r_diag(config))
        
        # Build the CasADi NLP graph
        self.opti = ca.Opti()

        self.X = self.opti.variable(self.sim.nx, self.N + 1)
        self.U = self.opti.variable(self.sim.nu, self.N)

        self.x0_param = self.opti.parameter(self.sim.nx)
        self.goal_param = self.opti.parameter(self.sim.nx)

        cost = 0

        q_blocks = [self.Q[state_slice, state_slice] for state_slice in self.robot_state_slices]
        r_blocks = [self.R[action_slice, action_slice] for action_slice in self.robot_action_slices]

        # Build dynamics constraints (kept in a loop because casadi_dynamics
        # is not guaranteed to support batched matrix inputs).
        for k in range(self.N):
            self.opti.subject_to(self.X[:, k + 1] ==
                                 self.sim.casadi_dynamics(self.X[:, k],
                                                          self.U[:, k]))

        # Vectorized actuator limits over all horizon steps.
        for sub_sim, action_slice in zip(self.sub_simulators, self.robot_action_slices):
            robot_max_action = float(getattr(sub_sim, "max_action", self.sim.max_action))
            self.opti.subject_to(ca.vec(self.U[action_slice, :]) >= -robot_max_action)
            self.opti.subject_to(ca.vec(self.U[action_slice, :]) <= robot_max_action)

        # Vectorized optional per-robot state bounds over steps 1..N.
        for sub_sim, state_slice in zip(self.sub_simulators, self.robot_state_slices):
            sub_lower = getattr(sub_sim, "state_lower_bounds", None)
            sub_upper = getattr(sub_sim, "state_upper_bounds", None)

            if sub_lower is not None:
                lower_bounds = np.asarray(sub_lower, dtype=float)
                finite_lower = np.isfinite(lower_bounds)
                if np.any(finite_lower):
                    lower_indices = (state_slice.start + np.flatnonzero(finite_lower)).tolist()
                    lower_values = ca.repmat(
                        ca.reshape(ca.DM(lower_bounds[finite_lower]), len(lower_indices), 1),
                        1,
                        self.N,
                    )
                    self.opti.subject_to(
                        ca.vec(self.X[lower_indices, 1:]) >= ca.vec(lower_values)
                    )

            if sub_upper is not None:
                upper_bounds = np.asarray(sub_upper, dtype=float)
                finite_upper = np.isfinite(upper_bounds)
                if np.any(finite_upper):
                    upper_indices = (state_slice.start + np.flatnonzero(finite_upper)).tolist()
                    upper_values = ca.repmat(
                        ca.reshape(ca.DM(upper_bounds[finite_upper]), len(upper_indices), 1),
                        1,
                        self.N,
                    )
                    self.opti.subject_to(
                        ca.vec(self.X[upper_indices, 1:]) <= ca.vec(upper_values)
                    )

        # Running costs over all N steps.
        for sub_sim, state_slice, action_slice, q_block, r_block in zip(
            self.sub_simulators,
            self.robot_state_slices,
            self.robot_action_slices,
            q_blocks,
            r_blocks,
        ):
            goal_matrix = ca.repmat(self.goal_param[state_slice], 1, self.N)
            error_matrix = self.X[state_slice, :-1] - goal_matrix
            q_block_no_theta = q_block
            angular_state_indices = tuple(getattr(sub_sim, "angular_state_indices", ()))
            if len(angular_state_indices) > 0:
                q_block_no_theta = np.array(q_block, copy=True)
                for angular_idx in angular_state_indices:
                    if angular_idx < 0 or angular_idx >= q_block.shape[0]:
                        raise ValueError(
                            f"Simulator {type(sub_sim).__name__} declares invalid angular_state_indices entry {angular_idx} "
                            f"for local state dimension {q_block.shape[0]}."
                        )
                    q_theta = float(q_block[angular_idx, angular_idx])
                    q_block_no_theta[angular_idx, angular_idx] = 0.0
                    theta_error = error_matrix[angular_idx, :]
                    cost += q_theta * ca.sum2(1.0 - ca.cos(theta_error))

            cost += ca.sum2(ca.sum1(ca.mtimes(q_block_no_theta, error_matrix) * error_matrix))
            cost += ca.sum2(
                ca.sum1(ca.mtimes(r_block, self.U[action_slice, :]) * self.U[action_slice, :])
            )

        pairwise_collision_is_active = self.d_safe > 0.0 and len(self.robot_state_slices) > 1
        xy_index_by_robot: list[tuple[int, int]] = []
        if pairwise_collision_is_active:
            for robot_idx, state_slice in enumerate(self.robot_state_slices):
                if state_slice.stop - state_slice.start < 2:
                    raise ValueError(
                        f"Robot {robot_idx} state must include at least x/y in first two entries."
                    )
                xy_index_by_robot.append((state_slice.start, state_slice.start + 1))

        collision_pairs: list[tuple[int, int]] = []
        if pairwise_collision_is_active:
            for i in range(len(self.robot_state_slices)):
                for j in range(i + 1, len(self.robot_state_slices)):
                    collision_pairs.append((i, j))

        if pairwise_collision_is_active and len(collision_pairs) > 0:
            # One slack variable per pair and horizon step (including terminal step).
            # This avoids coupling violations across unrelated pairs.
            collision_slack = self.opti.variable(len(collision_pairs), self.N + 1)
            self.opti.subject_to(collision_slack >= 0)

            for pair_idx, (i, j) in enumerate(collision_pairs):
                xi_idx, yi_idx = xy_index_by_robot[i]
                xj_idx, yj_idx = xy_index_by_robot[j]

                xi = self.X[xi_idx, :-1]
                yi = self.X[yi_idx, :-1]
                xj = self.X[xj_idx, :-1]
                yj = self.X[yj_idx, :-1]
                self.opti.subject_to(
                    (xi - xj) ** 2 + (yi - yj) ** 2 + collision_slack[pair_idx, :-1] >= self.d_safe ** 2
                )

                xi_terminal = self.X[xi_idx, self.N]
                yi_terminal = self.X[yi_idx, self.N]
                xj_terminal = self.X[xj_idx, self.N]
                yj_terminal = self.X[yj_idx, self.N]
                self.opti.subject_to(
                    (xi_terminal - xj_terminal) ** 2
                    + (yi_terminal - yj_terminal) ** 2
                    + collision_slack[pair_idx, self.N]
                    >= self.d_safe ** 2
                )

            cost += self.collision_slack_penalty_weight * ca.sum2(ca.sum1(collision_slack))

        for sub_sim, state_slice, q_block in zip(self.sub_simulators, self.robot_state_slices, q_blocks):
            terminal_error = self.X[state_slice, self.N] - self.goal_param[state_slice]
            q_block_no_theta = q_block
            angular_state_indices = tuple(getattr(sub_sim, "angular_state_indices", ()))
            if len(angular_state_indices) > 0:
                q_block_no_theta = np.array(q_block, copy=True)
                for angular_idx in angular_state_indices:
                    if angular_idx < 0 or angular_idx >= q_block.shape[0]:
                        raise ValueError(
                            f"Simulator {type(sub_sim).__name__} declares invalid angular_state_indices entry {angular_idx} "
                            f"for local state dimension {q_block.shape[0]}."
                        )
                    q_theta = float(q_block[angular_idx, angular_idx])
                    q_block_no_theta[angular_idx, angular_idx] = 0.0
                    cost += self.terminal_cost_multiplier * q_theta * (1.0 - ca.cos(terminal_error[angular_idx]))

            cost += ca.mtimes(
                terminal_error.T,
                ca.mtimes(q_block_no_theta * self.terminal_cost_multiplier, terminal_error),
            )

        self.opti.minimize(cost)
        self.opti.subject_to(self.X[:, 0] == self.x0_param)

        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        self.opti.solver("ipopt", opts)

    def reset(self) -> None:
        """Signals the start of a new episode."""
        self.last_X_sol = None
        self.last_U_sol = None
        self.opti.set_initial(self.X, 0.0)
        self.opti.set_initial(self.U, 0.0)
        if self.mode == "open_loop":
            self.cached_plan = None
            self.step_idx = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "mpc":
            x0 = self.sim.invert_obs(obs, validate=False)
            self.opti.set_value(self.x0_param, x0)
            self.opti.set_value(self.goal_param, self.sim.goal_state)

            if self.last_X_sol is not None and self.last_U_sol is not None:
                shifted_X = np.hstack([self.last_X_sol[:, 1:], self.last_X_sol[:, -1:]])
                shifted_U = np.hstack([self.last_U_sol[:, 1:], self.last_U_sol[:, -1:]])
                self.opti.set_initial(self.X, shifted_X)
                self.opti.set_initial(self.U, shifted_U)

            try:
                sol = self.opti.solve()
                self.last_X_sol = sol.value(self.X)
                self.last_U_sol = sol.value(self.U)
                return self.last_U_sol[:, 0]
            except RuntimeError as exc:
                raise PlannerSolveError(
                    "CasADi planner solve failed in MPC mode. "
                    f"Current state estimate: {x0.tolist()}, goal: {self.sim.goal_state.tolist()}."
                ) from exc

        elif self.mode == "open_loop":
            # Plan once on the first step
            if self.cached_plan is None:
                x0 = self.sim.invert_obs(obs, validate=False)
                self.opti.set_value(self.x0_param, x0)
                self.opti.set_value(self.goal_param, self.sim.goal_state)

                try:
                    sol = self.opti.solve()
                    self.cached_plan = sol.value(self.U)
                except RuntimeError as exc:
                    raise PlannerSolveError(
                        "CasADi planner solve failed in open-loop mode. "
                        f"Initial state estimate: {x0.tolist()}, goal: {self.sim.goal_state.tolist()}."
                    ) from exc

            # Iterate through the cached plan
            if self.step_idx < self.cached_plan.shape[1]:
                action = self.cached_plan[:, self.step_idx]
                self.step_idx += 1
                return action
            else:
                return np.zeros(self.sim.nu)  # Ran out of planned steps

        else:
            raise ValueError(f"Unknown CasadiPlanner mode: {self.mode}")
