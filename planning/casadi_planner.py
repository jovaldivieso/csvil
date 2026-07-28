import casadi as ca
import numpy as np
from typing import Any, Mapping

from .planner import Planner
from systems.dynamics import DynamicsProtocol


class PlannerSolveError(RuntimeError):
    """Raised when the CasADi optimization problem cannot be solved."""


class CasadiPlanner(Planner):
    """
    This planner solves a discrete-time finite-horizon optimal control problem
    with CasADi for dynamical systems.

    Modes:
        - "mpc" (Receding Horizon): Solves the NLP at every step and executes
          only the first action. Stateless between steps.
        - "open_loop" (Trajectory Optimization): Solves the NLP once at the
          start of the episode, caches the trajectory, and executes it blindly.

    Mathematical Formulation
    ------------------------
    Optimization Variables:
        X : State trajectory over the horizon.
            X = [x_0, x_1, ..., x_N]
        U : Control (action) trajectory over the horizon.
            U = [u_0, u_1, ..., u_{N-1}]

    Runtime Parameters:
        x_init : Current observed state of the robot.
        x_goal : Target goal state.
        Q      : State error penalty matrix (diagonal).
        R      : Control effort penalty matrix.
        u_max  : Maximum symmetric actuator limit.

    Objective (Cost Function to Minimize):
        J(X, U) = sum_{k=0}^{N-1} [ (x_k - x_goal)^T * Q * (x_k - x_goal) +
                                    u_k^T * R * u_k ]
                  + (x_N - x_goal)^T * (10 * Q) * (x_N - x_goal)

        * The sum represents the "Running Cost": penalizing distance to the
          goal and aggressive control efforts at each step.
        * The final term is the "Terminal Cost": heavily penalizing the robot
          if it misses the goal at the very last step of the horizon.

    Subject to Constraints:
        1. Initial Condition: x_0 == x_init
        2. System Dynamics:   x_{k+1} == f(x_k, u_k)     for k = 0, ..., N-1
        3. Actuator Limits:   -u_max <= u_k <= u_max     for k = 0, ..., N-1
    """

    def __init__(self, simulator: DynamicsProtocol, config: Mapping[str, Any]):
        self.sim = simulator
        self.N = config.get("horizon", 20)
        self.mode = config.get("mode", "mpc")  # Default to MPC
        self.terminal_cost_multiplier = float(config.get("terminal_cost_multiplier", 10.0))

        if self.N <= 0:
            raise ValueError("'horizon' must be a positive integer.")
        if self.mode not in {"mpc", "open_loop"}:
            raise ValueError("'mode' must be one of {'mpc', 'open_loop'}.")
        if self.terminal_cost_multiplier <= 0:
            raise ValueError("'terminal_cost_multiplier' must be positive.")

        # State variables for open-loop planning
        self.cached_plan = None
        self.step_idx = 0

        # Cost matrices depending on system state
        q_diag = config.get("Q_diag", [10.0] * self.sim.nx)
        if len(q_diag) != self.sim.nx:
            raise ValueError(
                f"'Q_diag' length must equal simulator state dimension nx={self.sim.nx}, got {len(q_diag)}."
            )
        self.Q = np.diag(q_diag)
        self.R = np.eye(self.sim.nu) * config.get("R_weight", 0.1)
        
        state_lower_bounds = getattr(self.sim, "state_lower_bounds", None)
        state_upper_bounds = getattr(self.sim, "state_upper_bounds", None)

        # Build the CasADi NLP graph
        self.opti = ca.Opti()

        self.X = self.opti.variable(self.sim.nx, self.N + 1)
        self.U = self.opti.variable(self.sim.nu, self.N)

        self.x0_param = self.opti.parameter(self.sim.nx)
        self.goal_param = self.opti.parameter(self.sim.nx)

        cost = 0

        # Build constraints and costs
        for k in range(self.N):
            self.opti.subject_to(self.X[:, k + 1] ==
                                 self.sim.casadi_dynamics(self.X[:, k],
                                                          self.U[:, k]))

            for i in range(self.sim.nu):
                self.opti.subject_to(self.U[i, k] >= -self.sim.max_action)
                self.opti.subject_to(self.U[i, k] <= self.sim.max_action)
            
            # adds lower and upper bounds for state variables:    
            if state_lower_bounds is not None:
                for i in range(self.sim.nx):
                    if np.isfinite(state_lower_bounds[i]):
                        self.opti.subject_to(
                            self.X[i, k + 1] >= state_lower_bounds[i]
                        )
            if state_upper_bounds is not None:
                for i in range(self.sim.nx):
                    if np.isfinite(state_upper_bounds[i]):
                        self.opti.subject_to(
                            self.X[i, k + 1] <= state_upper_bounds[i]
                        )

            error = self.X[:, k] - self.goal_param
            cost += ca.mtimes(error.T, ca.mtimes(self.Q, error))
            cost += ca.mtimes(self.U[:, k].T, ca.mtimes(self.R, self.U[:, k]))

        terminal_error = self.X[:, self.N] - self.goal_param
        cost += ca.mtimes(terminal_error.T, ca.mtimes(self.Q * self.terminal_cost_multiplier,
                                                      terminal_error))

        self.opti.minimize(cost)
        self.opti.subject_to(self.X[:, 0] == self.x0_param)

        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        self.opti.solver("ipopt", opts)

    def reset(self) -> None:
        """Signals the start of a new episode."""
        if self.mode == "open_loop":
            self.cached_plan = None
            self.step_idx = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "mpc":
            x0 = self.sim.invert_obs(obs)
            self.opti.set_value(self.x0_param, x0)
            self.opti.set_value(self.goal_param, self.sim.goal_state)

            try:
                sol = self.opti.solve()
                return sol.value(self.U[:, 0])
            except RuntimeError as exc:
                raise PlannerSolveError(
                    "CasADi planner solve failed in MPC mode. "
                    f"Current state estimate: {x0.tolist()}, goal: {self.sim.goal_state.tolist()}."
                ) from exc

        elif self.mode == "open_loop":
            # Plan once on the first step
            if self.cached_plan is None:
                x0 = self.sim.invert_obs(obs)
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
