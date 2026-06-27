import casadi as ca
import numpy as np


class SingleRobotCasadiPlanner:
    """
    Dynamics-Agnostic Optimal Control Planner using CasADi.

    This planner separates the optimization logic from the physics. It accepts
    a symbolic CasADi dynamics function, making it reusable for arbitrary robot
    dynamics.
    """

    def __init__(self, config, casadi_dynamics_fn=None, nx=2, nu=2):
        self.dt = config.get("dt", 0.05)
        self.N = config.get("horizon", 20)
        self.goal = np.array(config.get("goal", [1.0, 1.0]))

        # System Dimensions (Agnostic setup)
        self.nx = nx
        self.nu = nu

        # Control Bounds (assumes symmetric limits for simplicity here)
        self.u_min = -config.get("max_speed", 1.0)
        self.u_max = config.get("max_speed", 1.0)

        # Cost Matrices
        self.Q = np.eye(self.nx) * 10.0
        self.R = np.eye(self.nu) * 0.1

        # Inject the dynamics function (or fallback to Single Integrator if
        # none provided)
        if casadi_dynamics_fn is not None:
            self.discrete_dynamics = casadi_dynamics_fn
        else:
            self.discrete_dynamics = self._default_single_integrator_dynamics

    def _default_single_integrator_dynamics(self, x, u):
        """
        CasADi-compatible symbolic representation of the single integrator.
        This replaces np.clip and standard numpy arrays with symbolic graph
        operations.
        """
        # x_{k+1} = x_k + u_k * dt
        return x + u * self.dt

    def __call__(self, obs):
        # 1. State Estimation: Reconstruct global position from relative
        # observation
        # (For more complex pipelines, this logic would sit in a dedicated
        # StateEstimator)
        pos_x = self.goal[0] - obs[0]
        pos_y = self.goal[1] - obs[1]
        x0 = np.array([pos_x, pos_y])

        # 2. Setup CasADi NLP
        opti = ca.Opti()

        # Agnostic variable creation based on provided dimensions nx and nu
        X = opti.variable(self.nx, self.N + 1)
        U = opti.variable(self.nu, self.N)

        goal_vec = ca.vertcat(self.goal[0], self.goal[1])
        cost = 0

        # 3. Build Constraints and Cost Function
        for k in range(self.N):
            # AGNOSTIC DYNAMICS INJECTION:
            # We call the generic function, regardless of what the underlying
            # physics are!
            opti.subject_to(X[:, k + 1] == self.discrete_dynamics(X[:, k],
                                                                  U[:, k]))

            # Apply agnostic control bounds dynamically
            for i in range(self.nu):
                opti.subject_to(U[i, k] >= self.u_min)
                opti.subject_to(U[i, k] <= self.u_max)

            # Running cost
            error = X[:, k] - goal_vec
            cost += ca.mtimes(error.T, ca.mtimes(self.Q, error))
            cost += ca.mtimes(U[:, k].T, ca.mtimes(self.R, U[:, k]))

        # Terminal cost
        terminal_error = X[:, self.N] - goal_vec
        cost += ca.mtimes(terminal_error.T, ca.mtimes(self.Q * 10,
                                                      terminal_error))

        opti.minimize(cost)

        # Initial condition constraint
        opti.subject_to(X[:, 0] == x0)

        # 4. Solver configuration
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
            # Return the optimal first control command
            optimal_u = sol.value(U[:, 0])
            return optimal_u
        except RuntimeError as e:
            print(f"CasADi Solver Failed: {e}")
            return np.zeros(self.nu)
