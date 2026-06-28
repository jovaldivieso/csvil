import casadi as ca
import numpy as np


class SingleRobotCasadiPlanner:
    """
    Dynamics Agnostic Optimal Control Planner.
    Uses the simulator's properties (nx, nu, max_action, and casadi_dynamics)
    to formulate the NLP for any single-robot system implemented in systems/.
    """

    def __init__(self, simulator, config):
        self.sim = simulator
        self.N = config.get("horizon", 20)
        self.goal = np.array(config.get("goal", [1.0, 1.0]))

        # Cost Matrices
        self.Q = np.diag(config.get("Q_diag", [10.0, 10.0, 1.0, 1.0]))
        self.R = np.eye(self.sim.nu) * config.get("R_weight", 0.1)

    def __call__(self, obs):
        # State Estimation
        pos_x = self.goal[0] - obs[0]
        pos_y = self.goal[1] - obs[1]
        x0 = np.array([pos_x, pos_y, obs[2], obs[3]])

        # Setup CasADi NLP based on simulator dimensions
        opti = ca.Opti()
        X = opti.variable(self.sim.nx, self.N + 1)
        U = opti.variable(self.sim.nu, self.N)

        goal_vec = ca.vertcat(self.goal[0], self.goal[1], 0.0, 0.0)
        cost = 0

        # Build Constraints
        for k in range(self.N):
            # Dynamics agnostic planner: let the simulator define the physics!
            opti.subject_to(X[:, k + 1] == self.sim.casadi_dynamics(X[:, k],
                                                                    U[:, k]))

            for i in range(self.sim.nu):
                opti.subject_to(U[i, k] >= -self.sim.max_action)
                opti.subject_to(U[i, k] <= self.sim.max_action)

            error = X[:, k] - goal_vec
            cost += ca.mtimes(error.T, ca.mtimes(self.Q, error))
            cost += ca.mtimes(U[:, k].T, ca.mtimes(self.R, U[:, k]))

        terminal_error = X[:, self.N] - goal_vec
        cost += ca.mtimes(terminal_error.T, ca.mtimes(self.Q * 10,
                                                      terminal_error))

        opti.minimize(cost)
        opti.subject_to(X[:, 0] == x0)

        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
            return sol.value(U[:, 0])
        except RuntimeError as e:
            print(f"CasADi Solver Failed: {e}")
            return np.zeros(self.sim.nu)
