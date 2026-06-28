import casadi as ca
import numpy as np


class SingleRobotCasadiPlanner:
    """
    Dynamics Agnostic Optimal Control Planner. Uses the simulator's properties
    to formulate the NLP for any single-robot system implemented in systems/.
    """

    def __init__(self, simulator, config):
        self.sim = simulator
        self.N = config.get("horizon", 20)
        self.goal = np.array(config.get("goal", [1.0, 1.0]))

        # Cost matrices
        self.Q = np.diag(config.get("Q_diag", [10.0, 10.0, 1.0, 1.0]))
        self.R = np.eye(self.sim.nu) * config.get("R_weight", 0.1)

        self.opti = ca.Opti()
        self.X = self.opti.variable(self.sim.nx, self.N + 1)
        self.U = self.opti.variable(self.sim.nu, self.N)
        self.x0_param = self.opti.parameter(self.sim.nx)

        goal_vec = ca.vertcat(self.goal[0], self.goal[1], 0.0, 0.0)
        cost = 0

        for k in range(self.N):
            self.opti.subject_to(self.X[:, k + 1] ==
                                 self.sim.casadi_dynamics(self.X[:, k],
                                                          self.U[:, k]))

            for i in range(self.sim.nu):
                self.opti.subject_to(self.U[i, k] >= -self.sim.max_action)
                self.opti.subject_to(self.U[i, k] <= self.sim.max_action)

            error = self.X[:, k] - goal_vec
            cost += ca.mtimes(error.T, ca.mtimes(self.Q, error))
            cost += ca.mtimes(self.U[:, k].T, ca.mtimes(self.R, self.U[:, k]))

        terminal_error = self.X[:, self.N] - goal_vec
        cost += ca.mtimes(terminal_error.T, ca.mtimes(self.Q * 10,
                                                      terminal_error))

        self.opti.minimize(cost)
        self.opti.subject_to(self.X[:, 0] == self.x0_param)

        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        self.opti.solver("ipopt", opts)

    def __call__(self, obs):
        pos_x = self.goal[0] - obs[0]
        pos_y = self.goal[1] - obs[1]
        x0 = np.array([pos_x, pos_y, obs[2], obs[3]])

        self.opti.set_value(self.x0_param, x0)

        try:
            sol = self.opti.solve()
            return sol.value(self.U[:, 0])
        except RuntimeError:
            return np.zeros(self.sim.nu)
