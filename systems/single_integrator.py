from .dynamics import DynamicsSimulator
import casadi as ca
import numpy as np


class SingleIntegrator(DynamicsSimulator):
    """
    Single integrator dynamics:
        u = [vx, vy]
        s = [x, y, vx, vy]
    """

    def __init__(self, config):
        super().__init__(config)
        self.goal = config.get("goal", np.array([0.0, 0.0]))
        self.max_action = config.get("max_speed", 1.0)
        self.nx = 4
        self.nu = 2

    def step(self, state, action):
        pos = state[:2]
        action = np.clip(action, -self.max_action, self.max_action)
        next_pos = pos + action * self.dt
        return np.concatenate([next_pos, action])

    def observe(self, state):
        pos = state[:2]
        vel = state[2:4]
        return np.concatenate([self.goal - pos, vel])

    def is_done(self, state):
        return np.linalg.norm(state[:2] - self.goal) < 0.05

    def casadi_dynamics(self, x, u):
        """Symbolic single integrator for CasADi"""
        next_pos = x[:2] + u * self.dt
        # Next velocity is the control input for single integrator
        return ca.vertcat(next_pos[0], next_pos[1], u[0], u[1])
