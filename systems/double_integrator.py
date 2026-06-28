from .dynamics import DynamicsSimulator
import casadi as ca
import numpy as np


class DoubleIntegrator(DynamicsSimulator):
    """
    Double integrator dynamics:
        u = [ax, ay]
        s = [x, y, vx, vy]
    """

    def __init__(self, config):
        super().__init__(config)
        self.goal = config.get("goal", np.array([0.0, 0.0]))
        self.max_action = config.get("max_accel", 2.0)
        self.nx = 4  # [x, y, vx, vy]
        self.nu = 2  # [ax, ay]

    def step(self, state, action):
        pos = state[:2]
        vel = state[2:4]
        action = np.clip(action, -self.max_action, self.max_action)

        next_pos = pos + vel * self.dt + 0.5 * action * (self.dt**2)
        next_vel = vel + action * self.dt
        return np.concatenate([next_pos, next_vel])

    def observe(self, state):
        pos = state[:2]
        vel = state[2:4]
        return np.concatenate([self.goal - pos, vel])

    def is_done(self, state):
        # Must reach goal AND stop moving
        dist = np.linalg.norm(state[:2] - self.goal)
        speed = np.linalg.norm(state[2:4])
        return dist < 0.05 and speed < 0.05

    def casadi_dynamics(self, x, u):
        """Symbolic double integrator for CasADi"""
        pos = x[:2]
        vel = x[2:4]
        next_pos = pos + vel * self.dt + 0.5 * u * (self.dt**2)
        next_vel = vel + u * self.dt
        return ca.vertcat(next_pos[0], next_pos[1], next_vel[0], next_vel[1])
