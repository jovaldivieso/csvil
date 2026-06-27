from .dynamics import DynamicsSimulator
import numpy as np


class SingleIntegrator(DynamicsSimulator):
    """Single integrator dynamics: v_dot = a"""

    def __init__(self, config):
        super().__init__(config)
        self.goal = config.get("goal", np.array([0, 0]))
        self.max_speed = config.get("max_speed", 1.0)

    def step(self, state, action):
        """state = [x, y, vx, vy], action = [ax, ay]"""
        pos = state[:2]

        # Single integrator: x_dot = v
        # Input action is the velocity (no acceleration integration)
        action = np.clip(action, -self.max_speed, self.max_speed)
        next_pos = pos + action * self.dt

        return np.concatenate([next_pos, action])

    def observe(self, state):
        """Return: [goal_relative_x, goal_relative_y, vx, vy]"""
        pos = state[:2]
        vel = state[2:4]
        goal_rel = self.goal - pos

        return np.concatenate([goal_rel, vel])

    def is_done(self, state):
        """Check if state is goal"""
        pos = state[:2]
        return np.linalg.norm(pos - self.goal) < 0.05
