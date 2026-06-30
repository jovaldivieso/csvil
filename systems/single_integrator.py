from .dynamics import DynamicsSimulator
import casadi as ca
import numpy as np
import torch


class SingleIntegrator(DynamicsSimulator):
    """
    Single integrator dynamics:
        u = [vx, vy]
        s = [x, y]
    """

    def __init__(self, config):
        super().__init__(config)
        self.goal = config.get("goal", np.array([0.0, 0.0]))
        self.max_action = config.get("max_vel", 1.0)
        self.nx = 2
        self.nu = 2

    def step(self, state, action):
        action = np.clip(action, -self.max_action, self.max_action)
        next_pos = state + action * self.dt
        return next_pos

    def observe(self, state):
        return self.goal - state

    def is_done(self, state):
        dist = np.linalg.norm(state - self.goal)
        return dist < 0.05

    def casadi_dynamics(self, x, u):
        """Symbolic single integrator for CasADi"""
        next_pos = x + u * self.dt
        return ca.vertcat(next_pos[0], next_pos[1])

    def get_dataset_features(self):
        """Return the LeRobot features dictionary for the single integrator"""
        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["goal_rel_x", "goal_rel_y"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["goal_rel_x", "goal_rel_y"]
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["vx", "vy"]
            },
        }

    def random_initial_state(self, rng):
        return rng.uniform(low=-5.0, high=5.0, size=2)

    def invert_obs(self, obs):
        return self.goal - obs

    @property
    def goal_state(self):
        return np.array([self.goal[0], self.goal[1]])

    def reset_random(self):
        """Randomize both the goal and the start position"""
        self.goal = np.random.uniform(low=-5.0, high=5.0, size=2)

        radius = np.random.uniform(0.5, 3.0)
        angle = np.random.uniform(0, 2 * np.pi)
        offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        start_pos = self.goal + offset
        return self.reset(start_pos)

    def format_dataset_frame(self, obs, action):
        """Package the observation and action into a dictionary for LeRobot"""
        return {
            # Pass relative position to both to satisfy LeRobot's architecture
            "observation.environment_state": torch.from_numpy(obs).float(),
            "observation.state": torch.from_numpy(obs).float(),
            "action": torch.from_numpy(action).float(),
        }
