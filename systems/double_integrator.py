from .dynamics import DynamicsSimulator
import casadi as ca
import numpy as np
import torch


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
        self.nx = 4
        self.nu = 2

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
        # Must reach goal and stop moving
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

    def get_dataset_features(self):
        """Return the LeRobot features dictionary for the double integrator"""
        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["goal_rel_x", "goal_rel_y"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["vx", "vy"]
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["ax", "ay"]
            },
        }

    def random_initial_state(self, rng):
        pos = rng.uniform(low=-5.0, high=5.0, size=2)
        return np.array([pos[0], pos[1], 0.0, 0.0])

    def invert_obs(self, obs):
        return np.array([self.goal[0] - obs[0], self.goal[1] - obs[1], obs[2], obs[3]])

    @property
    def goal_state(self):
        return np.array([self.goal[0], self.goal[1], 0.0, 0.0])

    def reset_random(self):
        """Randomize both the goal and the start position"""
        # Randomize the Goal anywhere in a predefined workspace
        self.goal = np.random.uniform(low=-5.0, high=5.0, size=2)

        # Uniform polar Sampling for the start position, relative to the goal
        radius = np.random.uniform(0.5, 3.0)
        angle = np.random.uniform(0, 2 * np.pi)
        offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        start_pos = self.goal + offset

        # Initialize at rest
        initial_state = np.array([start_pos[0], start_pos[1], 0.0, 0.0])
        return self.reset(initial_state)

    def format_dataset_frame(self, obs, action):
        """Package the observation and action into a dictionary for LeRobot"""
        return {
            "observation.environment_state":
            torch.from_numpy(obs[0:2]).float(),
            "observation.state": torch.from_numpy(obs[2:4]).float(),
            "action": torch.from_numpy(action).float(),
        }
