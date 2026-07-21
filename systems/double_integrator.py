from .dynamics import DynamicsSimulator
import casadi as ca
import numpy as np
import torch
from typing import Any, Mapping


class DoubleIntegrator(DynamicsSimulator):
    """
    Double integrator dynamics:
        u = [ax, ay]
        s = [x, y, vx, vy]
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.goal = np.array(config.get("goal", [0.0, 0.0]))
        # Determine if we should randomize the goal based on config
        self.randomize_goal = config.get("randomize_goal",
                                         "goal" not in config)
        self.max_action = config.get("max_accel", 2.0)
        self.nx = 4
        self.nu = 2
        self.error_tolerance = float(config.get("error_tolerance", 0.05))

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state = self.validate_state(state)
        action = self.validate_action(action)
        pos = state[:2]
        vel = state[2:4]
        action = np.clip(action, -self.max_action, self.max_action)

        next_pos = pos + vel * self.dt + 0.5 * action * (self.dt**2)
        next_vel = vel + action * self.dt
        return np.concatenate([next_pos, next_vel])

    def observe(self, state: np.ndarray) -> np.ndarray:
        state = self.validate_state(state)
        pos = state[:2]
        vel = state[2:4]
        obs = np.concatenate([self.goal - pos, vel])
        return self.validate_observation(obs)

    def is_done(self, state: np.ndarray) -> bool:
        state = self.validate_state(state)
        # Must reach goal and stop moving
        dist = np.linalg.norm(state[:2] - self.goal)
        speed = np.linalg.norm(state[2:4])
        return dist < self.error_tolerance and speed < self.error_tolerance

    def casadi_dynamics(self, x: Any, u: Any):
        """Symbolic double integrator for CasADi"""
        pos = x[:2]
        vel = x[2:4]
        next_pos = pos + vel * self.dt + 0.5 * u * (self.dt**2)
        next_vel = vel + u * self.dt
        return ca.vertcat(next_pos[0], next_pos[1], next_vel[0], next_vel[1])

    def get_dataset_features(self) -> dict[str, Any]:
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

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        pos = rng.uniform(low=-5.0, high=5.0, size=2)
        return np.array([pos[0], pos[1], 0.0, 0.0])

    def invert_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = self.validate_observation(obs)
        return np.array([self.goal[0] - obs[0], self.goal[1] - obs[1], obs[2],
                         obs[3]])

    @property
    def goal_state(self) -> np.ndarray:
        return np.array([self.goal[0], self.goal[1], 0.0, 0.0])

    def reset_random(self) -> np.ndarray:
        """Randomize start position, and optionally the goal."""
        if self.randomize_goal:
            # Randomize the goal anywhere in a predefined workspace
            self.goal = np.random.uniform(low=-5.0, high=5.0, size=2)

        # Uniform polar sampling for the start position, relative to the goal
        radius = np.random.uniform(0.5, 3.0)
        angle = np.random.uniform(0, 2 * np.pi)
        offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        start_pos = self.goal + offset

        # Initialize at rest
        initial_state = np.array([start_pos[0], start_pos[1], 0.0, 0.0])
        return self.reset(initial_state)

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, torch.Tensor]:
        """Package the observation and action into a dictionary for LeRobot"""
        obs = self.validate_observation(obs)
        action = self.validate_action(action)
        return {
            "observation.environment_state":
            torch.from_numpy(obs[0:2]).float(),
            "observation.state": torch.from_numpy(obs[2:4]).float(),
            "action": torch.from_numpy(action).float(),
        }
