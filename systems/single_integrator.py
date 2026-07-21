from .dynamics import DynamicsSimulator
import casadi as ca
import numpy as np
import torch
from typing import Any, Mapping


class SingleIntegrator(DynamicsSimulator):
    """
    Single integrator dynamics:
        u = [vx, vy]
        s = [x, y]
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.goal = np.array(config.get("goal", [0.0, 0.0]))
        # Determine if we should randomize the goal based on config
        self.randomize_goal = config.get("randomize_goal",
                                         "goal" not in config)
        self.max_action = config.get("max_vel", 1.0)
        self.nx = 2
        self.nu = 2
        self.error_tolerance = float(config.get("error_tolerance", 0.05))

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state = self.validate_state(state)
        action = self.validate_action(action)
        action = np.clip(action, -self.max_action, self.max_action)
        next_pos = state + action * self.dt
        return next_pos

    def observe(self, state: np.ndarray) -> np.ndarray:
        state = self.validate_state(state)
        obs = self.goal - state
        return self.validate_observation(obs)

    def is_done(self, state: np.ndarray) -> bool:
        state = self.validate_state(state)
        dist = np.linalg.norm(state - self.goal)
        return dist < self.error_tolerance

    def casadi_dynamics(self, x: Any, u: Any):
        """Symbolic single integrator for CasADi"""
        next_pos = x + u * self.dt
        return ca.vertcat(next_pos[0], next_pos[1])

    def get_dataset_features(self) -> dict[str, Any]:
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

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(low=-5.0, high=5.0, size=2)

    def invert_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = self.validate_observation(obs)
        return self.goal - obs

    @property
    def goal_state(self) -> np.ndarray:
        return np.array([self.goal[0], self.goal[1]])

    def reset_random(self) -> np.ndarray:
        """Randomize start position, and optionally the goal."""
        if self.randomize_goal:
            self.goal = np.random.uniform(low=-5.0, high=5.0, size=2)

        radius = np.random.uniform(0.5, 3.0)
        angle = np.random.uniform(0, 2 * np.pi)
        offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        start_pos = self.goal + offset
        return self.reset(start_pos)

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, torch.Tensor]:
        """Package the observation and action into a dictionary for LeRobot"""
        obs = self.validate_observation(obs)
        action = self.validate_action(action)
        return {
            # Pass relative position to both to satisfy LeRobot's architecture
            "observation.environment_state": torch.from_numpy(obs).float(),
            "observation.state": torch.from_numpy(obs).float(),
            "action": torch.from_numpy(action).float(),
        }
