from typing import Any, Mapping

import casadi as ca
import numpy as np

from core.types import VectorSpec, as_vector
from .dynamics import DynamicsSimulator


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
        self.goal_position_bounds = tuple(
            float(value) for value in config.get("goal_position_bounds", [-1.0, 1.0])
        )
        self.max_action = config.get("max_vel", 1.0)
        self.nx = 2
        self.nu = 2
        self.obs_dim = 4
        self.error_tolerance = float(config.get("error_tolerance", 0.05))
        self.current_action = np.zeros(self.nu, dtype=float)
        self.initial_position_min_goal_distance = float(
            config.get("initial_position_min_goal_distance", self.error_tolerance)
        )
        self.initial_position_radius_bounds = tuple(
            float(value)
            for value in config.get(
                "initial_position_radius_bounds",
                [self.initial_position_min_goal_distance, 1.0],
            )
        )
        environment = config.get("environment", {})
        self.environment_min = np.asarray(environment.get("min", [-5.0, -5.0]), dtype=float)
        self.environment_max = np.asarray(environment.get("max", [5.0, 5.0]),
            dtype=float,
        )

        self.db_lacam_robot_type = "integrator1_2d_v0"

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        return as_vector(observation, VectorSpec(name="observation", size=self.obs_dim))

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        state = super().reset(initial_state)
        self.current_action = np.zeros(self.nu, dtype=float)
        return state

    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        action_array = self.validate_action(action) if validate else np.asarray(action, dtype=float)
        clipped_action = np.clip(action_array, -self.max_action, self.max_action)
        self.current_action = clipped_action.copy()
        return state_array + clipped_action * self.dt

    def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        obs = np.concatenate([self.goal - state_array, self.current_action])
        return self.validate_observation(obs) if validate else obs

    def is_done(self, state: np.ndarray, validate: bool = True) -> bool:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        dist = np.linalg.norm(state_array - self.goal)
        return dist < self.error_tolerance

    def casadi_dynamics(self, x: Any, u: Any):
        """Symbolic single integrator for CasADi"""
        next_pos = x + u * self.dt
        return ca.vertcat(next_pos[0], next_pos[1])

    def get_dataset_features(self) -> dict[str, Any]:
        """Return the LeRobot features dictionary for the single integrator"""
        exteroception_names = [
            "goal_rel_x",
            "goal_rel_y",
        ]

        proprioception_names = [
            "vx",
            "vy",
        ]

        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (2,),
                "names": exteroception_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": proprioception_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["vx", "vy"],
            },
        }

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        while True:
            offset = self.sample_planar_start_offset(
                rng,
                radius_bounds=self.initial_position_radius_bounds,
                min_goal_distance=self.initial_position_min_goal_distance,
            )
            initial_state = self.goal + offset

            if np.all((initial_state >= self.environment_min) & (initial_state <= self.environment_max)):
                return initial_state

    def invert_obs(self, obs: np.ndarray, validate: bool = True) -> np.ndarray:
        obs_array = self.validate_observation(obs) if validate else np.asarray(obs, dtype=float)
        return self.goal - obs_array[:2]

    @property
    def goal_state(self) -> np.ndarray:
        return np.array([self.goal[0], self.goal[1]])

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        if self.randomize_goal:
            self.goal = rng.uniform(
                low=self.goal_position_bounds[0],
                high=self.goal_position_bounds[1],
                size=self.goal.shape[0],
            )

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
        """Package the observation and action into a dictionary for LeRobot"""
        obs = self.validate_observation(obs)
        action = self.validate_action(action)
        return {
            "observation.environment_state": np.asarray(obs[:2], dtype=np.float32),
            "observation.state": np.asarray(obs[2:4], dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
        }