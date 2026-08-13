from .dynamics import DynamicsSimulator
from .state_space_types import (
    Euclidean2DAction,
    SE2PoseAndEuclidean2DObservation,
    SE2PoseState,
    SO2State,
)
from core.types import VectorSpec, as_vector
import casadi as ca
import numpy as np
import torch
from typing import Any, Mapping


class Unicycle1(DynamicsSimulator):
    """
    1st order unicycle dynamics:
        u = [v, omega]
        s = [x, y, theta]
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.goal = np.array(config.get("goal", [0.0, 0.0, 0.0]))
        self.randomize_goal = config.get("randomize_goal",
                                         "goal" not in config)
        self.goal_position_bounds = tuple(
            float(value) for value in config.get("goal_position_bounds", [-1.0, 1.0])
        )
        self.max_action = config.get("max_v", 2.0)
        self.nx = 3
        self.nu = 2
        self.obs_dim = 6
        self.error_tolerance = float(config.get("error_tolerance", 0.05))
        self.current_action = np.zeros(self.nu, dtype=float)
        
        environment = config.get("environment", {})
        self.environment_min = np.asarray(environment.get("min", [-5.0, -5.0]), dtype=float)
        self.environment_max = np.asarray(environment.get("max", [5.0, 5.0]), dtype=float)
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
        self.db_lacam_robot_type = "unicycle1_v0"

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        return as_vector(observation, VectorSpec(name="observation", size=self.obs_dim))

    @property
    def is_euclidean(self) -> bool:
        return False

    @property
    def angular_state_indices(self) -> tuple[int, ...]:
        return (2,)

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        state = super().reset(initial_state)
        self.current_action = np.zeros(self.nu, dtype=float)
        return state

    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        action_array = self.validate_action(action) if validate else np.asarray(action, dtype=float)
        clipped_action = np.clip(action_array, -self.max_action, self.max_action)
        self.current_action = clipped_action.copy()

        x, y, theta = state_array
        v, omega = clipped_action
        next_x = x + v * np.cos(theta) * self.dt
        next_y = y + v * np.sin(theta) * self.dt
        next_theta = np.arctan2(np.sin(theta + omega * self.dt), np.cos(theta + omega * self.dt))
        return np.array([next_x, next_y, next_theta], dtype=float)

    def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        rel_pos = self.goal[0:2] - state_array[:2]
        rel_theta = np.arctan2(np.sin(self.goal[2] - state_array[2]), np.cos(self.goal[2] - state_array[2]))
        obs = np.array([
            rel_pos[0],
            rel_pos[1],
            np.sin(rel_theta),
            np.cos(rel_theta),
            self.current_action[0],
            self.current_action[1],
        ], dtype=float)
        return self.validate_observation(obs) if validate else obs

    def is_done(self, state: np.ndarray, validate: bool = True) -> bool:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        pos_error = np.linalg.norm(state_array[:2] - self.goal[0:2])
        theta_error = abs(np.arctan2(np.sin(self.goal[2] - state_array[2]), np.cos(self.goal[2] - state_array[2])))
        return (pos_error < self.error_tolerance and
                theta_error < self.error_tolerance)

    def casadi_dynamics(self, x: Any, u: Any):
        """Symbolic unicycle 1 dynamics for CasADi"""
        pos = x[:2]
        theta = x[2]
        next_pos = pos + ca.vertcat(u[0] * ca.cos(theta) * self.dt,
                                    u[0] * ca.sin(theta) * self.dt)
        next_theta = theta + u[1] * self.dt
        return ca.vertcat(next_pos[0], next_pos[1], next_theta)

    def get_dataset_features(self) -> dict[str, Any]:
        """Return the LeRobot features dictionary for the unicylce 1"""
        exteroception_names = [
            "goal_rel_x",
            "goal_rel_y",
            "sin_rel_theta",
            "cos_rel_theta",
        ]

        proprioception_names = [
            "v",
            "omega",
        ]

        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (4,),
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
                "names": ["v", "omega"],
            },
        }

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        while True:
            offset = self.sample_planar_start_offset(
                rng,
                radius_bounds=self.initial_position_radius_bounds,
                min_goal_distance=self.initial_position_min_goal_distance,
            )
            pos = self.goal[:2] + offset

            if np.all((pos >= self.environment_min) & (pos <= self.environment_max)):
                theta = rng.uniform(-np.pi, np.pi)
                return np.array([pos[0], pos[1], theta])

    def invert_obs(self, obs: np.ndarray, validate: bool = True) -> np.ndarray:
        obs_array = self.validate_observation(obs) if validate else np.asarray(obs, dtype=float)
        rel_theta = np.arctan2(obs_array[2], obs_array[3])
        return np.array([
            self.goal[0] - obs_array[0],
            self.goal[1] - obs_array[1],
            self.goal[2] - rel_theta,
        ], dtype=float)

    @property
    def goal_state(self) -> np.ndarray:
        return np.array([self.goal[0], self.goal[1], self.goal[2]])

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        if self.randomize_goal:
            goal_pos = rng.uniform(
                low=self.goal_position_bounds[0],
                high=self.goal_position_bounds[1],
                size=self.goal.shape[0] - 1,
            )
            goal_theta = rng.uniform(low=-np.pi, high=np.pi)
            self.goal = np.array([goal_pos[0], goal_pos[1], goal_theta])

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
        """Package the observation and action into a dictionary for LeRobot"""
        obs = self.validate_observation(obs)
        action = self.validate_action(action)
        return {
            "observation.environment_state": np.asarray(obs[:4], dtype=np.float32),
            "observation.state": np.asarray(obs[4:6], dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
        }