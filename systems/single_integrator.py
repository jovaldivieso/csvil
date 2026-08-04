from .dynamics import DynamicsSimulator
from .state_space_types import (
    Euclidean2DAction,
    Euclidean2DState,
    Euclidean4DObservation,
)
from core.types import VectorSpec, as_vector
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
        self.db_lacam_robot_type = "integrator1_2d_v0"

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        return as_vector(observation, VectorSpec(name="observation", size=self.obs_dim))

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        state = super().reset(initial_state)
        self.current_action = np.zeros(self.nu, dtype=float)
        return state

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state = self.validate_state(state)
        action = self.validate_action(action)
        state_view = Euclidean2DState.from_array(state)
        action_view = Euclidean2DAction.from_array(action).clipped(self.max_action)
        self.current_action = action_view.as_numpy().copy()
        next_pos = state_view.as_numpy() + action_view.as_numpy() * self.dt
        return next_pos

    def observe(self, state: np.ndarray) -> np.ndarray:
        state = self.validate_state(state)
        obs = np.concatenate([self.goal - state, self.current_action])
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
        offset = self.sample_planar_start_offset(
            rng,
            radius_bounds=self.initial_position_radius_bounds,
            min_goal_distance=self.initial_position_min_goal_distance,
        )
        return self.goal + offset

    def invert_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = self.validate_observation(obs)
        obs_view = Euclidean4DObservation.from_array(obs)
        return self.goal - obs_view.goal_relative

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

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, torch.Tensor]:
        """Package the observation and action into a dictionary for LeRobot"""
        obs = self.validate_observation(obs)
        action = self.validate_action(action)
        obs_view = Euclidean4DObservation.from_array(obs)
        action_view = Euclidean2DAction.from_array(action)
        return {
            "observation.environment_state": torch.from_numpy(obs_view.goal_relative).float(),
            "observation.state": torch.from_numpy(obs_view.velocity_like).float(),
            "action": action_view.as_torch(),
        }