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
        self.max_action = config.get("max_v", 2.0)
        self.nx = 3
        self.nu = 2
        self.obs_dim = 5
        self.error_tolerance = float(config.get("error_tolerance", 0.05))
        self.current_action = np.zeros(self.nu, dtype=float)

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        return as_vector(observation, VectorSpec(name="observation", size=self.obs_dim))

    @property
    def has_heading(self) -> bool:
        return True

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        state = super().reset(initial_state)
        self.current_action = np.zeros(self.nu, dtype=float)
        return state

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        state = self.validate_state(state)
        action = self.validate_action(action)
        state_view = SE2PoseState.from_array(state)
        action_view = Euclidean2DAction.from_array(action).clipped(self.max_action)
        self.current_action = action_view.as_numpy().copy()

        v = action_view.first
        omega = action_view.second
        body_frame_velocity = np.array([v, 0.0], dtype=float)
        next_pos = state_view.translation + state_view.orientation.act(body_frame_velocity) * self.dt
        next_orientation = state_view.orientation.compose(SO2State.from_angle(omega * self.dt))
        return np.concatenate([next_pos, [next_orientation.angle]])
        
        # db-lacam’s mapping from identifiers to motion-primitives is in src/run_dblacam.cpp:
        self.db_lacam_robot_type = "unicycle1_v0"

    def step(self, state, action):
        action = np.clip(action, -self.max_action, self.max_action)

        pos = state[:2]
        theta = state[2]
        v = action[0]
        omega = action[1]
        next_pos = pos + np.array([v * np.cos(theta),
                                   v * np.sin(theta)]) * self.dt
        next_theta = theta + omega * self.dt
        return np.concatenate([next_pos, [next_theta]])

    def observe(self, state: np.ndarray) -> np.ndarray:
        state = self.validate_state(state)
        state_view = SE2PoseState.from_array(state)
        rel_pos = self.goal[0:2] - state_view.translation
        rel_theta = state_view.orientation.error_to(SE2PoseState.from_array(self.goal).orientation)
        obs = np.concatenate([rel_pos, [rel_theta], self.current_action])
        return self.validate_observation(obs)

    def is_done(self, state: np.ndarray) -> bool:
        state = self.validate_state(state)
        state_view = SE2PoseState.from_array(state)
        pos_error = np.linalg.norm(state_view.translation - self.goal[0:2])
        theta_error = abs(state_view.orientation.error_to(SE2PoseState.from_array(self.goal).orientation))
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
            "rel_theta",
        ]

        proprioception_names = [
            "v",
            "omega",
        ]

        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (3,),
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

    def random_initial_state(self, rng):
        pos = rng.uniform(low=-5.0, high=5.0, size=2)
        theta = rng.uniform(low=-np.pi, high=np.pi)
        return np.array([pos[0], pos[1], theta])

    def invert_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = self.validate_observation(obs)
        obs_view = SE2PoseAndEuclidean2DObservation.from_array(obs)
        return np.array([
            self.goal[0] - obs_view.exteroception[0],
            self.goal[1] - obs_view.exteroception[1],
            self.goal[2] - obs_view.rel_theta,
        ])

    @property
    def goal_state(self) -> np.ndarray:
        return np.array([self.goal[0], self.goal[1], self.goal[2]])

    def reset_random(self) -> np.ndarray:
        """Randomize start position, and optionally the goal."""
        if self.randomize_goal:
            goal_pos = np.random.uniform(low=-5.0, high=5.0, size=2)
            goal_theta = np.random.uniform(low=-np.pi, high=np.pi)
            self.goal = np.array([goal_pos[0], goal_pos[1], goal_theta])

        radius = np.random.uniform(0.5, 3.0)
        angle = np.random.uniform(0, 2 * np.pi)
        offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        start_pos = self.goal[:2] + offset
        start_theta = np.random.uniform(low=-np.pi, high=np.pi)

        initial_state = np.array([start_pos[0], start_pos[1], start_theta])
        return self.reset(initial_state)

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, torch.Tensor]:
        """Package the observation and action into a dictionary for LeRobot"""
        obs = self.validate_observation(obs)
        action = self.validate_action(action)
        obs_view = SE2PoseAndEuclidean2DObservation.from_array(obs)
        action_view = Euclidean2DAction.from_array(action)
        return {
            "observation.environment_state":
            torch.from_numpy(obs_view.exteroception).float(),
            "observation.state":
            torch.from_numpy(obs_view.euclidean_2d).float(),
            "action": action_view.as_torch(),
        }
