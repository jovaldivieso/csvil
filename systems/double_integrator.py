from .dynamics import DynamicsSimulator
import casadi as ca
import numpy as np
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
        self.goal_position_bounds = tuple(
            float(value) for value in config.get("goal_position_bounds", [-1.0, 1.0])
        )
        self.max_action = config.get("max_accel", 2.0)
        self.nx = 4
        self.nu = 2
        self.error_tolerance = float(config.get("error_tolerance", 0.05))
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

    def predict_next_state(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        action_array = self.validate_action(action) if validate else np.asarray(action, dtype=float)
        clipped_action = np.clip(action_array, -self.max_action, self.max_action)

        position = state_array[:2]
        velocity = state_array[2:4]
        next_pos = position + velocity * self.dt + 0.5 * clipped_action * (self.dt**2)
        next_vel = velocity + clipped_action * self.dt
        return np.concatenate([next_pos, next_vel])

    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        return self.predict_next_state(state, action, validate=validate)

    def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        obs = np.concatenate([self.goal - state_array[:2], state_array[2:4]])
        return self.validate_observation(obs) if validate else obs

    def is_done(self, state: np.ndarray, validate: bool = True) -> bool:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        # Must reach goal and stop moving
        dist = np.linalg.norm(state_array[:2] - self.goal)
        speed = np.linalg.norm(state_array[2:4])
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
            "observation.neighbor_state": {"dtype": "float32", "shape": (0,), "names": []},
            "observation.neighbor_mask": {"dtype": "float32", "shape": (0,), "names": []},
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["ax", "ay"],
            },
        }

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        offset = self.sample_planar_start_offset(
            rng,
            radius_bounds=self.initial_position_radius_bounds,
            min_goal_distance=self.initial_position_min_goal_distance,
        )
        start_pos = self.goal + offset
        return np.array([start_pos[0], start_pos[1], 0.0, 0.0])

    def invert_obs(self, obs: np.ndarray, validate: bool = True) -> np.ndarray:
        obs_array = self.validate_observation(obs) if validate else np.asarray(obs, dtype=float)
        absolute_pos = self.goal - obs_array[:2]
        return np.concatenate([absolute_pos, obs_array[2:4]])

    @property
    def goal_state(self) -> np.ndarray:
        return np.array([self.goal[0], self.goal[1], 0.0, 0.0])

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        if self.randomize_goal:
            self.goal = rng.uniform(
                low=self.goal_position_bounds[0],
                high=self.goal_position_bounds[1],
                size=self.goal.shape[0],
            )

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> list[dict[str, np.ndarray]]:
        """Package the observation and action into a dictionary for LeRobot"""
        obs = self.validate_observation(obs)
        action = self.validate_action(action)
        return [{
            "observation.environment_state": np.asarray(obs[:2], dtype=np.float32),
            "observation.state": np.asarray(obs[2:4], dtype=np.float32),
            "observation.neighbor_state": np.empty(0, dtype=np.float32),
            "observation.neighbor_mask": np.empty(0, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
        }]
