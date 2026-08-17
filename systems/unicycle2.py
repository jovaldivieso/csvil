from typing import Any, Mapping

import casadi as ca
import numpy as np

from .dynamics import DynamicsSimulator


class Unicycle2(DynamicsSimulator):
    """
    2nd order unicycle dynamics:

    x = [x (position), y (position), theta (orientation), v (velocity), omega (angular velocity)] 
    u = [a_v (acceleration), a_omega (angular acceleration)]
    """

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)

        # fixed goal position:
        self.goal = np.asarray(config.get("goal", [0.0, 0.0, 0.0]))
        self.randomize_goal = config.get("randomize_goal", "goal" not in config)
        self.goal_position_bounds = tuple(
            float(value) for value in config.get("goal_position_bounds", [-1.0, 1.0])
        )

        self.randomize_initial_velocity = config.get("randomize_initial_velocity", False)

        # physical limits from unicycle2_v0:
        self.max_accel = float(config.get("max_accel", 0.25))
        self.max_action = self.max_accel
        
        self.max_speed = float(config.get("max_speed", 0.5))
        self.max_omega = float(config.get("max_omega", 0.5))

        self.pos_tol = float(config.get("pos_tol", 0.05))
        self.theta_tol = float(config.get("theta_tol", 0.05))
        self.vel_tol = float(config.get("vel_tol", 0.05))
        self.omega_tol = float(config.get("omega_tol", 0.05))
        self.initial_position_min_goal_distance = float(
            config.get("initial_position_min_goal_distance", self.pos_tol)
        )
        self.initial_position_radius_bounds = tuple(
            float(value)
            for value in config.get(
                "initial_position_radius_bounds",
                [self.initial_position_min_goal_distance, 1.0],
            )
        )
        
        # number of states and actions:
        self.nx = 5 
        self.nu = 2 
        self.obs_dim = 6

        # constrains velocity and orientation:
        self.state_lower_bounds = np.array(
            [-np.inf, -np.inf, -np.inf, -self.max_speed, -self.max_omega],
            dtype=float,
        )
        self.state_upper_bounds = np.array(
            [np.inf, np.inf, np.inf, self.max_speed, self.max_omega],
            dtype=float,
        )

    @property
    def is_euclidean(self) -> bool:
        return False

    @property
    def angular_state_indices(self) -> tuple[int, ...]:
        return (2,)

    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        """
        applies one simulation step
        """
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        action_array = self.validate_action(action) if validate else np.asarray(action, dtype=float)
        clipped_action = np.clip(action_array, -self.max_action, self.max_action)

        x, y, theta, v, omega = state_array
        a_v, a_omega = clipped_action

        x = x + v * np.cos(theta) * self.dt
        y = y + v * np.sin(theta) * self.dt
        theta = np.arctan2(np.sin(theta + omega * self.dt), np.cos(theta + omega * self.dt))

        v = np.clip(v + a_v * self.dt, -self.max_speed, self.max_speed)
        omega = np.clip(omega + a_omega * self.dt, -self.max_omega, self.max_omega)

        return np.array([x, y, theta, v, omega], dtype=float)

    def global_vector_to_ego(self, vec: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        rotates a global 2D vector by -theta into the robot body frame
        """
        state_array = self.validate_state(state)
        theta = state_array[2]
        ego_x = np.cos(theta) * vec[0] + np.sin(theta) * vec[1]
        ego_y = -np.sin(theta) * vec[0] + np.cos(theta) * vec[1]
        return np.array([ego_x, ego_y], dtype=float)

    def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray:
        """
        turns absoulte simulator state into observation,
        [goal_ego_x, goal_ego_y, sin(rel_theta), cos(rel_theta), velocity, angular velocity]
        """
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        rel_pos = self.goal[0:2] - state_array[0:2]
        ego_pos = self.global_vector_to_ego(rel_pos, state_array)
        rel_theta = np.arctan2(np.sin(self.goal[2] - state_array[2]), np.cos(self.goal[2] - state_array[2]))
        obs = np.array(
            [
                ego_pos[0],
                ego_pos[1],
                np.sin(rel_theta),
                np.cos(rel_theta),
                state_array[3],
                state_array[4],
            ],
            dtype=float,
        )
        return self.validate_observation(obs) if validate else obs

    def invert_obs(self, obs: np.ndarray, validate: bool = True) -> np.ndarray:
        """
        reconstructs absolute state from observation
        """
        obs_array = self.validate_observation(obs) if validate else np.asarray(obs, dtype=float)
        rel_theta = np.arctan2(obs_array[2], obs_array[3])
        theta = self.goal[2] - rel_theta

        ego_dx, ego_dy = obs_array[0], obs_array[1]
        global_dx = np.cos(theta) * ego_dx - np.sin(theta) * ego_dy
        global_dy = np.sin(theta) * ego_dx + np.cos(theta) * ego_dy

        return np.array(
            [
                self.goal[0] - global_dx,
                self.goal[1] - global_dy,
                theta,
                obs_array[4],
                obs_array[5],
            ],
            dtype=float,
        )

    @property
    def goal_state(self) -> np.ndarray:
        """
        defines full final state [x_goal, y_goal, theta_goal, 0.0, 0.0]
        """
        return np.array([self.goal[0], self.goal[1], self.goal[2], 0.0, 0.0])

    def is_done(self, state: np.ndarray, validate: bool = True) -> bool:
        """
        checks whether robot has successfully completed task
        """
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        pos_error = np.linalg.norm(state_array[0:2] - self.goal[0:2])
        theta_error = abs(np.arctan2(np.sin(self.goal[2] - state_array[2]), np.cos(self.goal[2] - state_array[2])))

        return (
            pos_error < self.pos_tol
            and theta_error < self.theta_tol
            and abs(state_array[3]) < self.vel_tol # speed error
            and abs(state_array[4]) < self.omega_tol # omega error
        )

    def casadi_dynamics(self, x: Any, u: Any):
        """
        symbolic second-order unicycle dynamics for CasADi
        """

        # state x = [x (position), y (position), theta (orientation), v (velocity), omega (angular velocity)] 
        next_x = x[0] + x[3] * ca.cos(x[2]) * self.dt
        next_y = x[1] + x[3] * ca.sin(x[2]) * self.dt
        next_theta = x[2] + x[4] * self.dt
        next_v = x[3] + u[0] * self.dt
        next_omega = x[4] + u[1] * self.dt
        # no clipping, planner enforces limits with explicit optimization constraints
        return ca.vertcat(
            next_x,
            next_y,
            next_theta,
            next_v,
            next_omega,
        )

    def get_dataset_features(self) -> dict[str, Any]:
        """
        creates LeRobot feature schema
        """
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
                "names": ["a_v", "a_omega"],
            },
        }

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """
        samples random state
        """
        offset = self.sample_planar_start_offset(
            rng,
            radius_bounds=self.initial_position_radius_bounds,
            min_goal_distance=self.initial_position_min_goal_distance,
        )
        pos = self.goal[:2] + offset
        theta = rng.uniform(low=-np.pi, high=np.pi)

        v = 0.0
        omega = 0.0
        if self.randomize_initial_velocity:
            v = rng.uniform(-self.max_speed, self.max_speed)
            omega = rng.uniform(-self.max_omega, self.max_omega)
        
        return np.array([pos[0], pos[1], theta, v, omega])

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
        """
        converts observation-action pair into format expected by LeRobot
        """
        obs_array = self.validate_observation(obs)
        action_array = self.validate_action(action)

        return {
            "observation.environment_state": np.asarray(obs_array[:4], dtype=np.float32),
            "observation.state": np.asarray(obs_array[4:6], dtype=np.float32),
            "action": np.asarray(action_array, dtype=np.float32),
        }