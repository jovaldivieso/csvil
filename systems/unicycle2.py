from .dynamics import DynamicsSimulator
from .state_space_types import (
    Euclidean2DAction,
    SE2PoseAndEuclidean2DObservation,
    SE2PoseAndEuclidean2DState,
)
from .utils import wrap_angle
from typing import Any, Mapping

import casadi as ca
import numpy as np
import torch


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

        self.randomize_initial_velocity = config.get("randomize_initial_velocity", False)

        # physical limits from unicycle2_v0:
        self.max_accel = float(config.get("max_accel", 0.25))
        self.max_action = self.max_accel
        
        self.max_speed = float(config.get("max_speed", 0.5))
        self.max_omega = float(config.get("max_omega", 0.5))

        self.error_tolerance = float(config.get("error_tolerance", 0.05))
        
        # number of states and actions:
        self.nx = 5 
        self.nu = 2 

        # constrains velocity and orientation:
        self.state_lower_bounds = np.array(
            [-np.inf, -np.inf, -np.inf, -self.max_speed, -self.max_omega],
            dtype=float,
        )
        self.state_upper_bounds = np.array(
            [np.inf, np.inf, np.inf, self.max_speed, self.max_omega],
            dtype=float,
        )

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        applies one simulation step
        """
        state = self.validate_state(state)
        action = self.validate_action(action)
        state_view = SE2PoseAndEuclidean2DState.from_array(state)
        action_view = Euclidean2DAction.from_array(action)
        # limits action to valid range: 
        clipped_action = action_view.clipped(self.max_action)

        pose = state_view.pose
        x = pose.translation[0]
        y = pose.translation[1]
        theta = pose.theta
        v = state_view.v
        omega = state_view.omega
        a_v = clipped_action.first
        a_omega = clipped_action.second

        x = x + v * np.cos(theta) * self.dt
        y = y + v * np.sin(theta) * self.dt
        theta = wrap_angle(theta + omega * self.dt)

        v = np.clip(v + a_v * self.dt, -self.max_speed, self.max_speed)
        omega = np.clip(omega + a_omega * self.dt, -self.max_omega, self.max_omega)

        return np.array([x, y, theta, v, omega])

    def observe(self, state: np.ndarray) -> np.ndarray:
        """
        turns absoulte simulator state into observation,
        [goal_rel_x, goal_rel_y, rel_theta, velocity, angular velocity]
        """
        state = self.validate_state(state)
        state_view = SE2PoseAndEuclidean2DState.from_array(state)
        rel_pos = self.goal[0:2] - state_view.pose.translation
        rel_theta = wrap_angle(self.goal[2] - state_view.pose.theta)
        obs = np.array([rel_pos[0], rel_pos[1], rel_theta, state_view.v, state_view.omega])
        return self.validate_observation(obs)

    def invert_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        reconstructs absolute state from observation
        """
        obs = self.validate_observation(obs)
        obs_view = SE2PoseAndEuclidean2DObservation.from_array(obs)
        return np.array(
            [
                self.goal[0] - obs_view.exteroception[0],
                self.goal[1] - obs_view.exteroception[1],
                wrap_angle(self.goal[2] - obs_view.rel_theta),
                obs_view.euclidean_2d[0],
                obs_view.euclidean_2d[1],
            ]
        )

    @property
    def goal_state(self) -> np.ndarray:
        """
        defines full final state [x_goal, y_goal, theta_goal, 0.0, 0.0]
        """
        return np.array([self.goal[0], self.goal[1], self.goal[2], 0.0, 0.0])

    def is_done(self, state: np.ndarray) -> bool:
        """
        checks whether robot has successfully completed task
        """
        state = self.validate_state(state)
        state_view = SE2PoseAndEuclidean2DState.from_array(state)
        pos_error = np.linalg.norm(state_view.pose.translation - self.goal[0:2])
        theta_error = abs(wrap_angle(state_view.pose.theta - self.goal[2]))

        return (
            pos_error < self.error_tolerance
            and theta_error < self.error_tolerance
            and abs(state_view.v) < self.error_tolerance # speed error
            and abs(state_view.omega) < self.error_tolerance # omega error
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
                "names": ["a_v", "a_omega"],
            },
        }

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """
        samples random state
        """
        pos = rng.uniform(low=-5.0, high=5.0, size=2)
        theta = rng.uniform(low=-np.pi, high=np.pi)

        v = 0.0
        omega = 0.0
        if self.randomize_initial_velocity:
            v = rng.uniform(-self.max_speed, self.max_speed)
            omega = rng.uniform(-self.max_omega, self.max_omega)
        
        return np.array([pos[0], pos[1], theta, v, omega])

    def reset_random(self) -> np.ndarray:
        """
        creates a random valid initial state around goal
        """

        radius = np.random.uniform(0.5, 3.0)
        angle = np.random.uniform(0.0, 2 * np.pi)

        pos = self.goal[:2] + radius * np.array(
            [np.cos(angle), np.sin(angle)]
        )
        theta = np.random.uniform(low=-np.pi, high=np.pi)

        v = 0.0
        omega = 0.0
        if self.randomize_initial_velocity:
            v = np.random.uniform(-self.max_speed, self.max_speed)
            omega = np.random.uniform(-self.max_omega, self.max_omega)
        
        initial_state = np.array([pos[0], pos[1], theta, v, omega])
        return self.reset(initial_state)

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, torch.Tensor]:
        """
        converts observation-action pair into format expected by LeRobot
        """
        obs = self.validate_observation(obs)
        action = self.validate_action(action)
        obs_view = SE2PoseAndEuclidean2DObservation.from_array(obs)
        action_view = Euclidean2DAction.from_array(action)

        return {
            "observation.environment_state": torch.as_tensor(
                obs_view.exteroception,
                dtype=torch.float32,
            ),
            "observation.state": torch.as_tensor(
                obs_view.euclidean_2d,
                dtype=torch.float32,
            ),
            "action": torch.as_tensor(
                action_view.as_numpy(),
                dtype=torch.float32,
            ),
        }