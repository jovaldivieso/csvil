from .dynamics import DynamicsSimulator
from .utils import wrap_angle, get_relative_position

import casadi as ca
import numpy as np
import torch


class Unicycle2(DynamicsSimulator):
    """
    2nd order unicycle dynamics:

    x = [x (position), y (position), theta (orientation), v (velocity), omega (angular velocity)] 
    u = [a_v (acceleration), a_omega (angular acceleration)]
    """

    def __init__(self, config):
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

    def step(self, state, action):
        """
        applies one simulation step
        """
        # limits action to valid range: 
        action = np.clip(np.asarray(action, dtype=float), -self.max_action, self.max_action)

        x, y, theta, v, omega = state
        a_v, a_omega = action

        x = x + v * np.cos(theta) * self.dt
        y = y + v * np.sin(theta) * self.dt
        theta = wrap_angle(theta + omega * self.dt)

        v = np.clip(v + a_v * self.dt, -self.max_speed, self.max_speed)
        omega = np.clip(omega + a_omega * self.dt, -self.max_omega, self.max_omega)

        return np.array([x, y, theta, v, omega])

    def observe(self, state):
        """
        turns absoulte simulator state into observation,
        [goal_rel_x, goal_rel_y, rel_theta, velocity, angular velocity]
        """
        goal_rel_x, goal_rel_y, rel_theta = get_relative_position(pos=state[:3], goal_pos=self.goal[:3])
        return np.array([goal_rel_x, goal_rel_y, rel_theta, state[3], state[4]])

    def invert_obs(self, obs):
        """
        reconstructs absolute state from observation
        """
        return np.array(
            [
                self.goal[0] - obs[0],
                self.goal[1] - obs[1],
                wrap_angle(self.goal[2] - obs[2]),
                obs[3],
                obs[4],
            ]
        )

    @property
    def goal_state(self):
        """
        defines full final state [x_goal, y_goal, theta_goal, 0.0, 0.0]
        """
        return np.array([self.goal[0], self.goal[1], self.goal[2], 0.0, 0.0])

    def is_done(self, state):
        """
        checks whether robot has successfully completed task
        """
        pos_error = np.linalg.norm(state[:2] - self.goal[:2])
        theta_error = abs(wrap_angle(state[2] - self.goal[2]))

        return (
            pos_error < self.error_tolerance
            and theta_error < self.error_tolerance
            and abs(state[3]) < self.error_tolerance # speed error
            and abs(state[4]) < self.error_tolerance # omega error
        )

    def casadi_dynamics(self, x, u):
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

    def get_dataset_features(self):
        """
        creates LeRobot feature schema
        """
        observation_names = [
            "goal_rel_x",
            "goal_rel_y",
            "rel_theta",
            "v",
            "omega",
        ]

        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (5,),
                "names": observation_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (5,),
                "names": observation_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["a_v", "a_omega"],
            },
        }

    def random_initial_state(self, rng):
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

    def reset_random(self):
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

    def format_dataset_frame(self, obs, action):
        """
        converts observation-action pair into format expected by LeRobot
        """

        return {
            "observation.environment_state": torch.as_tensor(
                obs,
                dtype=torch.float32,
            ),
            "observation.state": torch.as_tensor(
                obs,
                dtype=torch.float32,
            ),
            "action": torch.as_tensor(
                action,
                dtype=torch.float32,
            ),
        }