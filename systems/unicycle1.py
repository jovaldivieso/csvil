from .dynamics import DynamicsSimulator
import casadi as ca
import numpy as np
import torch


class Unicycle1(DynamicsSimulator):
    """
    1st order unicycle dynamics:
        u = [v, omega]
        s = [x, y, theta]
    """

    def __init__(self, config):
        super().__init__(config)
        self.goal = np.array(config.get("goal", [0.0, 0.0, 0.0]))
        self.randomize_goal = config.get("randomize_goal",
                                         "goal" not in config)
        self.max_action = config.get("max_v", 2.0)
        self.nx = 3
        self.nu = 2
        self.error_tolerance = float(config.get("error_tolerance", 0.05))
        
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

    def observe(self, state):
        pos = state[:2]
        theta = state[2]
        rel_pos = self.goal[:2] - pos
        rel_theta = self.goal[2] - theta
        rel_theta = (rel_theta + np.pi) % (2 * np.pi) - np.pi
        return np.concatenate([rel_pos, [rel_theta]])

    def is_done(self, state):
        pos_error = np.linalg.norm(state[:2] - self.goal[:2])
        theta_error = abs((state[2] - self.goal[2] + np.pi) %
                          (2 * np.pi) - np.pi)
        return (pos_error < self.error_tolerance and
                theta_error < self.error_tolerance)

    def casadi_dynamics(self, x, u):
        """Symbolic unicycle 1 dynamics for CasADi"""
        pos = x[:2]
        theta = x[2]
        next_pos = pos + ca.vertcat(u[0] * ca.cos(theta) * self.dt,
                                    u[0] * ca.sin(theta) * self.dt)
        next_theta = theta + u[1] * self.dt
        return ca.vertcat(next_pos[0], next_pos[1], next_theta)

    def get_dataset_features(self):
        """Return the LeRobot features dictionary for the unicylce 1"""
        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["goal_rel_x", "goal_rel_y", "rel_theta"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["goal_rel_x", "goal_rel_y", "rel_theta"]
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["v", "omega"]
            },
        }

    def random_initial_state(self, rng, environment_min, environment_max):
        position = rng.uniform(
            low=environment_min,
            high=environment_max,
        )
        theta = rng.uniform(-np.pi, np.pi)
        return np.array([position[0], position[1],theta,])

    def invert_obs(self, obs):
        return np.array([self.goal[0] - obs[0], self.goal[1] - obs[1],
                         self.goal[2] - obs[2]])

    @property
    def goal_state(self):
        return np.array([self.goal[0], self.goal[1], self.goal[2]])

    def reset_random(self):
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

    def format_dataset_frame(self, obs, action):
        """Package the observation and action into a dictionary for LeRobot"""
        return {
            "observation.environment_state":
            torch.from_numpy(obs[0:3]).float(),
            "observation.state":
            torch.from_numpy(obs[0:3]).float(),
            "action": torch.from_numpy(action).float(),
        }
