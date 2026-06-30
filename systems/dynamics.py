from abc import ABC, abstractmethod
import numpy as np


class DynamicsSimulator(ABC):
    """Base class for dynamics simulation"""

    def __init__(self, config):
        self.config = config
        self.state = None
        self.time = 0
        self.dt = config.get("dt", 0.05)

        # Agnostic dimensions so planners can read them automatically
        self.nx = None
        self.nu = None
        self.max_action = None

    @abstractmethod
    def step(self, state, action):
        """Get next state"""
        pass

    @abstractmethod
    def observe(self, state):
        """Get observation"""
        pass

    @abstractmethod
    def casadi_dynamics(self, x, u):
        """Must return symbolic CasADi representation of the dynamics"""
        pass

    @abstractmethod
    def get_dataset_features(self):
        """Return the LeRobot features dictionary for this specific robot"""
        pass

    @abstractmethod
    def reset_random(self):
        """Return a random, dynamically valid initial state"""
        pass

    @abstractmethod
    def format_dataset_frame(self, obs, action):
        """Package the observation and action into a dictionary for LeRobot"""
        pass

    @abstractmethod
    def random_initial_state(self, rng):
        """Return a random initial state without changing the goal"""
        pass

    @abstractmethod
    def invert_obs(self, obs):
        """Reconstruct absolute state from observation (inverse of observe())"""
        pass

    @property
    @abstractmethod
    def goal_state(self):
        """Return the full nx-dim goal vector in state space"""
        pass

    def reset(self, initial_state):
        """Reset system state and time step"""
        self.state = initial_state.copy()
        self.time = 0
        return self.state

    def simulate(self, initial_state, policy_fn, num_steps):
        """Simulate a trajectory"""
        states, observations, actions = [], [], []
        state = self.reset(initial_state)

        for _ in range(num_steps):
            obs = self.observe(state)
            action = policy_fn(obs)  # Call your motion planner here
            state = self.step(state, action)

            states.append(state.copy())
            observations.append(obs)
            actions.append(action)

        return {
            "states": np.array(states),
            "observations": np.array(observations),
            "actions": np.array(actions),
        }
