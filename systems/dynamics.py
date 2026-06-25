from abc import ABC, abstractmethod
import numpy as np


class DynamicsSimulator(ABC):
    """Base class for dynamics simulation"""

    def __init__(self, config):
        self.config = config
        self.state = None
        self.time = 0
        self.dt = config.get("dt", 0.01)

    @abstractmethod
    def step(self, state, action):
        """state + action -> next_state"""
        pass

    @abstractmethod
    def observe(self, state):
        """state -> observation (what the policy sees)"""
        pass

    def reset(self, initial_state):
        self.state = initial_state.copy()
        self.time = 0
        return self.state

    def rollout(self, initial_state, policy_fn, num_steps):
        """Generate a trajectory"""
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
