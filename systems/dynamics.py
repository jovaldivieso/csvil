from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

import numpy as np

from core.types import VectorSpec, as_vector


class DynamicsSimulator(ABC):
    """Base class for dynamics simulation"""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.state = None
        self.time = 0
        self.dt = config.get("dt", 0.05)

        # Agnostic dimensions so planners can read them automatically
        self.nx = None
        self.nu = None
        self.max_action = None

    def validate_state(self, state: np.ndarray) -> np.ndarray:
        if self.nx is None:
            raise ValueError("Simulator state dimension 'nx' must be set before use.")
        return as_vector(state, VectorSpec(name="state", size=self.nx))

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        if self.nu is None:
            raise ValueError("Simulator action dimension 'nu' must be set before use.")
        return as_vector(action, VectorSpec(name="action", size=self.nu))

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        if self.nx is None:
            raise ValueError("Simulator observation dimension cannot be validated before 'nx' is set.")
        return as_vector(observation, VectorSpec(name="observation", size=self.nx))

    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Get next state"""
        pass

    @abstractmethod
    def observe(self, state: np.ndarray) -> np.ndarray:
        """Get observation"""
        pass

    @abstractmethod
    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        """Must return symbolic CasADi representation of the dynamics"""
        pass

    @abstractmethod
    def get_dataset_features(self) -> dict[str, Any]:
        """Return the LeRobot features dictionary for this specific robot"""
        pass

    @abstractmethod
    def reset_random(self) -> np.ndarray:
        """Return a random, dynamically valid initial state"""
        pass

    @abstractmethod
    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, Any]:
        """Package the observation and action into a dictionary for LeRobot"""
        pass

    @abstractmethod
    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Return a random initial state without changing the goal"""
        pass

    @abstractmethod
    def invert_obs(self, obs: np.ndarray) -> np.ndarray:
        """Reconstruct absolute state from observation (inverse of observe())"""
        pass

    @property
    @abstractmethod
    def goal_state(self) -> np.ndarray:
        """Return the full nx-dim goal vector in state space"""
        pass

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        """Reset system state and time step"""
        state = self.validate_state(initial_state)
        self.state = state.copy()
        self.time = 0
        return self.state

    def simulate(
        self,
        initial_state: np.ndarray,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        num_steps: int,
    ) -> dict[str, np.ndarray]:
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
