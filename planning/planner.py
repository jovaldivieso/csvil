from abc import ABC, abstractmethod

import numpy as np


class Planner(ABC):
    """
    Abstract base class for all motion planners.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Clear internal state (e.g., cached trajectories) for a new episode.
        Stateless planners (like pure MPC) can implement this as a pass/no-op.
        """
        pass

    @abstractmethod
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute and return the next action based on the current observation.
        """
        pass
