from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PlannerProtocol(Protocol):
    """Structural contract for planners used by simulation and data pipelines."""

    def reset(self) -> None:
        """Clear any episode-specific internal state."""
        ...

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Compute and return the action for the provided observation."""
        ...


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
