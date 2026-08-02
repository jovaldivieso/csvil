from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from core.types import VectorSpec, as_vector


@runtime_checkable
class DynamicsProtocol(Protocol):
    """Structural contract for pluggable dynamics simulators.

    Any class with these attributes and methods is accepted by planners,
    collectors, and training code without requiring inheritance from
        DynamicsSimulator.

        Fleet-of-1 contract invariants
        ------------------------------
        Every simulator must expose a fleet view, even when representing a single
        physical robot:

        - ``num_robots`` is the number of fleet members.
        - ``simulators`` contains one sub-simulator per fleet member.
        - ``robot_state_slices`` and ``robot_action_slices`` partition the global
            state/action vectors into contiguous, non-overlapping per-robot blocks.
        - ``obs_dim`` is the validated observation size and may differ from ``nx``.

        Observation and dataset invariants
        ----------------------------------
        - ``observe()`` returns a vector of length ``obs_dim``.
        - ``get_dataset_features()`` defines the canonical observation/action
            schema consumed by learners.
        - ``format_dataset_frame()`` must return tensors matching that schema
            exactly (keys and dimensions).

        Geometry invariant
        ------------------
        For manifold-valued systems, internal simulator logic should use the
        corresponding Lie-group/Lie-algebra operations whenever possible.
        Flattened chart coordinates (for example ``theta`` or concatenated pose
        vectors) are boundary representations reserved for interfaces the codebase
        does not fully control, such as CasADi, LeRobot, YAML/CLI I/O, or other
        external tool APIs.

        State-ordering invariant
        ------------------------
        Global state/action vectors use concatenated per-robot ordering aligned with
        ``robot_state_slices`` and ``robot_action_slices``.
    """

    config: Mapping[str, Any]
    state: np.ndarray | None
    time: int
    dt: float
    nx: int
    nu: int
    obs_dim: int
    max_action: float
    has_heading: bool
    num_robots: int
    simulators: list["DynamicsProtocol"]
    robot_state_slices: list[slice]
    robot_action_slices: list[slice]

    def validate_state(self, state: np.ndarray) -> np.ndarray: ...

    def validate_action(self, action: np.ndarray) -> np.ndarray: ...

    def validate_observation(self, observation: np.ndarray) -> np.ndarray: ...

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray: ...

    def observe(self, state: np.ndarray) -> np.ndarray: ...

    def is_done(self, state: np.ndarray) -> bool: ...

    def casadi_dynamics(self, x: Any, u: Any) -> Any: ...

    def get_dataset_features(self) -> dict[str, Any]: ...

    def reset_random(self) -> np.ndarray: ...

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, Any]: ...

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray: ...

    def invert_obs(self, obs: np.ndarray) -> np.ndarray: ...

    @property
    def goal_state(self) -> np.ndarray: ...

    def reset(self, initial_state: np.ndarray) -> np.ndarray: ...

    def simulate(
        self,
        initial_state: np.ndarray,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        num_steps: int,
    ) -> dict[str, np.ndarray]: ...


class DynamicsSimulator(ABC):
    """Base class for dynamics simulation.

    This base class implements the Fleet-of-1 defaults expected by
    ``DynamicsProtocol``:

    - ``num_robots == 1``
    - ``simulators == [self]``
    - ``robot_state_slices == [slice(0, nx)]``
    - ``robot_action_slices == [slice(0, nu)]``

    Subclasses modeling composite fleets can override these properties by
    assigning explicit values (e.g. ``MultiRobotSimulator``).
    """

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.state = None
        self.time = 0
        self.dt = config.get("dt", 0.05)

        # Agnostic dimensions so planners can read them automatically
        self.nx = None
        self.nu = None
        self.obs_dim = getattr(self, "obs_dim", self.nx)
        self.max_action = None

    @property
    def has_heading(self) -> bool:
        """Whether the simulator state/goal semantics include orientation (theta)."""
        return False

    @property
    def num_robots(self) -> int:
        return getattr(self, "_num_robots", 1)

    @num_robots.setter
    def num_robots(self, value: int) -> None:
        if value <= 0:
            raise ValueError("'num_robots' must be a positive integer.")
        self._num_robots = int(value)

    @property
    def simulators(self) -> list["DynamicsProtocol"]:
        return getattr(self, "_simulators", [self])

    @simulators.setter
    def simulators(self, value: list["DynamicsProtocol"]) -> None:
        self._simulators = list(value)

    @property
    def robot_state_slices(self) -> list[slice]:
        if hasattr(self, "_robot_state_slices"):
            return self._robot_state_slices
        if self.nx is None:
            raise ValueError("Simulator state slices are unavailable before 'nx' is set.")
        return [slice(0, int(self.nx))]

    @robot_state_slices.setter
    def robot_state_slices(self, value: list[slice]) -> None:
        self._robot_state_slices = list(value)

    @property
    def robot_action_slices(self) -> list[slice]:
        if hasattr(self, "_robot_action_slices"):
            return self._robot_action_slices
        if self.nu is None:
            raise ValueError("Simulator action slices are unavailable before 'nu' is set.")
        return [slice(0, int(self.nu))]

    @robot_action_slices.setter
    def robot_action_slices(self, value: list[slice]) -> None:
        self._robot_action_slices = list(value)

    def validate_state(self, state: np.ndarray) -> np.ndarray:
        if self.nx is None:
            raise ValueError("Simulator state dimension 'nx' must be set before use.")
        return as_vector(state, VectorSpec(name="state", size=self.nx))

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        if self.nu is None:
            raise ValueError("Simulator action dimension 'nu' must be set before use.")
        return as_vector(action, VectorSpec(name="action", size=self.nu))

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        if self.obs_dim is None:
            if self.nx is None:
                raise ValueError(
                    "Simulator observation dimension cannot be validated before 'obs_dim' or 'nx' is set."
                )
            self.obs_dim = int(self.nx)
        return as_vector(observation, VectorSpec(name="observation", size=int(self.obs_dim)))

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
