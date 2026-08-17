from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from core.types import VectorSpec, as_vector


DEFAULT_DONE_HOLD_STEPS = 5
DEFAULT_INITIAL_STATE_SEED = 0


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
        Internal simulator logic should keep orientation and position handling
        consistent with each system definition. Flattened coordinates (for example
        ``theta`` or concatenated pose vectors) are boundary representations used
        at interfaces the codebase does not fully control, such as CasADi,
        LeRobot, YAML/CLI I/O, or other external tool APIs.

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
    is_euclidean: bool
    angular_state_indices: tuple[int, ...]
    num_robots: int
    simulators: list["DynamicsProtocol"]
    robot_state_slices: list[slice]
    robot_action_slices: list[slice]

    def validate_state(self, state: np.ndarray) -> np.ndarray: ...

    def validate_action(self, action: np.ndarray) -> np.ndarray: ...

    def validate_observation(self, observation: np.ndarray) -> np.ndarray: ...

    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray: ...

    def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray: ...

    def global_vector_to_ego(self, vec: np.ndarray, state: np.ndarray) -> np.ndarray: ...

    def is_done(self, state: np.ndarray, validate: bool = True) -> bool: ...

    def casadi_dynamics(self, x: Any, u: Any) -> Any: ...

    def get_dataset_features(self) -> dict[str, Any]: ...

    def reset_random(self) -> np.ndarray: ...

    def reset_rollout_termination(self) -> None: ...

    def set_early_rollout_termination(self, enabled: bool) -> None: ...

    def should_terminate_rollout(self, state: np.ndarray) -> bool: ...

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, Any]: ...

    def decentralized_policy_observation(self, obs: np.ndarray, robot_id: int = 0) -> dict[str, np.ndarray]: ...

    def get_decentralized_dataset_features(self) -> dict[str, Any]: ...

    def format_decentralized_dataset_frames(self, obs: np.ndarray, action: np.ndarray) -> list[dict[str, Any]]: ...

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray: ...

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None: ...

    def invert_obs(self, obs: np.ndarray, validate: bool = True) -> np.ndarray: ...

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
        initial_state_seed = config.get("initial_state_seed", DEFAULT_INITIAL_STATE_SEED)
        if initial_state_seed is None:
            initial_state_seed = DEFAULT_INITIAL_STATE_SEED
        self._sampling_rng = np.random.default_rng(int(initial_state_seed))
        self._rollout_done_counter = 0
        self._early_rollout_termination_enabled = True

        # Agnostic dimensions so planners can read them automatically
        self.nx = None
        self.nu = None
        self.obs_dim = getattr(self, "obs_dim", self.nx)
        self.max_action = None

    def sample_planar_start_offset(
        self,
        rng: np.random.Generator,
        radius_bounds: tuple[float, float],
        min_goal_distance: float,
    ) -> np.ndarray:
        radius_min, radius_max = radius_bounds
        effective_radius_min = max(radius_min, min_goal_distance)
        if radius_max <= effective_radius_min:
            raise ValueError(
                "Initial-state sampling requires the maximum initial radius to exceed the minimum goal distance."
            )

        # Sample uniformly with respect to planar area on the annulus, not uniformly in radius.
        radius_sq = rng.uniform(effective_radius_min**2, radius_max**2)
        radius = float(np.sqrt(radius_sq))
        angle = rng.uniform(0.0, 2.0 * np.pi)
        return np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=float)

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        del rng

    def reset_rollout_termination(self) -> None:
        self._rollout_done_counter = 0

    def set_early_rollout_termination(self, enabled: bool) -> None:
        enabled_flag = bool(enabled)
        if enabled_flag != self._early_rollout_termination_enabled:
            self._rollout_done_counter = 0
        self._early_rollout_termination_enabled = enabled_flag

    def should_terminate_rollout(self, state: np.ndarray) -> bool:
        if not self._early_rollout_termination_enabled:
            return False

        hold_steps = int(self.config.get("done_hold_steps", DEFAULT_DONE_HOLD_STEPS))
        if hold_steps <= 0:
            raise ValueError("'done_hold_steps' must be a positive integer.")

        if self.is_done(state, validate=False):
            self._rollout_done_counter += 1
        else:
            self._rollout_done_counter = 0

        return self._rollout_done_counter >= hold_steps

    @property
    def is_euclidean(self) -> bool:
        """Whether squared Euclidean residuals are valid across all state coordinates."""
        return True

    @property
    def angular_state_indices(self) -> tuple[int, ...]:
        """Local state coordinates whose planner residuals should use wrapped angular distance."""
        return ()

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
    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        """Get next state"""
        pass

    @abstractmethod
    def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray:
        """Get observation"""
        pass

    def global_vector_to_ego(self, vec: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Project a global vector into the robot's ego frame; identity unless the system has orientation"""
        return np.asarray(vec, dtype=float)

    @abstractmethod
    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        """Must return symbolic CasADi representation of the dynamics"""
        pass

    @abstractmethod
    def get_dataset_features(self) -> dict[str, Any]:
        """Return the LeRobot features dictionary for this specific robot"""
        pass

    def reset_random(self) -> np.ndarray:
        """Return a random, dynamically valid initial state"""
        self.randomize_goal_for_reset(self._sampling_rng)
        return self.reset(self.random_initial_state(self._sampling_rng))

    @abstractmethod
    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, Any]:
        """Package the observation and action into a dictionary for LeRobot"""
        pass

    def decentralized_policy_observation(self, obs: np.ndarray, robot_id: int = 0) -> dict[str, np.ndarray]:
        if robot_id != 0:
            raise IndexError("Fleet-of-1 simulators only expose robot_id=0.")

        frame = self.format_dataset_frame(obs, np.zeros(int(self.nu), dtype=np.float32))
        environment_state = np.asarray(frame["observation.environment_state"], dtype=np.float32).reshape(-1)
        state = np.asarray(frame["observation.state"], dtype=np.float32).reshape(-1)
        return {
            "ego_obs": np.concatenate([environment_state, state]),
            "neighbor_obs": np.empty((0, 2), dtype=np.float32),
            "neighbor_mask": np.empty((0, 1), dtype=np.float32),
        }

    def get_decentralized_dataset_features(self) -> dict[str, Any]:
        features = self.get_dataset_features()
        environment_feature = features["observation.environment_state"]
        state_feature = features["observation.state"]
        action_feature = features["action"]
        return {
            "observation.environment_state": dict(environment_feature),
            "observation.state": dict(state_feature),
            "observation.neighbor_state": {
                "dtype": "float32",
                "shape": (0,),
                "names": [],
            },
            "observation.neighbor_mask": {
                "dtype": "float32",
                "shape": (0,),
                "names": [],
            },
            "action": dict(action_feature),
        }

    def format_decentralized_dataset_frames(self, obs: np.ndarray, action: np.ndarray) -> list[dict[str, Any]]:
        frame = self.format_dataset_frame(obs, action)
        return [
            {
                "observation.environment_state": np.asarray(
                    frame["observation.environment_state"], dtype=np.float32
                ),
                "observation.state": np.asarray(frame["observation.state"], dtype=np.float32),
                "observation.neighbor_state": np.empty(0, dtype=np.float32),
                "observation.neighbor_mask": np.empty(0, dtype=np.float32),
                "action": np.asarray(frame["action"], dtype=np.float32),
            }
        ]

    @abstractmethod
    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Return a random initial state without changing the goal"""
        pass

    @abstractmethod
    def invert_obs(self, obs: np.ndarray, validate: bool = True) -> np.ndarray:
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
        self.reset_rollout_termination()
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
            obs = self.observe(state, validate=False)
            action = policy_fn(obs)  # Call your motion planner here
            state = self.step(state, action, validate=False)
            states.append(state.copy())
            observations.append(obs)
            actions.append(action)

            if self.should_terminate_rollout(state):
                break

        return {
            "states": np.array(states),
            "observations": np.array(observations),
            "actions": np.array(actions),
        }
