from __future__ import annotations

from typing import Any, Iterable, Mapping

import casadi as ca
import numpy as np

from core.types import VectorSpec, as_vector
from systems.collision_checker import check_homogeneous_fleet_collisions
from systems.dynamics import DynamicsProtocol, DynamicsSimulator


SAFE_INITIAL_STATE_MAX_ATTEMPTS = 256


class MultiRobotSimulator(DynamicsSimulator):
    """Composite simulator that aggregates multiple robot simulators.

    The simulator exposes a flat global state/action/observation interface while
    delegating per-robot dynamics and observation logic to each sub-simulator.

        Contract invariants
        -------------------
        - ``num_robots == len(simulators)``.
        - ``robot_state_slices`` and ``robot_action_slices`` partition global state
            and action vectors into contiguous, non-overlapping per-robot segments.
        - Global state/action ordering is the concatenation of robot-local vectors
            in simulator list order.

        Observation packing invariants
        ------------------------------
        For each robot, augmented observation ordering is:

        ``[base_environment_features, relative_xy_to_other_robots..., visibility_mask..., base_state_features]``

        The global observation is the concatenation of these per-robot augmented
        observations in simulator list order.

        Dataset schema invariants
        -------------------------
        ``get_dataset_features()`` and ``format_dataset_frame()`` expose unified
        per-robot frame dictionaries with these keys:

        - ``observation.environment_state``
        - ``observation.neighbor_state``
        - ``observation.neighbor_mask``
        - ``observation.state``
        - ``action``

        ``format_dataset_frame()`` returns one frame dictionary per robot, using
        local observation and action dimensions from the homogeneous fleet.

        Visibility-gated relative observation invariant
        ----------------------------------------------
        Relative x/y terms to other robots are included only when the other robot
        lies within the observing robot's configured visibility radius. When out
        of range, the corresponding relative term is set to zero and the matching
        mask slot is set to zero so collisions at distance 0 remain distinct from
        invisible neighbors.
    """

    def __init__(self, simulators: Iterable[DynamicsProtocol], config: Mapping[str, Any] | None = None):
        self.simulators = list(simulators)
        if len(self.simulators) == 0:
            raise ValueError("MultiRobotSimulator requires at least one sub-simulator.")
        self.num_robots = len(self.simulators)

        merged_config: dict[str, Any] = dict(config or {})
        merged_config.setdefault("dt", float(self.simulators[0].dt))
        super().__init__(merged_config)

        dt0 = float(self.simulators[0].dt)
        for idx, sim in enumerate(self.simulators[1:], start=1):
            if not np.isclose(float(sim.dt), dt0):
                raise ValueError(
                    "All sub-simulators in MultiRobotSimulator must share the same dt. "
                    f"Robot 0 has dt={dt0}, robot {idx} has dt={sim.dt}."
                )

        self.dt = dt0

        self.robot_state_slices: list[slice] = []
        self.robot_action_slices: list[slice] = []
        self.robot_observation_slices: list[slice] = []
        self.robot_base_observation_dims: list[int] = []
        self.robot_env_dims: list[int] = []
        self.robot_relative_dims: list[int] = []
        self.robot_neighbor_mask_dims: list[int] = []
        self.robot_proprio_dims: list[int] = []

        state_start = 0
        action_start = 0
        obs_start = 0
        for sim in self.simulators:
            state_end = state_start + int(sim.nx)
            action_end = action_start + int(sim.nu)
            env_dim, proprio_dim = self._observation_feature_dims(sim)
            relative_dim = 2 * (len(self.simulators) - 1)
            mask_dim = len(self.simulators) - 1
            base_obs_dim = env_dim + relative_dim + mask_dim + proprio_dim
            obs_end = obs_start + base_obs_dim

            self.robot_state_slices.append(slice(state_start, state_end))
            self.robot_action_slices.append(slice(action_start, action_end))
            self.robot_observation_slices.append(slice(obs_start, obs_end))
            self.robot_base_observation_dims.append(base_obs_dim)
            self.robot_env_dims.append(env_dim)
            self.robot_relative_dims.append(relative_dim)
            self.robot_neighbor_mask_dims.append(mask_dim)
            self.robot_proprio_dims.append(proprio_dim)

            state_start = state_end
            action_start = action_end
            obs_start = obs_end

        self.nx = state_start
        self.nu = action_start
        self.obs_dim = obs_start

        visibility_spec = merged_config.get("inter_robot_visibility_radius", np.inf)
        if isinstance(visibility_spec, (int, float)):
            self.robot_visibility_radii = np.full(
                self.num_robots,
                float(visibility_spec),
                dtype=float,
            )
        elif isinstance(visibility_spec, list):
            if len(visibility_spec) != self.num_robots:
                raise ValueError(
                    "When list-valued, 'inter_robot_visibility_radius' length must match number of robots "
                    f"({self.num_robots}), got {len(visibility_spec)}."
                )
            self.robot_visibility_radii = np.asarray(visibility_spec, dtype=float)
        else:
            raise ValueError(
                "'inter_robot_visibility_radius' must be either a number or a per-robot list of numbers."
            )

        if np.any(self.robot_visibility_radii < 0):
            raise ValueError("Visibility radii must be non-negative.")

        # Used by some planners as a symmetric control bound fallback.
        self.max_action = float(max(float(sim.max_action) for sim in self.simulators))
        self.d_safe = float(self.config.get("d_safe", 0.0))
        self._is_homogeneous = all(
            type(sim) is type(self.simulators[0])
            and int(sim.nx) == int(self.simulators[0].nx)
            and int(sim.nu) == int(self.simulators[0].nu)
            for sim in self.simulators[1:]
        )
        if not self._is_homogeneous:
            raise ValueError(
                "MultiRobotSimulator only supports strictly homogeneous teams (same simulator type and dimensions)."
            )
        self._casadi_mapped_dynamics: ca.Function | None = None

    @property
    def is_euclidean(self) -> bool:
        return all(bool(sub_sim.is_euclidean) for sub_sim in self.simulators)

    @property
    def angular_state_indices(self) -> tuple[int, ...]:
        global_indices: list[int] = []
        for sub_sim, state_slice in zip(self.simulators, self.robot_state_slices):
            local_indices = tuple(getattr(sub_sim, "angular_state_indices", ()))
            local_state_dim = int(sub_sim.nx)

            for local_idx_raw in local_indices:
                local_idx = int(local_idx_raw)
                if local_idx < 0 or local_idx >= local_state_dim:
                    raise ValueError(
                        f"Sub-simulator {type(sub_sim).__name__} declares invalid angular_state_indices "
                        f"entry {local_idx} for local state dimension {local_state_dim}."
                    )
                global_indices.append(int(state_slice.start) + local_idx)

        return tuple(global_indices)

    @property
    def position_indices(self) -> tuple[int, ...]:
        """Aggregate global position indices from all sub-simulators."""
        global_indices: list[int] = []
        for sub_sim, state_slice in zip(self.simulators, self.robot_state_slices):
            local_indices = tuple(getattr(sub_sim, "position_indices", (0, 1)))
            local_state_dim = int(sub_sim.nx)

            for local_idx_raw in local_indices:
                local_idx = int(local_idx_raw)
                if local_idx < 0 or local_idx >= local_state_dim:
                    raise ValueError(
                        f"Sub-simulator {type(sub_sim).__name__} declares invalid position_indices "
                        f"entry {local_idx} for local state dimension {local_state_dim}."
                    )
                global_indices.append(int(state_slice.start) + local_idx)

        return tuple(global_indices)

    @staticmethod
    def _observation_feature_dims(simulator: DynamicsProtocol) -> tuple[int, int]:
        env_dim = 0
        proprio_dim = 0
        for feature_name, feature_info in simulator.get_dataset_features().items():
            if not feature_name.startswith("observation."):
                continue

            shape = feature_info.get("shape")
            if not shape:
                raise ValueError(f"Observation feature '{feature_name}' must define a non-empty shape.")

            dim = int(shape[0])
            if feature_name == "observation.environment_state":
                env_dim += dim
            elif feature_name == "observation.state":
                proprio_dim += dim
            else:
                # Keep compatibility if additional observation fields are added in future.
                env_dim += dim

        if env_dim + proprio_dim == 0:
            raise ValueError("Each robot simulator must expose observation features.")

        return env_dim, proprio_dim

    def _split_state(self, state: np.ndarray, validate: bool = True) -> list[np.ndarray]:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        return [state_array[s].copy() for s in self.robot_state_slices]

    def _split_action(self, action: np.ndarray, validate: bool = True) -> list[np.ndarray]:
        action_array = self.validate_action(action) if validate else np.asarray(action, dtype=float)
        return [action_array[s].copy() for s in self.robot_action_slices]

    def _split_observation(self, observation: np.ndarray, validate: bool = True) -> list[np.ndarray]:
        observation_array = self.validate_observation(observation) if validate else np.asarray(observation, dtype=float)
        return [observation_array[s].copy() for s in self.robot_observation_slices]

    def _base_observation_from_augmented(self, robot_id: int, robot_obs: np.ndarray) -> np.ndarray:
        """Drop relative-to-other-robot terms and reconstruct base per-robot observation.

        Base ordering must match sub-simulator observe(): [environment_state, state].
        """
        base_env_dim = self.robot_env_dims[robot_id]
        rel_dim = self.robot_relative_dims[robot_id]
        mask_dim = self.robot_neighbor_mask_dims[robot_id]
        proprio_dim = self.robot_proprio_dims[robot_id]

        # Augmented layout per robot: [base_env, rel_terms, mask_terms, proprio]
        base_env = robot_obs[:base_env_dim]
        proprio_start = base_env_dim + rel_dim + mask_dim
        proprio_end = proprio_start + proprio_dim
        proprio = robot_obs[proprio_start:proprio_end]
        return np.concatenate([base_env, proprio])

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        return as_vector(observation, VectorSpec(name="observation", size=self.obs_dim))

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        state = self.validate_state(initial_state)
        split_state = self._split_state(state, validate=False)
        for sim, robot_state in zip(self.simulators, split_state):
            sim.reset(robot_state)
        self.state = state.copy()
        self.time = 0
        self.reset_rollout_termination()
        return self.state

    def is_collision(self, state: np.ndarray) -> bool:
        robot_states = self._split_state(state)
        return check_homogeneous_fleet_collisions(
            robot_states,
            self.simulators[0].position_indices,
            self.d_safe,
        )

    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        split_state = self._split_state(state, validate=validate)
        split_action = self._split_action(action, validate=validate)
        next_parts = [
            sim.step(robot_state, robot_action, validate=False)
            for sim, robot_state, robot_action in zip(self.simulators, split_state, split_action)
        ]
        next_state = np.concatenate(next_parts)
        self.state = next_state.copy()
        self.time += 1
        return next_state

    def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray:
        split_state = self._split_state(state, validate=validate)
        positions = []
        for robot_idx, robot_state in enumerate(split_state):
            if robot_state.shape[0] < 2:
                raise ValueError(
                    f"Robot {robot_idx} state must include x/y in first two entries for relative observations."
                )
            positions.append(robot_state[:2])

        observations: list[np.ndarray] = []
        for robot_idx, (sim, robot_state) in enumerate(zip(self.simulators, split_state)):
            base_obs = sim.observe(robot_state, validate=False)
            env_dim = self.robot_env_dims[robot_idx]
            base_env = base_obs[:env_dim]
            base_proprio = base_obs[env_dim:]

            rel_parts: list[np.ndarray] = []
            mask_parts: list[float] = []
            for other_idx, other_pos in enumerate(positions):
                if other_idx == robot_idx:
                    continue
                delta = other_pos - positions[robot_idx]
                visibility_radius = float(self.robot_visibility_radii[robot_idx])
                if np.isfinite(visibility_radius) and np.linalg.norm(delta) > visibility_radius:
                    rel_parts.append(np.zeros(2, dtype=float))
                    mask_parts.append(0.0)
                else:
                    rel_parts.append(sim.global_vector_to_ego(delta, robot_state))
                    mask_parts.append(1.0)

            if rel_parts:
                rel_obs = np.concatenate(rel_parts)
                rel_mask = np.asarray(mask_parts, dtype=float)
                observations.append(np.concatenate([base_env, rel_obs, rel_mask, base_proprio]))
            else:
                observations.append(np.concatenate([base_env, base_proprio]))

        observation = np.concatenate(observations)
        return self.validate_observation(observation) if validate else observation

    def is_done(self, state: np.ndarray, validate: bool = True) -> bool:
        split_state = self._split_state(state, validate=validate)
        return all(sim.is_done(robot_state, validate=False) for sim, robot_state in zip(self.simulators, split_state))

    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        reference_sim = self.simulators[0]
        sim_nx = int(reference_sim.nx)
        sim_nu = int(reference_sim.nu)
        if self._casadi_mapped_dynamics is None:
            x_sym = ca.SX.sym("x_single", sim_nx)
            u_sym = ca.SX.sym("u_single", sim_nu)
            x_next_sym = reference_sim.casadi_dynamics(x_sym, u_sym)
            step_fn = ca.Function(
                "multi_robot_step",
                [x_sym, u_sym],
                [x_next_sym],
                ["x", "u"],
                ["x_next"],
            )
            self._casadi_mapped_dynamics = step_fn.map(self.num_robots)

        mapped_state = ca.reshape(x, sim_nx, self.num_robots)
        mapped_action = ca.reshape(u, sim_nu, self.num_robots)
        mapped_next = self._casadi_mapped_dynamics(mapped_state, mapped_action)
        return ca.vec(mapped_next)

    def get_dataset_features(self) -> dict[str, Any]:
        env_dim, proprio_dim, action_dim = self._validate_homogeneous_decentralized_dimensions()
        neighbor_count = len(self.simulators) - 1
        base_features = self.simulators[0].get_dataset_features()
        env_feature = base_features["observation.environment_state"]
        state_feature = base_features["observation.state"]
        action_feature = base_features["action"]
        return {
            "observation.environment_state": dict(env_feature),
            "observation.state": dict(state_feature),
            "observation.neighbor_state": {
                "dtype": "float32",
                "shape": (2 * neighbor_count,),
                "names": [
                    name
                    for neighbor_idx in range(neighbor_count)
                    for name in (f"neighbor_{neighbor_idx}_x", f"neighbor_{neighbor_idx}_y")
                ],
            },
            "observation.neighbor_mask": {
                "dtype": "float32",
                "shape": (neighbor_count,),
                "names": [f"neighbor_{neighbor_idx}_visible" for neighbor_idx in range(neighbor_count)],
            },
            "action": dict(action_feature),
        }

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        return self._sample_safe_initial_state(rng=rng, randomize_goals=False)

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        for _ in range(SAFE_INITIAL_STATE_MAX_ATTEMPTS):
            for sim in self.simulators:
                sim.randomize_goal_for_reset(rng)

            goals = [sim.goal_state for sim in self.simulators]
            if self._positions_respect_d_safe(goals):
                return

        raise RuntimeError(
            "Failed to sample safe multi-robot goals. "
            f"Tried {SAFE_INITIAL_STATE_MAX_ATTEMPTS} attempts with d_safe={self.d_safe}."
        )

    def reset_random(self) -> np.ndarray:
        return self.reset(
            self._sample_safe_initial_state(rng=self._sampling_rng, randomize_goals=True)
        )

    def _positions_respect_d_safe(self, states: list[np.ndarray]) -> bool:
        return not check_homogeneous_fleet_collisions(
            states,
            self.simulators[0].position_indices,
            self.d_safe,
        )

    def _sample_safe_initial_state(
        self,
        rng: np.random.Generator,
        randomize_goals: bool,
    ) -> np.ndarray:
        for _ in range(SAFE_INITIAL_STATE_MAX_ATTEMPTS):
            states: list[np.ndarray] = []
            if randomize_goals:
                self.randomize_goal_for_reset(rng)
            for sim in self.simulators:
                states.append(sim.random_initial_state(rng))

            if self._positions_respect_d_safe(states):
                return np.concatenate(states)

        raise RuntimeError(
            "Unable to sample a multi-robot initial state that satisfies d_safe. "
            f"Tried {SAFE_INITIAL_STATE_MAX_ATTEMPTS} attempts with d_safe={self.d_safe}."
        )

    def invert_obs(self, obs: np.ndarray, validate: bool = True) -> np.ndarray:
        split_obs = self._split_observation(obs, validate=validate)
        states = []
        for robot_id, (sim, robot_obs) in enumerate(zip(self.simulators, split_obs)):
            base_obs = self._base_observation_from_augmented(robot_id, robot_obs)
            states.append(sim.invert_obs(base_obs, validate=False))
        state = np.concatenate(states)
        return self.validate_state(state) if validate else state

    @property
    def goal_state(self) -> np.ndarray:
        goals = [sim.goal_state for sim in self.simulators]
        return self.validate_state(np.concatenate(goals))

    def _validate_homogeneous_decentralized_dimensions(self) -> tuple[int, int, int]:
        env_dim = int(self.robot_env_dims[0])
        proprio_dim = int(self.robot_proprio_dims[0])
        action_dim = int(self.simulators[0].nu)
        reference_simulator = self.simulators[0]

        def schema_signature(simulator: DynamicsProtocol) -> tuple[tuple[str, tuple[int, ...], tuple[str, ...]], ...]:
            signature: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = []
            for feature_name, feature_info in simulator.get_dataset_features().items():
                shape = tuple(int(value) for value in feature_info.get("shape", ()))
                names = tuple(str(name) for name in feature_info.get("names", ()))
                signature.append((feature_name, shape, names))
            return tuple(signature)

        reference_signature = schema_signature(reference_simulator)

        for robot_idx, sim in enumerate(self.simulators[1:], start=1):
            if type(sim) is not type(reference_simulator):
                raise ValueError(
                    "Decentralized multi-robot policies require identical simulator types; "
                    f"robot 0 uses {type(reference_simulator).__name__}, "
                    f"robot {robot_idx} uses {type(sim).__name__}."
                )
            if schema_signature(sim) != reference_signature:
                raise ValueError(
                    "Decentralized multi-robot policies require identical dataset feature semantics "
                    "(names and shapes) for every robot."
                )
            if int(self.robot_env_dims[robot_idx]) != env_dim:
                raise ValueError(
                    "Decentralized multi-robot policies require homogeneous base environment dimensions."
                )
            if int(self.robot_proprio_dims[robot_idx]) != proprio_dim:
                raise ValueError(
                    "Decentralized multi-robot policies require homogeneous proprioceptive dimensions."
                )
            if int(sim.nu) != action_dim:
                raise ValueError(
                    "Decentralized multi-robot policies require homogeneous action dimensions."
                )

        return env_dim, proprio_dim, action_dim

    def decentralized_policy_observation(self, obs: np.ndarray, robot_id: int = 0) -> dict[str, np.ndarray]:
        split_obs = self._split_observation(obs)
        if robot_id < 0 or robot_id >= len(split_obs):
            raise IndexError(f"robot_id {robot_id} is out of bounds for {len(split_obs)} robots.")

        robot_obs = split_obs[robot_id]
        base_env_dim = self.robot_env_dims[robot_id]
        rel_dim = self.robot_relative_dims[robot_id]
        mask_dim = self.robot_neighbor_mask_dims[robot_id]
        proprio_dim = self.robot_proprio_dims[robot_id]

        env_end = base_env_dim + rel_dim
        mask_start = env_end
        mask_end = mask_start + mask_dim
        proprio_start = mask_end
        proprio_end = proprio_start + proprio_dim

        ego_obs = np.concatenate(
            [
                np.asarray(robot_obs[:base_env_dim], dtype=np.float32),
                np.asarray(robot_obs[proprio_start:proprio_end], dtype=np.float32),
            ]
        )
        neighbor_state = np.asarray(robot_obs[base_env_dim:env_end], dtype=np.float32)
        neighbor_mask = np.asarray(robot_obs[mask_start:mask_end], dtype=np.float32)

        return {
            "observation.environment_state": np.asarray(ego_obs[:base_env_dim], dtype=np.float32),
            "observation.state": np.asarray(ego_obs[base_env_dim:], dtype=np.float32),
            "observation.neighbor_state": neighbor_state,
            "observation.neighbor_mask": neighbor_mask,
        }

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> list[dict[str, Any]]:
        self._validate_homogeneous_decentralized_dimensions()
        split_obs = self._split_observation(obs)
        split_action = self._split_action(action)
        return [
            {
                **self.decentralized_policy_observation(obs, robot_id),
                "action": np.asarray(robot_action, dtype=np.float32),
            }
            for robot_id, robot_action in enumerate(split_action)
        ]
