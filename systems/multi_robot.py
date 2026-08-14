from __future__ import annotations

from typing import Any, Iterable, Mapping

import casadi as ca
import numpy as np

from core.types import VectorSpec, as_vector
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
        top-level keys:

        - ``observation.environment_state``
        - ``observation.neighbor_mask``
        - ``observation.state``
        - ``action``

        with dimensions equal to concatenated per-robot contributions.

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
            rel_dim = self.robot_relative_dims[robot_idx]
            mask_dim = self.robot_neighbor_mask_dims[robot_idx]
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
                    rel_parts.append(delta)
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
        next_parts = []
        for sim, state_slice, action_slice in zip(
            self.simulators,
            self.robot_state_slices,
            self.robot_action_slices,
        ):
            next_parts.append(sim.casadi_dynamics(x[state_slice], u[action_slice]))
        if len(next_parts) == 1:
            return next_parts[0]
        return ca.vertcat(*next_parts)

    def get_dataset_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        env_names: list[str] = []
        mask_names: list[str] = []
        state_names: list[str] = []
        action_names: list[str] = []
        env_dim = 0
        mask_dim = 0
        state_dim = 0
        action_dim = 0

        for robot_id, sim in enumerate(self.simulators):
            base_features = sim.get_dataset_features()

            env_feature = base_features.get("observation.environment_state")
            proprio_feature = base_features.get("observation.state")
            action_feature = base_features.get("action")
            if env_feature is None or proprio_feature is None or action_feature is None:
                raise ValueError(
                    "Each robot simulator must define observation.environment_state, "
                    "observation.state, and action features."
                )

            robot_env_dim = int(env_feature["shape"][0]) + 2 * (len(self.simulators) - 1)
            robot_mask_dim = len(self.simulators) - 1
            robot_state_dim = int(proprio_feature["shape"][0])
            robot_action_dim = int(action_feature["shape"][0])

            env_dim += robot_env_dim
            mask_dim += robot_mask_dim
            state_dim += robot_state_dim
            action_dim += robot_action_dim

            env_names.extend([f"robot_{robot_id}.{name}" for name in env_feature.get("names", [])])
            for other_id in range(len(self.simulators)):
                if other_id == robot_id:
                    continue
                env_names.extend([
                    f"robot_{robot_id}.rel_robot_{other_id}_x",
                    f"robot_{robot_id}.rel_robot_{other_id}_y",
                ])
                mask_names.append(f"robot_{robot_id}.neighbor_mask_robot_{other_id}")

            state_names.extend([f"robot_{robot_id}.{name}" for name in proprio_feature.get("names", [])])
            action_names.extend([f"robot_{robot_id}.{name}" for name in action_feature.get("names", [])])

        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (env_dim,),
            "names": env_names,
        }
        features["observation.neighbor_mask"] = {
            "dtype": "float32",
            "shape": (mask_dim,),
            "names": mask_names,
        }
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": action_names,
        }
        return features

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        return self._sample_safe_initial_state(rng=rng, randomize_goals=False)

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        for sim in self.simulators:
            sim.randomize_goal_for_reset(rng)

    def reset_random(self) -> np.ndarray:
        return self.reset(
            self._sample_safe_initial_state(rng=self._sampling_rng, randomize_goals=True)
        )

    def _positions_respect_d_safe(self, states: list[np.ndarray]) -> bool:
        if self.d_safe <= 0.0 or len(states) < 2:
            return True

        min_dist_sq = float(self.d_safe * self.d_safe)
        for robot_idx, state in enumerate(states):
            if state.shape[0] < 2:
                raise ValueError(
                    f"Robot {robot_idx} state must include x/y in first two entries for d_safe checks."
                )

        for i in range(len(states)):
            p_i = states[i][:2]
            for j in range(i + 1, len(states)):
                p_j = states[j][:2]
                if float(np.sum((p_i - p_j) ** 2)) < min_dist_sq:
                    return False
        return True

    def _sample_safe_initial_state(
        self,
        rng: np.random.Generator,
        randomize_goals: bool,
    ) -> np.ndarray:
        for _ in range(SAFE_INITIAL_STATE_MAX_ATTEMPTS):
            states: list[np.ndarray] = []
            for sim in self.simulators:
                if randomize_goals:
                    sim.randomize_goal_for_reset(rng)
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

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, Any]:
        split_obs = self._split_observation(obs)
        split_action = self._split_action(action)

        env_tensors: list[np.ndarray] = []
        mask_tensors: list[np.ndarray] = []
        state_tensors: list[np.ndarray] = []
        action_tensors: list[np.ndarray] = []
        for robot_id, (sim, robot_obs, robot_action) in enumerate(
            zip(self.simulators, split_obs, split_action)
        ):
            base_env_dim = self.robot_env_dims[robot_id]
            rel_dim = self.robot_relative_dims[robot_id]
            mask_dim = self.robot_neighbor_mask_dims[robot_id]
            proprio_dim = self.robot_proprio_dims[robot_id]
            env_end = base_env_dim + rel_dim
            mask_start = env_end
            mask_end = mask_start + mask_dim
            proprio_start = mask_end
            proprio_end = proprio_start + proprio_dim

            env_tensors.append(np.asarray(robot_obs[:env_end], dtype=np.float32))
            mask_tensors.append(np.asarray(robot_obs[mask_start:mask_end], dtype=np.float32))
            state_tensors.append(np.asarray(robot_obs[proprio_start:proprio_end], dtype=np.float32))
            action_tensors.append(np.asarray(robot_action, dtype=np.float32))

        return {
            "observation.environment_state": np.concatenate(env_tensors),
            "observation.neighbor_mask": np.concatenate(mask_tensors) if len(mask_tensors) > 0 else np.empty(0, dtype=np.float32),
            "observation.state": np.concatenate(state_tensors),
            "action": np.concatenate(action_tensors),
        }

    def _validate_homogeneous_decentralized_dimensions(self) -> tuple[int, int, int]:
        env_dim = int(self.robot_env_dims[0])
        proprio_dim = int(self.robot_proprio_dims[0])
        action_dim = int(self.simulators[0].nu)

        for robot_idx, sim in enumerate(self.simulators[1:], start=1):
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

    def decentralized_policy_observation(self, obs: np.ndarray, robot_id: int) -> dict[str, np.ndarray]:
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

        if mask_dim == 0:
            neighbor_obs = np.empty((0, 2), dtype=np.float32)
            neighbor_mask_2d = np.empty((0, 1), dtype=np.float32)
        else:
            neighbor_obs = neighbor_state.reshape(mask_dim, 2)
            neighbor_mask_2d = neighbor_mask.reshape(mask_dim, 1)

        return {
            "ego_obs": ego_obs,
            "neighbor_obs": neighbor_obs,
            "neighbor_mask": neighbor_mask_2d,
        }

    def get_decentralized_dataset_features(self) -> dict[str, Any]:
        env_dim, proprio_dim, action_dim = self._validate_homogeneous_decentralized_dimensions()
        neighbor_count = len(self.simulators) - 1

        env_feature = self.simulators[0].get_dataset_features().get("observation.environment_state")
        proprio_feature = self.simulators[0].get_dataset_features().get("observation.state")
        action_feature = self.simulators[0].get_dataset_features().get("action")
        if env_feature is None or proprio_feature is None or action_feature is None:
            raise ValueError(
                "Each robot simulator must define observation.environment_state, observation.state, and action features."
            )

        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (env_dim,),
                "names": [name for name in env_feature.get("names", [])],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (proprio_dim,),
                "names": [name for name in proprio_feature.get("names", [])],
            },
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
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [name for name in action_feature.get("names", [])],
            },
        }

    def format_decentralized_dataset_frames(self, obs: np.ndarray, action: np.ndarray) -> list[dict[str, Any]]:
        self._validate_homogeneous_decentralized_dimensions()
        split_obs = self._split_observation(obs)
        split_action = self._split_action(action)

        frames: list[dict[str, Any]] = []
        for robot_id, robot_action in enumerate(split_action):
            robot_policy_obs = self.decentralized_policy_observation(obs, robot_id)
            frames.append(
                {
                    "observation.environment_state": np.asarray(
                        robot_policy_obs["ego_obs"][: self.robot_env_dims[robot_id]],
                        dtype=np.float32,
                    ),
                    "observation.state": np.asarray(
                        robot_policy_obs["ego_obs"][self.robot_env_dims[robot_id] :],
                        dtype=np.float32,
                    ),
                    "observation.neighbor_state": np.asarray(
                        robot_policy_obs["neighbor_obs"].reshape(-1),
                        dtype=np.float32,
                    ),
                    "observation.neighbor_mask": np.asarray(
                        robot_policy_obs["neighbor_mask"].reshape(-1),
                        dtype=np.float32,
                    ),
                    "action": np.asarray(robot_action, dtype=np.float32),
                }
            )

        return frames
