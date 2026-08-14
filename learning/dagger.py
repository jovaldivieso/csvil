from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torch

from planning.casadi_planner import PlannerSolveError
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol
from systems.seed_utils import (
    DEFAULT_MULTI_ROBOT_SEED_STRIDE,
    action_noise_seed_for_rollout,
    action_noise_rng_for_rollout,
    expert_mixing_seed_for_rollout,
    initial_state_seed_for_rollout,
)


@dataclass(frozen=True)
class DaggerEvalMetrics:
    success_rate: float
    mean_steps: float
    min_steps: int
    max_steps: int
    num_episodes: int


@dataclass
class ExpertMixBetaController:
    beta_start: float
    beta_end: float
    decay_rounds: int
    beta_decay_rate: float | None = None
    decay_after_success_rate: float | None = None
    adaptive_recovery: bool = False

    current_beta: float = 0.0
    decay_active: bool = False
    previous_eval_success_rate: float | None = None

    def __post_init__(self) -> None:
        self.current_beta = float(self.beta_start)

    def _gate_is_open(self, eval_success_rate: float) -> bool:
        if self.decay_after_success_rate is None:
            return True
        return float(eval_success_rate) > float(self.decay_after_success_rate)

    def _step_delta(self) -> float:
        if self.beta_decay_rate is not None:
            return abs(float(self.beta_decay_rate))

        if self.decay_rounds <= 1:
            return 0.0

        return abs(float(self.beta_end) - float(self.beta_start)) / float(self.decay_rounds - 1)

    def _beta_bounds(self) -> tuple[float, float]:
        if self.beta_decay_rate is not None:
            # Additive mode follows beta_t = max(0, beta_start - rate * t).
            return 0.0, max(0.0, float(self.beta_start))
        return min(float(self.beta_start), float(self.beta_end)), max(float(self.beta_start), float(self.beta_end))

    def _decrease_beta(self) -> None:
        delta = self._step_delta()
        if self.beta_decay_rate is not None:
            next_beta = self.current_beta - delta
        else:
            schedule_direction = 1.0 if float(self.beta_end) > float(self.beta_start) else -1.0
            next_beta = self.current_beta + schedule_direction * delta

        lower_bound, upper_bound = self._beta_bounds()
        self.current_beta = float(min(max(next_beta, lower_bound), upper_bound))

    def _increase_beta(self) -> None:
        delta = self._step_delta()
        # Recovery must always increase expert mixing, independent of decay schedule direction.
        next_beta = self.current_beta + delta

        lower_bound, upper_bound = self._beta_bounds()
        self.current_beta = float(min(max(next_beta, lower_bound), upper_bound))

    def update_after_evaluation(self, eval_success_rate: float | None) -> None:
        if eval_success_rate is None:
            if self.decay_after_success_rate is None:
                if not self.decay_active:
                    self.decay_active = True
                self._decrease_beta()
            return

        eval_rate = float(eval_success_rate)

        if not self.decay_active:
            if self._gate_is_open(eval_rate):
                self.decay_active = True
                self._decrease_beta()
            self.previous_eval_success_rate = eval_rate
            return

        previous_rate = self.previous_eval_success_rate
        if self.adaptive_recovery and previous_rate is not None and eval_rate < previous_rate:
            self._increase_beta()
        else:
            self._decrease_beta()

        self.previous_eval_success_rate = eval_rate

    def prime_from_evaluation(self, eval_success_rate: float | None) -> None:
        """Prime gate state from an evaluation without advancing the beta schedule."""
        if eval_success_rate is None:
            return

        eval_rate = float(eval_success_rate)
        if self._gate_is_open(eval_rate):
            self.decay_active = True
        self.previous_eval_success_rate = eval_rate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def with_seeded_initial_state_config(
    system_name: str,
    config: Mapping[str, object],
    base_seed: int,
) -> dict[str, object]:
    """Ensure all simulator RNG entrypoints get deterministic initial-state seeds."""
    seeded_config = dict(config)

    if system_name != "multi_robot":
        seeded_config.setdefault("initial_state_seed", int(base_seed))
        return seeded_config

    seeded_config.setdefault("initial_state_seed", int(base_seed))
    robots_raw = seeded_config.get("robots", [])
    seeded_robots: list[dict[str, object]] = []
    for robot_idx, robot_entry in enumerate(robots_raw):
        if not isinstance(robot_entry, Mapping):
            seeded_robots.append(dict(robot_entry))
            continue

        robot_system = robot_entry.get("system")
        robot_cfg_raw = robot_entry.get("config", {})
        robot_cfg = dict(robot_cfg_raw) if isinstance(robot_cfg_raw, Mapping) else {}
        robot_cfg.setdefault("initial_state_seed", int(base_seed + 1000 * (robot_idx + 1)))
        seeded_robots.append(
            {
                "system": robot_system,
                "config": robot_cfg,
            }
        )

    seeded_config["robots"] = seeded_robots
    return seeded_config


def resolve_initial_state_seed(config: Mapping[str, object], fallback_seed: int) -> int:
    return int(config.get("initial_state_seed", fallback_seed))


def is_observation_feature(feature_name: str) -> bool:
    return feature_name.startswith("observation.") or ".observation." in feature_name


def observation_feature_names(simulator: DynamicsProtocol) -> list[str]:
    return [
        feature_name
        for feature_name in simulator.get_dataset_features().keys()
        if is_observation_feature(feature_name)
    ]


def action_feature_names(simulator: DynamicsProtocol) -> list[str]:
    return [
        feature_name
        for feature_name in simulator.get_dataset_features().keys()
        if feature_name == "action" or feature_name.endswith(".action")
    ]


@dataclass(frozen=True, slots=True)
class ObservationFeaturePackCache:
    feature_names: tuple[str, ...]
    feature_indices: np.ndarray
    feature_index_slices: tuple[slice, ...]


def build_observation_feature_pack_cache(
    simulator: DynamicsProtocol,
    feature_names: list[str],
    allow_schema_subset: bool = False,
) -> ObservationFeaturePackCache:
    dataset_features = simulator.get_dataset_features()
    schema_observation_features = tuple(
        feature_name
        for feature_name in dataset_features.keys()
        if is_observation_feature(feature_name)
    )
    provided_feature_names = tuple(feature_names)

    if not allow_schema_subset and provided_feature_names != schema_observation_features:
        mismatch_index = next(
            (
                idx
                for idx, (expected_name, provided_name) in enumerate(
                    zip(schema_observation_features, provided_feature_names)
                )
                if expected_name != provided_name
            ),
            None,
        )
        if mismatch_index is not None:
            mismatch_details = (
                f"first mismatch at index {mismatch_index}: "
                f"expected '{schema_observation_features[mismatch_index]}' "
                f"but got '{provided_feature_names[mismatch_index]}'"
            )
        else:
            mismatch_details = (
                "feature list lengths differ: "
                f"expected {len(schema_observation_features)} but got {len(provided_feature_names)}"
            )

        raise ValueError(
            "Observation feature ordering does not match simulator dataset schema; "
            "this can cause silent policy input misalignment. "
            f"{mismatch_details}"
        )

    total_dim = observation_dim_from_features(simulator)
    dummy_obs = np.arange(total_dim, dtype=np.float32)
    dummy_act = np.zeros(int(simulator.nu), dtype=np.float32)
    dummy_frame = simulator.format_dataset_frame(dummy_obs, dummy_act)

    feature_indices: list[np.ndarray] = []
    feature_index_slices: list[slice] = []
    start = 0
    for feature_name in provided_feature_names:
        if feature_name not in dummy_frame:
            raise KeyError(
                f"Observation feature '{feature_name}' is missing from simulator dataset formatter."
            )

        feature_array = np.asarray(dummy_frame[feature_name])
        if feature_array.ndim == 0:
            raise ValueError(
                f"Observation feature '{feature_name}' must be indexable from the formatted frame."
            )

        feature_index_array = feature_array.astype(int, copy=False).reshape(-1)
        stop = start + feature_index_array.shape[0]
        feature_indices.append(feature_index_array)
        feature_index_slices.append(slice(start, stop))
        start = stop

    if len(feature_indices) == 0:
        stacked_feature_indices = np.empty(0, dtype=int)
    else:
        stacked_feature_indices = np.concatenate(feature_indices).astype(int, copy=False)

    return ObservationFeaturePackCache(
        feature_names=provided_feature_names,
        feature_indices=stacked_feature_indices,
        feature_index_slices=tuple(feature_index_slices),
    )


def pack_observation_features_from_cache(
    observation: np.ndarray,
    feature_cache: ObservationFeaturePackCache,
) -> dict[str, np.ndarray]:
    observation_array = np.asarray(observation)
    packed_values = np.asarray(observation_array[feature_cache.feature_indices], dtype=np.float32)
    return {
        feature_name: np.asarray(packed_values[feature_slice], dtype=np.float32)
        for feature_name, feature_slice in zip(feature_cache.feature_names, feature_cache.feature_index_slices)
    }


def observation_dim_from_features(simulator: DynamicsProtocol) -> int:
    total_dim = 0
    for feature_name, feature_info in simulator.get_dataset_features().items():
        if is_observation_feature(feature_name):
            total_dim += int(feature_info["shape"][0])
    return total_dim


def pack_observation_features(
    simulator: DynamicsProtocol,
    observation: np.ndarray,
    feature_names: list[str],
    feature_cache: ObservationFeaturePackCache | None = None,
) -> dict[str, np.ndarray]:
    if feature_cache is None:
        feature_cache = build_observation_feature_pack_cache(simulator, feature_names)
    return pack_observation_features_from_cache(observation, feature_cache)


def uses_deep_set_policy(simulator: DynamicsProtocol, policy: object) -> bool:
    return bool(getattr(policy, "use_deepset", False)) and hasattr(
        simulator, "decentralized_policy_observation"
    )


def build_deep_set_policy_input(
    simulator: DynamicsProtocol,
    observation: np.ndarray,
    robot_id: int,
    device: torch.device,
    add_batch_dim: bool = True,
) -> dict[str, torch.Tensor]:
    robot_policy_obs = simulator.decentralized_policy_observation(observation, robot_id)

    policy_input = {
        name: torch.as_tensor(robot_policy_obs[name], dtype=torch.float32)
        for name in ("ego_obs", "neighbor_obs", "neighbor_mask")
    }
    if add_batch_dim:
        policy_input = {name: tensor.unsqueeze(0) for name, tensor in policy_input.items()}

    if device.type != "cpu":
        policy_input = {name: tensor.to(device) for name, tensor in policy_input.items()}

    return policy_input


def build_deep_set_joint_action(
    simulator: DynamicsProtocol,
    policy,
    observation: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Query the shared decentralized policy once per robot and concatenate local actions."""
    action_parts: list[np.ndarray] = []
    for robot_id in range(int(simulator.num_robots)):
        policy_input = build_deep_set_policy_input(
            simulator=simulator,
            observation=observation,
            robot_id=robot_id,
            device=device,
        )
        with torch.inference_mode():
            action_tensor = policy.select_action(policy_input)
        action_parts.append(action_tensor.squeeze(0).detach().cpu().numpy())

    return np.concatenate(action_parts)


def scheduled_expert_mix_beta(
    round_offset: int,
    beta_start: float,
    beta_end: float,
    decay_rounds: int,
    beta_decay_rate: float | None = None,
) -> float:
    if beta_decay_rate is not None:
        clamped_offset = max(round_offset, 0)
        return float(max(0.0, beta_start - float(beta_decay_rate) * float(clamped_offset)))

    if decay_rounds <= 1:
        return float(beta_start)

    clamped_offset = min(max(round_offset, 0), decay_rounds - 1)
    progress = float(clamped_offset) / float(decay_rounds - 1)
    return float(beta_start + (beta_end - beta_start) * progress)


def resolve_round_steps(
    num_frames: int,
    batch_size: int,
    target_epochs_per_round: float,
    max_train_steps: int | None,
) -> tuple[int, float]:
    if num_frames <= 0:
        raise ValueError("Training dataset must contain at least one frame.")
    if batch_size <= 0:
        raise ValueError("'batch_size' must be positive.")
    if target_epochs_per_round <= 0:
        raise ValueError("'target_epochs_per_round' must be positive.")

    steps = math.ceil(float(target_epochs_per_round) * float(num_frames) / float(batch_size))
    if max_train_steps is not None:
        steps = min(steps, int(max_train_steps))

    if steps <= 0:
        raise ValueError("Per-round training steps must remain positive.")
    approx_epochs = float(steps) * float(batch_size) / float(num_frames)
    return steps, approx_epochs


def print_rollout_metrics(label: str, prefix: str, metrics: DaggerEvalMetrics) -> None:
    print(
        f"{label}: {prefix}_success_rate={100.0 * metrics.success_rate:.1f}% "
        f"{prefix}_mean_steps={metrics.mean_steps:.2f} "
        f"{prefix}_min_steps={metrics.min_steps} {prefix}_max_steps={metrics.max_steps} "
        f"episodes={metrics.num_episodes}"
    )


def evaluation_seed_specs(
    simulator: DynamicsProtocol,
    num_episodes: int,
    seed_start: int,
) -> list[int | list[int]]:
    if num_episodes < 0:
        raise ValueError("'num_episodes' must be non-negative.")

    if simulator.num_robots <= 1:
        return [int(seed_start) + idx for idx in range(num_episodes)]

    return [
        [
            int(seed_start) + idx + DEFAULT_MULTI_ROBOT_SEED_STRIDE * robot_idx
            for robot_idx in range(simulator.num_robots)
        ]
        for idx in range(num_episodes)
    ]


def sample_initial_state(
    simulator: DynamicsProtocol,
    seed_spec: int | list[int],
) -> np.ndarray:
    if isinstance(seed_spec, int):
        rng = np.random.default_rng(int(seed_spec))
        simulator.randomize_goal_for_reset(rng)
        return simulator.random_initial_state(rng)

    sub_simulators = simulator.simulators
    if len(seed_spec) != len(sub_simulators):
        raise ValueError(
            "Per-robot seed specification length must match robot count. "
            f"Got {len(seed_spec)} seeds for {len(sub_simulators)} robots."
        )

    joint_seed_seq = np.random.SeedSequence([int(robot_seed) for robot_seed in seed_spec])
    rng = np.random.default_rng(joint_seed_seq)
    for sub_sim in sub_simulators:
        sub_sim.randomize_goal_for_reset(rng)

    return simulator.random_initial_state(rng)


def apply_execution_noise(
    simulator: DynamicsProtocol,
    action: np.ndarray,
    action_noise_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if action_noise_std <= 0.0:
        return action

    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(
        loc=0.0,
        scale=action_noise_std,
        size=action.shape,
    ).astype(action.dtype, copy=False)
    return np.clip(
        action + noise,
        -simulator.max_action,
        simulator.max_action,
    )


def collect_dagger_rollouts(
    simulator: DynamicsProtocol,
    expert_planner: PlannerProtocol,
    dataset_writer,
    trajectories_per_iteration: int,
    steps_per_trajectory: int,
    action_noise_std: float,
    action_noise_seed: int,
    initial_state_seed: int,
    expert_mixing_beta: float,
    policy_action_fn: Callable[[np.ndarray], np.ndarray] | None,
    policy_reset_fn: Callable[[], None] | None = None,
    frame_builder: Callable[
        [np.ndarray, np.ndarray],
        Mapping[str, object] | list[Mapping[str, object]] | tuple[Mapping[str, object], ...],
    ] | None = None,
) -> DaggerEvalMetrics:
    """Collect DAgger trajectories with expert relabeling and mixed execution."""
    if policy_action_fn is None and expert_mixing_beta < 1.0:
        raise ValueError("'policy_action_fn' is required when 'expert_mixing_beta' is less than 1.0.")
    should_query_policy = policy_action_fn is not None and expert_mixing_beta < 1.0

    successful_episodes = 0
    attempted_episodes = 0
    max_attempts = max(trajectories_per_iteration * 3, trajectories_per_iteration)
    reached_goal_count = 0
    steps_taken: list[int] = []
    expert_executed_steps = 0
    total_executed_steps = 0

    if frame_builder is None:
        def frame_builder(observation: np.ndarray, expert_action: np.ndarray) -> dict[str, object]:
            frame = simulator.format_dataset_frame(observation, expert_action)
            frame["task"] = "reach target"
            return frame

    def ensure_task_field(frame: dict[str, object]) -> dict[str, object]:
        if "task" not in frame:
            frame = dict(frame)
            frame["task"] = "reach target"
        return frame

    while successful_episodes < trajectories_per_iteration:
        attempted_episodes += 1
        if attempted_episodes > max_attempts:
            raise RuntimeError(
                "Too many failed DAgger rollout attempts. "
                f"Collected {successful_episodes}/{trajectories_per_iteration} episodes."
            )

        episode_initial_state_seed = initial_state_seed_for_rollout(
            initial_state_seed,
            rollout_index=attempted_episodes,
        )
        sampled_initial_state = sample_initial_state(
            simulator=simulator,
            seed_spec=episode_initial_state_seed,
        )
        state = simulator.reset(sampled_initial_state)
        episode_initial_state = state.copy()
        episode_noise_seed = action_noise_seed_for_rollout(
            action_noise_seed,
            rollout_index=attempted_episodes,
        )
        episode_action_noise_rng = np.random.default_rng(episode_noise_seed)
        episode_expert_mixing_seed = expert_mixing_seed_for_rollout(
            action_noise_seed,
            rollout_index=attempted_episodes,
        )
        episode_expert_mixing_rng = np.random.default_rng(episode_expert_mixing_seed)
        planner_failed = False

        if policy_reset_fn is not None:
            policy_reset_fn()
        if hasattr(expert_planner, "reset"):
            expert_planner.reset()

        reached_goal = False
        rollout_steps = steps_per_trajectory
        episode_frame_buffers: list[list[dict[str, object]]] | None = None

        for step in range(1, steps_per_trajectory + 1):
            observation = simulator.observe(state, validate=False)

            try:
                expert_action = expert_planner(observation)
            except PlannerSolveError as exc:
                print(
                    "Skipping episode due to planner failure "
                    f"(attempt={attempted_episodes}, step={step}, action_noise_std={action_noise_std:.6f}, "
                    f"noise_seed={episode_noise_seed})."
                )
                print(
                    "Planner failure context: "
                    f"initial_state={np.array2string(np.asarray(episode_initial_state), precision=6)}, "
                    f"current_state={np.array2string(np.asarray(state), precision=6)}, "
                    f"goal_state={np.array2string(np.asarray(simulator.goal_state), precision=6)}"
                )
                print(f"Underlying solver error: {exc}")
                planner_failed = True
                break

            built_frame = frame_builder(observation, expert_action)
            if isinstance(built_frame, (list, tuple)):
                if episode_frame_buffers is None:
                    episode_frame_buffers = [[] for _ in built_frame]
                elif len(episode_frame_buffers) != len(built_frame):
                    raise ValueError(
                        "'frame_builder' must return the same number of per-robot frames "
                        "at every rollout step."
                    )

                for robot_idx, frame in enumerate(built_frame):
                    if not isinstance(frame, Mapping):
                        raise TypeError(
                            "'frame_builder' list/tuple items must be mappings, "
                            f"got {type(frame).__name__}."
                        )
                    episode_frame_buffers[robot_idx].append(ensure_task_field(dict(frame)))
            elif isinstance(built_frame, Mapping):
                if episode_frame_buffers is None:
                    episode_frame_buffers = [[]]
                elif len(episode_frame_buffers) != 1:
                    raise ValueError(
                        "'frame_builder' cannot switch between per-robot and single-frame "
                        "outputs within one rollout."
                    )
                episode_frame_buffers[0].append(ensure_task_field(dict(built_frame)))
            else:
                raise TypeError(
                    "'frame_builder' must return a mapping or a list/tuple of mappings, "
                    f"got {type(built_frame).__name__}."
                )

            use_expert_action = bool(episode_expert_mixing_rng.random() < expert_mixing_beta)
            policy_action = None
            # Keep stateful policies (e.g. ACT/Diffusion with history/action queues)
            # synchronized with environment time even on expert-executed steps.
            if should_query_policy:
                policy_action = policy_action_fn(observation)
            if use_expert_action or policy_action is None:
                base_action = expert_action
            else:
                base_action = policy_action
            expert_executed_steps += int(use_expert_action)
            total_executed_steps += 1

            executed_action = apply_execution_noise(
                simulator=simulator,
                action=base_action,
                action_noise_std=action_noise_std,
                rng=episode_action_noise_rng,
            )
            state = simulator.step(state, executed_action, validate=False)

            if simulator.should_terminate_rollout(state):
                reached_goal = True
                rollout_steps = step
                break

        if planner_failed:
            continue

        if episode_frame_buffers is None:
            raise RuntimeError("'frame_builder' produced no frames for the rollout.")
        for frame_buffer in episode_frame_buffers:
            for frame_data in frame_buffer:
                dataset_writer.add_frame(frame_data)
            dataset_writer.save_episode()
        successful_episodes += 1
        reached_goal_count += int(reached_goal)
        steps_taken.append(int(rollout_steps))

        if successful_episodes % 10 == 0:
            print(
                "Collected "
                f"{successful_episodes}/{trajectories_per_iteration} trajectories"
            )

    if total_executed_steps > 0:
        realized_expert_fraction = float(expert_executed_steps) / float(total_executed_steps)
        print(
            "Execution mixing stats: "
            f"requested_beta={expert_mixing_beta:.3f}, "
            f"realized_expert_fraction={realized_expert_fraction:.3f}, "
            f"executed_steps={total_executed_steps}"
        )

    return DaggerEvalMetrics(
        success_rate=float(reached_goal_count) / float(successful_episodes),
        mean_steps=float(np.mean(np.asarray(steps_taken, dtype=float))),
        min_steps=min(steps_taken),
        max_steps=max(steps_taken),
        num_episodes=successful_episodes,
    )


def rollout_policy_with_action_fn(
    simulator: DynamicsProtocol,
    initial_state: np.ndarray,
    num_steps: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None] | None = None,
    action_noise_std: float = 0.0,
    action_noise_rng: np.random.Generator | None = None,
) -> tuple[bool, int]:
    state = simulator.reset(initial_state)
    if reset_fn is not None:
        reset_fn()

    if simulator.should_terminate_rollout(state):
        return True, 0

    for step in range(1, num_steps + 1):
        observation = simulator.observe(state, validate=False)
        action = action_fn(observation)
        executed_action = apply_execution_noise(
            simulator=simulator,
            action=action,
            action_noise_std=action_noise_std,
            rng=action_noise_rng,
        )
        state = simulator.step(state, executed_action, validate=False)

        if simulator.should_terminate_rollout(state):
            return True, step

    return False, num_steps


def evaluate_policy_rollouts(
    simulator: DynamicsProtocol,
    num_episodes: int,
    num_steps: int,
    seed_start: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None] | None = None,
    action_noise_std: float = 0.0,
    action_noise_seed: int = 0,
) -> DaggerEvalMetrics | None:
    if num_episodes == 0:
        return None
    if num_steps <= 0:
        raise ValueError("'num_steps' must be positive.")

    seed_specs = evaluation_seed_specs(
        simulator=simulator,
        num_episodes=num_episodes,
        seed_start=seed_start,
    )

    successes = 0
    steps_taken: list[int] = []
    for seed_spec in seed_specs:
        initial_state = sample_initial_state(simulator=simulator, seed_spec=seed_spec)
        action_noise_rng = action_noise_rng_for_rollout(
            action_noise_seed,
            seed_spec=seed_spec,
        )
        reached_goal, rollout_steps = rollout_policy_with_action_fn(
            simulator=simulator,
            initial_state=initial_state,
            num_steps=num_steps,
            action_fn=action_fn,
            reset_fn=reset_fn,
            action_noise_std=action_noise_std,
            action_noise_rng=action_noise_rng,
        )
        successes += int(reached_goal)
        steps_taken.append(int(rollout_steps))

    return DaggerEvalMetrics(
        success_rate=float(successes) / float(num_episodes),
        mean_steps=float(np.mean(np.asarray(steps_taken, dtype=float))),
        min_steps=min(steps_taken),
        max_steps=max(steps_taken),
        num_episodes=num_episodes,
    )