from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch

from planning.casadi_planner import PlannerSolveError
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol
from systems.seed_utils import (
    action_noise_rng_for_rollout,
    action_noise_seed_for_rollout,
    expert_mixing_seed_for_rollout,
    initial_state_seed_for_rollout,
)

from .metrics import DaggerEvalMetrics
from .utils import evaluation_seed_specs, sample_initial_state
from systems.initial_state_utils import normalize_initial_state_specs


def build_decentralized_joint_action(
    simulator: DynamicsProtocol,
    policy,
    observation: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Query the shared decentralized policy once for the entire robot fleet."""
    robot_observations = [
        simulator.decentralized_policy_observation(observation, robot_id)
        for robot_id in range(int(simulator.num_robots))
    ]
    policy_input = {
        name: torch.as_tensor(
            np.stack([robot_observation[name] for robot_observation in robot_observations]),
            dtype=torch.float32,
        ).to(device)
        for name in robot_observations[0]
    }

    with torch.inference_mode():
        action_tensor = policy.select_action(policy_input)

    if action_tensor.ndim == 3:
        action_tensor = action_tensor[:, 0, :]
    elif action_tensor.ndim != 2:
        raise ValueError(
            "Decentralized policy must return actions shaped "
            "(robots, horizon, action_dim) or (robots, action_dim)."
        )
    return action_tensor.detach().cpu().numpy().reshape(-1)


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
    noise = rng.normal(0.0, action_noise_std, size=action.shape).astype(action.dtype, copy=False)
    return np.clip(action + noise, -simulator.max_action, simulator.max_action)


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
    frame_builder: Callable[[np.ndarray, np.ndarray], Mapping[str, object] | list[Mapping[str, object]] | tuple[Mapping[str, object], ...]] | None = None,
    initial_states: list[np.ndarray] | None = None,
) -> DaggerEvalMetrics:
    """Collect DAgger trajectories with expert relabeling and mixed execution."""
    if policy_action_fn is None and expert_mixing_beta < 1.0:
        raise ValueError("'policy_action_fn' is required when 'expert_mixing_beta' is less than 1.0.")
    should_query_policy = policy_action_fn is not None and expert_mixing_beta < 1.0
    successful_episodes = attempted_episodes = reached_goal_count = 0
    max_attempts = max(trajectories_per_iteration * 3, trajectories_per_iteration)
    steps_taken: list[int] = []
    expert_executed_steps = total_executed_steps = 0

    if frame_builder is None:
        def frame_builder(
            observation: np.ndarray,
            expert_action: np.ndarray,
        ) -> list[dict[str, object]]:
            frames = simulator.format_dataset_frame(observation, expert_action)
            for frame in frames:
                frame["task"] = "reach target"
            return frames

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
        provided_initial_states = normalize_initial_state_specs(simulator, initial_states)
        if attempted_episodes <= len(provided_initial_states):
            sampled_initial_state = provided_initial_states[attempted_episodes - 1]
        else:
            episode_initial_state_seed = initial_state_seed_for_rollout(
                initial_state_seed,
                rollout_index=attempted_episodes,
            )
            sampled_initial_state = sample_initial_state(simulator, episode_initial_state_seed)
        state = simulator.reset(sampled_initial_state)
        if simulator.is_collision(state):
            print(
                "Skipping DAgger episode due to colliding initial state "
                f"(attempt={attempted_episodes})."
            )
            continue
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
        expert_label_collision = False
        if policy_reset_fn is not None:
            policy_reset_fn()
        if hasattr(expert_planner, "reset"):
            expert_planner.reset()
        reached_goal = False
        rollout_steps = steps_per_trajectory
        episode_frame_buffers: list[list[dict[str, object]]] | None = None

        for step in range(1, steps_per_trajectory + 1):
            observation = simulator.observe(state)
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

            expert_next_state = simulator.predict_next_state(state, expert_action)
            expert_label_collided, expert_label_summary = _detect_collision(
                simulator, expert_next_state
            )
            if expert_label_collided:
                print(
                    "Discarding DAgger episode because the expert corrective label "
                    "leads to collision "
                    f"(attempt={attempted_episodes}, step={step}, {expert_label_summary})."
                )
                expert_label_collision = True
                break

            built_frame = frame_builder(observation, expert_action)
            if isinstance(built_frame, (list, tuple)):
                if episode_frame_buffers is None:
                    episode_frame_buffers = [[] for _ in built_frame]
                elif len(episode_frame_buffers) != len(built_frame):
                    raise ValueError("'frame_builder' must return the same number of per-robot frames at every rollout step.")
                for robot_idx, frame in enumerate(built_frame):
                    if not isinstance(frame, Mapping):
                        raise TypeError("'frame_builder' list/tuple items must be mappings, got " f"{type(frame).__name__}.")
                    episode_frame_buffers[robot_idx].append(ensure_task_field(dict(frame)))
            elif isinstance(built_frame, Mapping):
                if episode_frame_buffers is None:
                    episode_frame_buffers = [[]]
                elif len(episode_frame_buffers) != 1:
                    raise ValueError("'frame_builder' cannot switch between per-robot and single-frame outputs within one rollout.")
                episode_frame_buffers[0].append(ensure_task_field(dict(built_frame)))
            else:
                raise TypeError("'frame_builder' must return a mapping or a list/tuple of mappings, got " f"{type(built_frame).__name__}.")

            use_expert_action = bool(episode_expert_mixing_rng.random() < expert_mixing_beta)
            policy_action = policy_action_fn(observation) if should_query_policy else None
            base_action = expert_action if use_expert_action or policy_action is None else policy_action
            expert_executed_steps += int(use_expert_action)
            total_executed_steps += 1
            state = simulator.step(
                state,
                apply_execution_noise(simulator, base_action, action_noise_std, episode_action_noise_rng),
            )
            collided, summary = _detect_collision(simulator, state)
            if collided:
                print(
                    "Stopping DAgger episode after collision "
                    f"(attempt={attempted_episodes}, step={step}, "
                    f"actor={'expert' if use_expert_action else 'policy'}, "
                    f"{summary})."
                )
                rollout_steps = step
                break
            if simulator.should_terminate_rollout(state):
                reached_goal = True
                rollout_steps = step
                break

        if planner_failed or expert_label_collision:
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
            print(f"Collected {successful_episodes}/{trajectories_per_iteration} trajectories")

    if total_executed_steps > 0:
        print(
            "Execution mixing stats: "
            f"requested_beta={expert_mixing_beta:.3f}, "
            f"realized_expert_fraction={float(expert_executed_steps) / float(total_executed_steps):.3f}, "
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
    if simulator.is_collision(state):
        return False, 0
    if simulator.should_terminate_rollout(state):
        return True, 0
    for step in range(1, num_steps + 1):
        action = action_fn(simulator.observe(state))
        state = simulator.step(state, apply_execution_noise(simulator, action, action_noise_std, action_noise_rng))
        if simulator.is_collision(state):
            return False, step
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
    seed_specs = evaluation_seed_specs(simulator, num_episodes, seed_start)
    successes = 0
    steps_taken: list[int] = []
    for seed_spec in seed_specs:
        initial_state = sample_initial_state(simulator, seed_spec)
        reached_goal, rollout_steps = rollout_policy_with_action_fn(
            simulator=simulator,
            initial_state=initial_state,
            num_steps=num_steps,
            action_fn=action_fn,
            reset_fn=reset_fn,
            action_noise_std=action_noise_std,
            action_noise_rng=action_noise_rng_for_rollout(action_noise_seed, seed_spec=seed_spec),
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
