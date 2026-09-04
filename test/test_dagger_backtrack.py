from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.dagger.rollouts import (
    MAX_BACKTRACK_CANDIDATES,
    _geometric_backtrack_indices,
    collect_dagger_rollouts,
)
from systems.seed_utils import initial_state_seed_for_rollout


class _FakeSimulator:
    """Minimal 1-D fleet-of-1 stand-in exercising only what collect_dagger_rollouts needs.

    ``predict_next_state`` (used for the pre-execution safety checks) and
    ``step`` (the actual transition) are allowed to diverge via
    ``first_step_disturbance``, so a test can force the real, post-execution
    collision path distinctly from the raw-expert-label collision path.
    """

    def __init__(self, collision_threshold: float, goal_threshold: float) -> None:
        self.goal = np.array([0.0], dtype=float)
        self.simulators = [self]
        self.num_robots = 1
        self.robot_state_slices = [slice(0, 1)]
        self.nx = 1
        self.collision_threshold = collision_threshold
        self.goal_threshold = goal_threshold
        self.state: np.ndarray | None = None
        self.step_call_count = 0
        self.first_step_disturbance = 0.0

    @property
    def goal_dim(self) -> int:
        return 1

    def set_goal(self, goal: np.ndarray) -> None:
        self.goal = np.asarray(goal, dtype=float).copy()

    def reset(self, state: np.ndarray) -> np.ndarray:
        self.state = np.asarray(state, dtype=float).copy()
        return self.state.copy()

    def observe(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float).copy()

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float) + np.asarray(action, dtype=float)

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.step_call_count += 1
        next_state = self.predict_next_state(state, action)
        if self.step_call_count == 1:
            next_state = next_state + self.first_step_disturbance
        return next_state

    def is_collision(self, state: np.ndarray) -> bool:
        return bool(np.asarray(state, dtype=float)[0] >= self.collision_threshold)

    def should_terminate_rollout(self, state: np.ndarray) -> bool:
        return bool(np.asarray(state, dtype=float)[0] >= self.goal_threshold)


class _ScriptedPlanner:
    """Returns an unsafe (colliding) action on its first call, then a safe one afterwards."""

    def __init__(self, unsafe_action: list[float], safe_action: list[float]) -> None:
        self.unsafe_action = np.asarray(unsafe_action, dtype=float)
        self.safe_action = np.asarray(safe_action, dtype=float)
        self.call_count = 0
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return (self.unsafe_action if self.call_count == 1 else self.safe_action).copy()


class _ConstantPlanner:
    def __init__(self, action: list[float]) -> None:
        self.action = np.asarray(action, dtype=float)
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return self.action.copy()


class _FakeDatasetWriter:
    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []
        self.episode_boundaries: list[int] = []

    def add_frame(self, frame: dict[str, object]) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.episode_boundaries.append(len(self.frames))


def _frame_builder(observation: np.ndarray, action: np.ndarray) -> dict[str, object]:
    return {"observation": np.asarray(observation).tolist(), "action": np.asarray(action).tolist()}


class GeometricBacktrackIndicesTests(unittest.TestCase):
    def test_empty_history_returns_no_candidates(self) -> None:
        self.assertEqual(_geometric_backtrack_indices(0), [])

    def test_single_state_history_returns_only_index_zero(self) -> None:
        self.assertEqual(_geometric_backtrack_indices(1), [0])

    def test_candidate_count_is_bounded_regardless_of_horizon_length(self) -> None:
        for history_len in (5, 50, 200, 2000):
            indices = _geometric_backtrack_indices(history_len)
            # The hard budget cap plus the always-included s_0 fallback bounds the
            # count by a small constant, independent of the episode length -- this
            # is what turns the O(T^2) worst case into O(K * T) with K constant.
            self.assertLessEqual(len(indices), MAX_BACKTRACK_CANDIDATES + 1)

    def test_most_recent_state_and_s0_are_always_included(self) -> None:
        for history_len in (1, 2, 3, 12, 47, 200):
            indices = _geometric_backtrack_indices(history_len)
            self.assertEqual(indices[0], history_len - 1)
            self.assertEqual(indices[-1], 0)

    def test_indices_are_strictly_decreasing_and_in_range(self) -> None:
        indices = _geometric_backtrack_indices(200)
        self.assertTrue(all(0 <= idx < 200 for idx in indices))
        self.assertTrue(all(earlier > later for earlier, later in zip(indices, indices[1:])))

    def test_short_history_within_budget_is_scanned_without_gaps_beyond_stride(self) -> None:
        # With a small history the schedule should still terminate cleanly and
        # always land on 0 even when the natural stride would otherwise overshoot it.
        indices = _geometric_backtrack_indices(3)
        self.assertEqual(indices[-1], 0)
        self.assertEqual(indices[0], 2)

    def test_long_history_expands_geometric_spacing(self) -> None:
        indices = _geometric_backtrack_indices(2_000)
        self.assertIn(1_872, indices)
        self.assertIn(1_488, indices)

    def test_round_seeds_are_distinct_and_reproducible(self) -> None:
        first_round_seed = initial_state_seed_for_rollout(0, rollout_index=1, round_index=0)
        second_round_seed = initial_state_seed_for_rollout(0, rollout_index=1, round_index=1)

        self.assertNotEqual(first_round_seed, second_round_seed)
        self.assertEqual(
            first_round_seed,
            initial_state_seed_for_rollout(0, rollout_index=1, round_index=0),
        )


class CollectDaggerRolloutsBacktrackTests(unittest.TestCase):
    """End-to-end coverage of the backtrack/recovery path through collect_dagger_rollouts.

    Both call sites that trigger backtrack_and_complete() are exercised: the
    raw-expert-label pre-check (predict_next_state says the label is unsafe
    before it's ever executed) and the actual post-execution collision (the
    label looked safe but stepping the simulator wasn't). In both cases the
    scripted planner/simulator make exactly one candidate (s_0) succeed on
    retry, so the test can assert precisely which frame ends up written.
    """

    def test_unsafe_expert_label_triggers_backtrack_and_only_recovered_frame_is_saved(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        planner = _ScriptedPlanner(unsafe_action=[100.0], safe_action=[1.0])
        writer = _FakeDatasetWriter()

        metrics = collect_dagger_rollouts(
            simulator=simulator,
            expert_planner=planner,
            dataset_writer=writer,
            trajectories_per_iteration=1,
            steps_per_trajectory=2,
            action_noise_std=0.0,
            action_noise_seed=0,
            initial_state_seed=0,
            expert_mixing_beta=1.0,
            policy_action_fn=None,
            frame_builder=_frame_builder,
            initial_states=[[0.0]],
            goal_states=None,
        )

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(metrics.min_steps, 1)
        self.assertEqual(metrics.max_steps, 1)
        # The unsafe first attempt (action=[100.0]) must never reach the
        # dataset -- only the recovered, expert-only-control frame is saved.
        self.assertEqual(len(writer.frames), 1)
        self.assertEqual(writer.frames[0]["action"], [1.0])
        self.assertEqual(writer.episode_boundaries, [1])

    def test_post_execution_collision_triggers_backtrack_and_only_recovered_frame_is_saved(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        simulator.first_step_disturbance = 20.0  # only the very first .step() call collides
        planner = _ConstantPlanner(action=[1.0])
        writer = _FakeDatasetWriter()

        metrics = collect_dagger_rollouts(
            simulator=simulator,
            expert_planner=planner,
            dataset_writer=writer,
            trajectories_per_iteration=1,
            steps_per_trajectory=2,
            action_noise_std=0.0,
            action_noise_seed=0,
            initial_state_seed=0,
            expert_mixing_beta=1.0,
            policy_action_fn=None,
            frame_builder=_frame_builder,
            initial_states=[[0.0]],
            goal_states=None,
        )

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(metrics.min_steps, 1)
        self.assertEqual(metrics.max_steps, 1)
        self.assertEqual(len(writer.frames), 1)
        self.assertEqual(writer.frames[0]["action"], [1.0])
        self.assertEqual(writer.episode_boundaries, [1])


if __name__ == "__main__":
    unittest.main()
