from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.dagger.rollouts import (
    MAX_BACKTRACK_CANDIDATES,
    _geometric_backtrack_indices,
    collect_dagger_rollouts,
    evaluate_policy_rollouts,
)
from planning.casadi_planner import PlannerSolveError
from systems.seed_utils import (
    action_noise_seed_for_rollout,
    initial_state_seed_for_rollout,
    torch_inference_seed_for_rollout,
)


class _FakeSimulator:
    """Minimal 1-D fleet-of-1 stand-in exercising only what collect_dagger_rollouts needs.

    ``predict_next_state`` (used for the pre-execution safety checks) and
    ``step`` (the actual transition) are allowed to diverge via
    ``first_step_disturbance``, so a test can force the real, post-execution
    collision path distinctly from the raw-expert-label collision path.
    """

    def __init__(self, collision_threshold: float, goal_threshold: float, done_hold_steps: int = 1) -> None:
        self.goal = np.array([0.0], dtype=float)
        self.simulators = [self]
        self.num_robots = 1
        self.robot_state_slices = [slice(0, 1)]
        self.nx = 1
        self.collision_threshold = collision_threshold
        self.goal_threshold = goal_threshold
        self.done_hold_steps = done_hold_steps
        self.state: np.ndarray | None = None
        self.step_call_count = 0
        self.first_step_disturbance = 0.0
        self._rollout_done_counter = 0

    @property
    def goal_dim(self) -> int:
        return 1

    @property
    def goal_state(self) -> np.ndarray:
        return self.goal.copy()

    def set_goal(self, goal: np.ndarray) -> None:
        self.goal = np.asarray(goal, dtype=float).copy()

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        self.goal = np.array([rng.uniform(-1.0, 1.0)], dtype=float)

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.uniform(-1.0, 1.0)], dtype=float)

    def reset(self, state: np.ndarray) -> np.ndarray:
        self.state = np.asarray(state, dtype=float).copy()
        self._rollout_done_counter = 0
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

    def is_done(self, state: np.ndarray) -> bool:
        return bool(np.asarray(state, dtype=float)[0] >= self.goal_threshold)

    def should_terminate_rollout(self, state: np.ndarray) -> bool:
        """Mirrors DynamicsSimulator.should_terminate_rollout's consecutive-hold-steps contract."""
        if self.is_done(state):
            self._rollout_done_counter += 1
        else:
            self._rollout_done_counter = 0
        return self._rollout_done_counter >= self.done_hold_steps


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


class _SequencedPlanner:
    """Returns actions from an explicit per-call sequence, repeating the last once exhausted."""

    def __init__(self, actions: list[list[float]]) -> None:
        self.actions = [np.asarray(action, dtype=float) for action in actions]
        self.call_count = 0
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        self.call_count += 1
        index = min(self.call_count, len(self.actions)) - 1
        return self.actions[index].copy()


class _FailThenSucceedPlanner:
    """Raises PlannerSolveError (a genuine solve failure) on its first call, then
    returns a constant safe action -- unlike _ScriptedPlanner, which always
    successfully solves but returns an action that's unsafe to execute."""

    def __init__(self, safe_action: list[float]) -> None:
        self.safe_action = np.asarray(safe_action, dtype=float)
        self.call_count = 0
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        self.call_count += 1
        if self.call_count == 1:
            raise PlannerSolveError("forced solver failure for test")
        return self.safe_action.copy()


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

    All three call sites that trigger backtrack_and_complete() are exercised:
    the raw-expert-label pre-check (predict_next_state says the label is
    unsafe before it's ever executed), the actual post-execution collision
    (the label looked safe but stepping the simulator wasn't), and a genuine
    planner solve failure (PlannerSolveError, e.g. an infeasible or
    unsolvable state -- not merely an unsafe-but-solved label). In every case
    the scripted planner/simulator make exactly one candidate (s_0) succeed
    on retry, so the test can assert precisely which frame ends up written.
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

    def test_planner_solve_failure_triggers_backtrack_and_only_recovered_frame_is_saved(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        planner = _FailThenSucceedPlanner(safe_action=[1.0])
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
        # The failed first attempt must not discard the episode outright --
        # only the recovered, expert-only-control frame is saved.
        self.assertEqual(len(writer.frames), 1)
        self.assertEqual(writer.frames[0]["action"], [1.0])
        self.assertEqual(writer.episode_boundaries, [1])

    def test_backtrack_replaces_the_stale_pre_reset_action_label(self) -> None:
        """Regression guard: the frame retained at the candidate index must carry the
        RECOMPUTED action, not the one computed before the planner was reset.

        The post-execution collision path appends a normal frame for a step before
        discovering the executed action collided, so a frame already exists at
        candidate_index when recovery re-queries the (now-reset) planner. Using a
        planner that returns a different action on its second call -- regardless of
        the observation, which is identical either way -- makes a stale, unreplaced
        frame distinguishable from a correctly recomputed one.
        """
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=0.3)
        simulator.first_step_disturbance = 20.0  # only the very first .step() call collides
        planner = _ScriptedPlanner(unsafe_action=[1.0], safe_action=[0.5])
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
        # The recovered trajectory was built by executing [0.5] (the planner's
        # second-call action); the dataset must reflect that, not the first
        # call's [1.0], which never actually led anywhere in this episode.
        self.assertEqual(len(writer.frames), 1)
        self.assertEqual(writer.frames[0]["action"], [0.5])
        self.assertEqual(writer.episode_boundaries, [1])

    def test_backtrack_preserves_already_accumulated_done_hold_progress(self) -> None:
        """Regression guard: reset() zeroes the consecutive-done-steps hold counter,
        but the retained prefix up to the candidate already made real in-tolerance
        progress that must not be forgotten.

        Steps 1-2 reach and hold the goal region (accumulating a hold-count of 2
        out of done_hold_steps=3) before step 3's label is flagged unsafe and
        triggers backtracking to s_2 -- the same in-tolerance state. Recovery
        should need only one more in-tolerance step (2 + 1 = 3) to terminate, not
        a fresh three, so the total step count distinguishes a rebuilt counter
        from a silently reset one.
        """
        simulator = _FakeSimulator(collision_threshold=100.0, goal_threshold=1.0, done_hold_steps=3)
        planner = _SequencedPlanner(actions=[[1.0], [0.0], [200.0], [0.0]])
        writer = _FakeDatasetWriter()

        metrics = collect_dagger_rollouts(
            simulator=simulator,
            expert_planner=planner,
            dataset_writer=writer,
            trajectories_per_iteration=1,
            steps_per_trajectory=6,
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
        # With the hold-count correctly carried forward (2 already banked), only
        # one more in-tolerance step is needed to reach done_hold_steps=3, for a
        # total of candidate_index(2) + completion_steps(1) = 3 steps. Without the
        # fix, the counter silently resets to 0 and needs a fresh three more
        # steps (candidate_index(2) + completion_steps(3) = 5) instead.
        self.assertEqual(metrics.min_steps, 3)
        self.assertEqual(metrics.max_steps, 3)

    def test_backtrack_to_s0_does_not_count_the_uncounted_initial_state(self) -> None:
        """Regression guard: s_0 (the initial reset state) is never itself passed to
        should_terminate_rollout in the forward loop -- only s_1 onward, after the
        first transition. Replaying it during recovery would bank a hold-count
        that never happened, letting recovery report success one step early
        whenever the initial/candidate state already happens to be in tolerance.
        """
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=0.0, done_hold_steps=2)
        planner = _ScriptedPlanner(unsafe_action=[100.0], safe_action=[0.0])
        writer = _FakeDatasetWriter()

        metrics = collect_dagger_rollouts(
            simulator=simulator,
            expert_planner=planner,
            dataset_writer=writer,
            trajectories_per_iteration=1,
            steps_per_trajectory=3,
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
        # s_0=[0.0] already satisfies is_done() (goal_threshold=0.0), and the
        # unsafe first label backtracks straight to candidate_index=0 (the only
        # candidate when visited_states has length 1). A correct cold restart
        # from s_0 needs two genuine consecutive in-tolerance steps to reach
        # done_hold_steps=2. If s_0 were wrongly counted, one recovered step
        # would look sufficient instead.
        self.assertEqual(metrics.min_steps, 2)
        self.assertEqual(metrics.max_steps, 2)

    def test_initial_only_episode_still_randomizes_goal(self) -> None:
        """Regression guard: an explicit initial state without a paired goal must
        still honor randomize_goal_for_reset, matching what a fallback-sampled
        episode gets, instead of silently freezing at whatever goal the
        simulator started with for as many initial states as were provided.
        """
        simulator = _FakeSimulator(collision_threshold=100.0, goal_threshold=100.0)
        planner = _ConstantPlanner(action=[0.0])
        writer = _FakeDatasetWriter()
        self.assertEqual(simulator.goal.tolist(), [0.0])

        collect_dagger_rollouts(
            simulator=simulator,
            expert_planner=planner,
            dataset_writer=writer,
            trajectories_per_iteration=1,
            steps_per_trajectory=1,
            action_noise_std=0.0,
            action_noise_seed=0,
            initial_state_seed=0,
            expert_mixing_beta=1.0,
            policy_action_fn=None,
            frame_builder=_frame_builder,
            initial_states=[[0.0]],
            goal_states=None,
        )

        self.assertNotEqual(simulator.goal.tolist(), [0.0])

    def test_backtrack_recovery_correctly_accounts_recovered_steps_in_execution_mixing_stats(self) -> None:
        """Regression guard: expert-only recovery steps must be folded into the
        printed execution-mixing counters, and a discarded colliding step must not
        be counted at all.

        With expert_mixing_beta=0.0 every forward-loop step is policy-executed.
        The policy's action collides immediately, discarding that step's frame;
        backtrack recovery then completes the episode with a single, purely
        expert-driven step. The one frame actually saved is 100% expert-completed,
        so realized_expert_fraction must read 1.0. Without the fix, the discarded
        policy step's inline-incremented (expert=0, total=1) counts survive
        untouched -- since recovery never updates them -- reporting 0.0 instead.
        """
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=0.3)
        planner = _ConstantPlanner(action=[0.5])
        writer = _FakeDatasetWriter()

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            metrics = collect_dagger_rollouts(
                simulator=simulator,
                expert_planner=planner,
                dataset_writer=writer,
                trajectories_per_iteration=1,
                steps_per_trajectory=2,
                action_noise_std=0.0,
                action_noise_seed=0,
                initial_state_seed=0,
                expert_mixing_beta=0.0,
                policy_action_fn=lambda observation: np.array([100.0]),
                frame_builder=_frame_builder,
                initial_states=[[0.0]],
                goal_states=None,
            )

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(metrics.success_rate, 1.0)
        # Only the recovered, expert-only frame is saved -- the colliding
        # policy-executed step never reaches the dataset.
        self.assertEqual(len(writer.frames), 1)
        self.assertEqual(writer.frames[0]["action"], [0.5])

        output = stdout.getvalue()
        self.assertIn("realized_expert_fraction=1.000", output)
        self.assertIn("executed_steps=1", output)


class BacktrackRecoveryEscalationTests(unittest.TestCase):
    """Coverage for the gradual (escalating-beta) backtrack recovery path:
    beta_recovery < 1.0 mixes policy_action_fn in during recovery instead of
    jumping straight to expert-only control, escalating toward 1.0 (the
    original, still-default behavior) by beta_recovery_increment on each
    failed attempt at the same candidate.
    """

    def test_recovery_succeeds_at_beta_recovery_without_escalating(self) -> None:
        # expert_mixing_beta=1.0 keeps the forward loop from ever querying
        # the policy itself, so any policy call observed below can only have
        # come from recovery.
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        planner = _ScriptedPlanner(unsafe_action=[100.0], safe_action=[1.0])
        policy_calls: list[np.ndarray] = []

        def fake_policy(observation: np.ndarray) -> np.ndarray:
            policy_calls.append(np.asarray(observation).copy())
            return np.array([1.0])

        writer = _FakeDatasetWriter()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
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
                policy_action_fn=fake_policy,
                frame_builder=_frame_builder,
                initial_states=[[0.0]],
                goal_states=None,
                beta_recovery=0.0,
                beta_recovery_increment=1.0,
            )

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(metrics.success_rate, 1.0)
        # beta_recovery=0.0 deterministically picks the policy every retry
        # (Generator.random() never returns a negative value), so recovery
        # must have actually invoked it rather than falling back to expert.
        self.assertGreaterEqual(len(policy_calls), 1)
        self.assertEqual(len(writer.frames), 1)
        # The label is always the expert's own action regardless of who
        # executed, per collect_dagger_rollouts' DAgger convention.
        self.assertEqual(writer.frames[0]["action"], [1.0])
        self.assertIn("Recovery from s_0 succeeded at beta_recovery=0.000", stdout.getvalue())
        # The one recovered step was policy-executed (beta=0.0), so the
        # per-recovery expert/policy breakdown printed alongside the final
        # "Backtracked to state" line should read 0% expert, 0/1 steps.
        # candidate_index=0 means there's no pre-backtrack portion at all, so
        # the full-episode split matches the recovered segment exactly here.
        self.assertIn(
            "(recovered segment: 0.0% expert, 0/1 steps; "
            "full episode: 0.0% expert, 0/1 steps)",
            stdout.getvalue(),
        )

    def test_recovery_escalates_to_beta_one_after_a_failed_lower_beta_attempt(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        planner = _ScriptedPlanner(unsafe_action=[100.0], safe_action=[1.0])

        def unsafe_policy(observation: np.ndarray) -> np.ndarray:
            return np.array([100.0])

        writer = _FakeDatasetWriter()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
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
                policy_action_fn=unsafe_policy,
                frame_builder=_frame_builder,
                initial_states=[[0.0]],
                goal_states=None,
                beta_recovery=0.0,
                beta_recovery_increment=1.0,
            )

        # The beta=0.0 attempt deterministically executes the policy's
        # unsafe [100.0] and fails; escalating to beta=1.0 (pure expert,
        # safe_action=[1.0]) must still recover the episode.
        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(len(writer.frames), 1)
        self.assertEqual(writer.frames[0]["action"], [1.0])

        output = stdout.getvalue()
        self.assertIn("Recovery attempt 1 from s_0 failed at beta_recovery=0.000", output)
        self.assertIn("retrying with beta_recovery=1.000", output)
        # beta=1.0 short-circuits to the original deterministic path, which
        # never prints a "succeeded" line (see backtrack_and_complete's own
        # Phase 2).
        self.assertNotIn("Recovery from s_0 succeeded", output)
        # The kept (restored) segment is Phase 1's pure-expert completion,
        # not the failed policy-mixed attempt -- 100% expert.
        self.assertIn(
            "(recovered segment: 100.0% expert, 1/1 steps; "
            "full episode: 100.0% expert, 1/1 steps)",
            output,
        )

    def test_full_episode_split_includes_the_pre_backtrack_forward_pass(self) -> None:
        """Regression guard: 'full episode' must combine the pre-backtrack
        forward-pass portion (executed under the round's own
        expert_mixing_beta) with the recovered segment (under beta_recovery)
        -- not just repeat the recovered segment's own number.
        """
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        # Step 1's label [0.5] is safe, so it executes normally (via the
        # policy, since expert_mixing_beta=0.0 below) with no backtrack.
        # Step 2's label [100.0] is unsafe and triggers backtrack back to
        # s_1; from there the (pure-expert, beta_recovery omitted) recovery
        # uses [1.0] to reach the goal.
        planner = _SequencedPlanner(actions=[[0.5], [100.0], [1.0]])

        def fake_policy(observation: np.ndarray) -> np.ndarray:
            return np.array([0.3])

        writer = _FakeDatasetWriter()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            metrics = collect_dagger_rollouts(
                simulator=simulator,
                expert_planner=planner,
                dataset_writer=writer,
                trajectories_per_iteration=1,
                steps_per_trajectory=5,
                action_noise_std=0.0,
                action_noise_seed=0,
                initial_state_seed=0,
                expert_mixing_beta=0.0,
                policy_action_fn=fake_policy,
                frame_builder=_frame_builder,
                initial_states=[[0.0]],
                goal_states=None,
            )

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(metrics.success_rate, 1.0)
        # Step 1 (policy-executed, [0.3]) survives untouched; the recovered
        # step used the expert's [1.0] label.
        self.assertEqual(len(writer.frames), 2)
        self.assertEqual(writer.frames[0]["action"], [0.5])
        self.assertEqual(writer.frames[1]["action"], [1.0])
        # Recovered segment alone is 1/1 expert (100%); the full episode
        # also counts step 1's policy execution, landing at 1/2 (50%).
        self.assertIn(
            "(recovered segment: 100.0% expert, 1/1 steps; "
            "full episode: 50.0% expert, 1/2 steps)",
            stdout.getvalue(),
        )

    def test_default_beta_recovery_calls_policy_zero_times_during_recovery(self) -> None:
        """Regression guard: omitting beta_recovery/beta_recovery_increment
        (the pre-existing call signature) must reproduce the original
        expert-only recovery exactly -- no RNG draw, no policy_action_fn call.
        """
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        planner = _ScriptedPlanner(unsafe_action=[100.0], safe_action=[1.0])
        policy_calls: list[np.ndarray] = []

        def fake_policy(observation: np.ndarray) -> np.ndarray:
            policy_calls.append(np.asarray(observation).copy())
            return np.array([1.0])

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
            policy_action_fn=fake_policy,
            frame_builder=_frame_builder,
            initial_states=[[0.0]],
            goal_states=None,
            # beta_recovery / beta_recovery_increment intentionally omitted.
        )

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(policy_calls, [])
        self.assertEqual(writer.frames[0]["action"], [1.0])


class EvaluatePolicyRolloutsTorchSeedingTests(unittest.TestCase):
    """A flow/safeflow policy's inference draws from an unseeded torch.randn
    for its ODE initial noise, so without pinning torch's RNG per episode,
    the exact same checkpoint on the exact same scenario can succeed or fail
    from pure sampling luck -- evaluate_policy_rollouts must call
    torch.manual_seed with a value derived only from that episode's own seed
    spec, so re-evaluating the same scenario reproduces the same draw.
    """

    def test_manual_seed_is_called_once_per_episode_with_the_derived_seed(self) -> None:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1e9)
        with mock.patch("learning.dagger.rollouts.torch.manual_seed") as manual_seed:
            evaluate_policy_rollouts(
                simulator=simulator,
                num_episodes=3,
                num_steps=2,
                seed_start=100,
                action_fn=lambda observation: np.array([0.0]),
            )
        # _FakeSimulator has num_robots=1, so evaluation_seed_specs yields
        # plain ints seed_start, seed_start+1, ... ; the torch seed is the
        # *derived* (stream-ID-mixed) value, not that raw int -- see
        # torch_inference_seed_for_rollout.
        manual_seed.assert_has_calls([
            mock.call(torch_inference_seed_for_rollout(0, seed_spec=100)),
            mock.call(torch_inference_seed_for_rollout(0, seed_spec=101)),
            mock.call(torch_inference_seed_for_rollout(0, seed_spec=102)),
        ])

    def test_torch_seed_is_decorrelated_from_the_action_noise_seed(self) -> None:
        """Regression guard for the exact bug this stream ID exists to
        prevent: both seeds are derived from the same seed_spec, but must
        not collide or trivially track each other.
        """
        seed_spec = 100
        torch_seed = torch_inference_seed_for_rollout(0, seed_spec=seed_spec)
        noise_seed = action_noise_seed_for_rollout(0, seed_spec=seed_spec)
        self.assertNotEqual(torch_seed, noise_seed)

    def test_reevaluating_the_same_scenario_reproduces_the_same_seed(self) -> None:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1e9)
        seeds_seen: list[list[int]] = [[], []]
        for run_idx in range(2):
            with mock.patch("learning.dagger.rollouts.torch.manual_seed", side_effect=seeds_seen[run_idx].append):
                evaluate_policy_rollouts(
                    simulator=simulator,
                    num_episodes=2,
                    num_steps=2,
                    seed_start=7,
                    action_fn=lambda observation: np.array([0.0]),
                )
        self.assertEqual(seeds_seen[0], seeds_seen[1])


class RestartInitialStateRoundTests(unittest.TestCase):
    """Coverage for restart_initial_state_round: it must make initial-state/goal
    sampling reproducible across rounds that key off the same seed, without
    changing anything when left at its default.
    """

    @staticmethod
    def _sample_via_collect(*, round_index: int, restart_initial_state_round: bool) -> tuple[list[float], list[float]]:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1e9)
        planner = _ConstantPlanner(action=[0.0])
        writer = _FakeDatasetWriter()
        collect_dagger_rollouts(
            simulator=simulator,
            expert_planner=planner,
            dataset_writer=writer,
            trajectories_per_iteration=1,
            steps_per_trajectory=1,
            action_noise_std=0.0,
            action_noise_seed=0,
            initial_state_seed=7,
            expert_mixing_beta=1.0,
            policy_action_fn=None,
            frame_builder=_frame_builder,
            initial_states=None,
            goal_states=None,
            round_index=round_index,
            restart_initial_state_round=restart_initial_state_round,
        )
        # action=[0.0] never moves the state, so simulator.state is still exactly
        # the sampled initial state after the one recorded step.
        return simulator.goal.tolist(), simulator.state.tolist()

    def test_restart_true_makes_rounds_sharing_a_seed_sample_identically(self) -> None:
        round0 = self._sample_via_collect(round_index=0, restart_initial_state_round=True)
        round1 = self._sample_via_collect(round_index=1, restart_initial_state_round=True)
        self.assertEqual(round0, round1)

    def test_restart_false_preserves_todays_behavior_of_varying_by_round(self) -> None:
        round0 = self._sample_via_collect(round_index=0, restart_initial_state_round=False)
        round1 = self._sample_via_collect(round_index=1, restart_initial_state_round=False)
        self.assertNotEqual(round0, round1)


class EvaluatePolicyRolloutsSourceSplitTests(unittest.TestCase):
    """evaluate_policy_rollouts must report config- and random-sourced episode
    outcomes separately (DaggerEvalMetrics.config_*/random_*), not just the
    pooled success_rate -- the curriculum's explicit initial_states/goal_states
    are the leading `usable_count` episodes, everything after that falls back
    to simulator RNG sampling (see its own usable_count logic).
    """

    def test_config_and_random_successes_are_counted_separately(self) -> None:
        # goal_threshold=1000.0 with a constant zero action means:
        # - a config-sourced episode starting exactly at the threshold
        #   terminates instantly (reached_goal=True, 0 steps) every time.
        # - a random-sourced episode starts uniformly in [-1, 1] (see
        #   _FakeSimulator.random_initial_state) and a zero action never
        #   moves it, so it can never reach 1000.0 -- always a timeout.
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1000.0)
        metrics = evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=5,
            num_steps=3,
            seed_start=0,
            action_fn=lambda observation: np.array([0.0]),
            initial_states=[[1000.0], [1000.0]],
            goal_states=[[0.0], [0.0]],
        )
        assert metrics is not None
        self.assertEqual(metrics.num_episodes, 5)
        self.assertEqual(metrics.config_num_episodes, 2)
        self.assertEqual(metrics.config_successes, 2)
        self.assertEqual(metrics.random_num_episodes, 3)
        self.assertEqual(metrics.random_successes, 0)
        # Pooled rate still reflects both groups combined.
        self.assertAlmostEqual(metrics.success_rate, 2.0 / 5.0)

    def test_no_explicit_states_leaves_config_bucket_empty(self) -> None:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1000.0)
        metrics = evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=3,
            num_steps=2,
            seed_start=0,
            action_fn=lambda observation: np.array([0.0]),
        )
        assert metrics is not None
        self.assertEqual(metrics.config_num_episodes, 0)
        self.assertEqual(metrics.config_successes, 0)
        self.assertEqual(metrics.random_num_episodes, 3)


class EvaluatePolicyRolloutsFailureBreakdownTests(unittest.TestCase):
    """evaluate_policy_rollouts must report *why* a failed episode failed --
    collision, timeout, or solve_failure -- not just a pooled success_rate.
    A policy that times out while actively avoiding is a different (and
    better) outcome than one that collides outright, which success_rate
    alone can't distinguish.
    """

    def test_collision_timeout_and_solve_failure_are_counted_separately(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1000.0)

        def action_fn(observation: np.ndarray) -> np.ndarray:
            # Sentinel initial state [5.0] marks the solve_failure episode;
            # everything else is a plain no-op action (never moves, so it
            # can only time out or start already past the collision
            # threshold).
            if float(np.asarray(observation)[0]) == 5.0:
                raise PlannerSolveError("forced failure for test")
            return np.array([0.0])

        metrics = evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=3,
            num_steps=3,
            seed_start=0,
            action_fn=action_fn,
            # goal_threshold=1000.0 means none of these ever reach the goal;
            # no goal_states, so all 3 are sourced from this initial_states
            # list, indexed 0/1/2 in order.
            initial_states=[[15.0], [0.0], [5.0]],
        )

        assert metrics is not None
        self.assertEqual(metrics.success_rate, 0.0)
        self.assertEqual(metrics.collision_failures, 1)
        self.assertEqual(metrics.timeout_failures, 1)
        self.assertEqual(metrics.solve_failures, 1)

    def test_successful_episodes_are_not_counted_as_any_failure(self) -> None:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=0.0)
        metrics = evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=2,
            num_steps=3,
            seed_start=0,
            action_fn=lambda observation: np.array([0.0]),
            initial_states=[[0.0], [0.0]],
        )
        assert metrics is not None
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(metrics.collision_failures, 0)
        self.assertEqual(metrics.timeout_failures, 0)
        self.assertEqual(metrics.solve_failures, 0)


if __name__ == "__main__":
    unittest.main()
