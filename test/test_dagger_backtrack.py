from __future__ import annotations

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.dagger.rollouts import MAX_BACKTRACK_CANDIDATES, _geometric_backtrack_indices
from systems.seed_utils import initial_state_seed_for_rollout


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

    def test_round_seeds_are_distinct_and_reproducible(self) -> None:
        first_round_seed = initial_state_seed_for_rollout(0, rollout_index=1, round_index=0)
        second_round_seed = initial_state_seed_for_rollout(0, rollout_index=1, round_index=1)

        self.assertNotEqual(first_round_seed, second_round_seed)
        self.assertEqual(
            first_round_seed,
            initial_state_seed_for_rollout(0, rollout_index=1, round_index=0),
        )


if __name__ == "__main__":
    unittest.main()
