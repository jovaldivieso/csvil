from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.factory import DynamicsFactory
from learning.data_utils import (
    _bulk_future_samples,
    build_action_window_cache,
    collate_batch_for_policy,
    format_sample_for_policy,
)


def _make_dataset_row(index: int, episode_index: int) -> dict[str, object]:
    """A distinguishable synthetic dataset row for a given absolute dataset index."""
    return {
        "index": index,
        "episode_index": episode_index,
        "observation.environment_state": np.array([100.0 + index], dtype=np.float32),
        "observation.state": np.array([200.0 + index, 201.0 + index], dtype=np.float32),
        "observation.neighbor_state": np.array([300.0 + index, 301.0 + index], dtype=np.float32),
        "observation.neighbor_mask": np.array([1.0], dtype=np.float32),
        "action": np.array([index, index + 0.5], dtype=np.float32),
    }


class _ScalarIndexOnlyDataset:
    """Stand-in for a dataset backend that supports only single-row integer indexing.

    Exercises _fetch_samples' fallback branch (some LeRobotDataset backends
    reject list-style bulk indexing), not just the bulk/columnar path.
    """

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: object) -> dict[str, object]:
        if not isinstance(index, int):
            raise TypeError("this dataset only supports scalar integer indexing")
        return self._rows[index]


def _build_multi_robot_simulator():
    return DynamicsFactory.create(
        system_name="multi_robot",
        config={
            "dt": 0.05,
            "d_safe": 0.1,
            "robots": [
                {
                    "system": "double_integrator",
                    "config": {"dt": 0.05, "goal": [0.0, 0.0], "randomize_goal": False},
                },
                {
                    "system": "double_integrator",
                    "config": {"dt": 0.05, "goal": [2.0, -1.0], "randomize_goal": False},
                },
            ],
        },
    )


def _episodes_dataset() -> _ScalarIndexOnlyDataset:
    # Episode 0: absolute indices 0-4 (5 frames). Episode 1: absolute indices 5-9 (5 frames).
    rows = [_make_dataset_row(i, episode_index=0) for i in range(5)] + [
        _make_dataset_row(i, episode_index=1) for i in range(5, 10)
    ]
    return _ScalarIndexOnlyDataset(rows)


class ActionWindowCacheTests(unittest.TestCase):
    """Coverage for build_action_window_cache, the once-per-round replacement for
    _bulk_future_samples' every-batch, every-epoch future-frame refetching.
    """

    def test_matches_uncached_bulk_future_samples_path(self) -> None:
        simulator = _build_multi_robot_simulator()
        dataset = _episodes_dataset()
        for prediction_horizon in (1, 5, 40):
            cache = build_action_window_cache(dataset, prediction_horizon)
            self.assertEqual(tuple(cache.shape), (10, prediction_horizon, 2))
            for index in range(10):
                sample = dataset[index]
                subsequent_samples = _bulk_future_samples(
                    batch=[sample], dataset=dataset, prediction_horizon=prediction_horizon,
                )[0]
                _, uncached_actions = format_sample_for_policy(
                    sample=sample,
                    simulator=simulator,
                    prediction_horizon=prediction_horizon,
                    subsequent_samples=subsequent_samples,
                )
                np.testing.assert_allclose(
                    cache[index].numpy(),
                    uncached_actions.numpy(),
                    err_msg=f"cache mismatch at index={index}, prediction_horizon={prediction_horizon}",
                )

    def test_episode_boundary_repeats_last_action_without_crossing_into_next_episode(self) -> None:
        dataset = _episodes_dataset()
        cache = build_action_window_cache(dataset, prediction_horizon=4)

        # Index 3 (episode 0): real actions at 3, 4, then the episode ends --
        # the remaining slots repeat action(4), episode 0's last real action.
        expected_row3 = np.stack(
            [dataset[3]["action"], dataset[4]["action"], dataset[4]["action"], dataset[4]["action"]]
        )
        np.testing.assert_allclose(cache[3].numpy(), expected_row3)

        # Index 4 (episode 0's last frame): repeats its own action for the whole window.
        expected_row4 = np.stack([dataset[4]["action"]] * 4)
        np.testing.assert_allclose(cache[4].numpy(), expected_row4)

        # Index 5 (episode 1's first frame): must NOT pull in episode 0's action(4)
        # despite being dataset-adjacent to it.
        expected_row5 = np.stack(
            [dataset[5]["action"], dataset[6]["action"], dataset[7]["action"], dataset[8]["action"]]
        )
        np.testing.assert_allclose(cache[5].numpy(), expected_row5)

    def test_scalar_index_only_dataset_supported(self) -> None:
        # build_action_window_cache only ever calls dataset[int]; it must work
        # against a backend that rejects list-style bulk indexing, exactly like a
        # real LeRobotDataset does (see _ScalarIndexOnlyDataset above).
        dataset = _episodes_dataset()
        cache = build_action_window_cache(dataset, prediction_horizon=3)
        self.assertEqual(tuple(cache.shape), (10, 3, 2))

    def test_multi_robot_decentralized_action_dim_stays_local_per_robot(self) -> None:
        simulator = _build_multi_robot_simulator()
        per_robot_action_dim = simulator.nu // simulator.num_robots
        dataset = _episodes_dataset()
        cache = build_action_window_cache(dataset, prediction_horizon=3)
        self.assertEqual(cache.shape[-1], per_robot_action_dim)

    def test_collate_batch_for_policy_uses_cache_when_given(self) -> None:
        simulator = _build_multi_robot_simulator()
        dataset = _episodes_dataset()
        prediction_horizon = 4
        cache = build_action_window_cache(dataset, prediction_horizon)
        batch = [dataset[3], dataset[5]]

        _, actions = collate_batch_for_policy(
            batch=batch,
            simulator=simulator,
            prediction_horizon=prediction_horizon,
            dataset=dataset,
            action_window_cache=cache,
        )
        np.testing.assert_allclose(actions[0].numpy(), cache[3].numpy())
        np.testing.assert_allclose(actions[1].numpy(), cache[5].numpy())

    def test_non_positive_prediction_horizon_rejected(self) -> None:
        # format_sample_for_policy already rejects this; the cache must too,
        # rather than silently returning a one-step cache that violates the
        # horizon its own caller asked for.
        dataset = _episodes_dataset()
        for prediction_horizon in (0, -1):
            with self.assertRaises(ValueError):
                build_action_window_cache(dataset, prediction_horizon)


if __name__ == "__main__":
    unittest.main()
