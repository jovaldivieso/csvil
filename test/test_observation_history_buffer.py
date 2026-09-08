from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.factory import DynamicsFactory
from learning.dagger import ObservationHistoryBuffer
from learning.data_utils import (
    _bulk_future_samples,
    _bulk_past_samples,
    _HISTORY_STACKED_FIELDS,
    build_action_window_cache,
    build_observation_history_cache,
    collate_batch_for_policy,
    format_sample_for_policy,
)


def _make_frame(step: int) -> dict[str, np.ndarray]:
    """A distinguishable synthetic observation frame for a given rollout step."""
    return {
        "observation.environment_state": np.array([100.0 + step], dtype=np.float32),
        "observation.state": np.array([200.0 + step, 201.0 + step], dtype=np.float32),
        "observation.state_mask": np.array([1.0], dtype=np.float32),
        "observation.neighbor_state": np.array([300.0 + step, 301.0 + step], dtype=np.float32),
        "observation.neighbor_mask": np.array([1.0], dtype=np.float32),
    }


def _make_dataset_row(index: int, episode_index: int) -> dict[str, object]:
    """A distinguishable synthetic dataset row for a given absolute dataset index."""
    return {
        "index": index,
        "episode_index": episode_index,
        "observation.environment_state": np.array([100.0 + index], dtype=np.float32),
        "observation.state": np.array([200.0 + index, 201.0 + index], dtype=np.float32),
        "observation.state_mask": np.array([1.0], dtype=np.float32),
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


class ObservationHistoryBufferTests(unittest.TestCase):
    def test_warm_up_padding_zero_fills_missing_frames(self) -> None:
        buffer = ObservationHistoryBuffer(observation_horizon=3, num_robots=1)
        frame0 = _make_frame(0)

        stacked = buffer.append_and_stack(0, frame0)

        np.testing.assert_array_equal(
            stacked["observation.neighbor_state"],
            np.concatenate(
                [np.zeros_like(frame0["observation.neighbor_state"])] * 2
                + [frame0["observation.neighbor_state"]]
            ),
        )
        np.testing.assert_array_equal(
            stacked["observation.neighbor_mask"],
            np.concatenate(
                [np.zeros_like(frame0["observation.neighbor_mask"])] * 2
                + [frame0["observation.neighbor_mask"]]
            ),
        )
        np.testing.assert_array_equal(
            stacked["observation.environment_state"], frame0["observation.environment_state"]
        )
        np.testing.assert_array_equal(
            stacked["observation.state"],
            np.concatenate([np.zeros_like(frame0["observation.state"])] * 2 + [frame0["observation.state"]]),
        )

    def test_rolling_window_keeps_last_horizon_frames_oldest_to_newest(self) -> None:
        horizon = 3
        buffer = ObservationHistoryBuffer(observation_horizon=horizon, num_robots=1)
        frames = [_make_frame(i) for i in range(horizon + 2)]

        stacked = None
        for frame in frames:
            stacked = buffer.append_and_stack(0, frame)

        expected_window = frames[-horizon:]
        np.testing.assert_array_equal(
            stacked["observation.neighbor_state"],
            np.concatenate([f["observation.neighbor_state"] for f in expected_window]),
        )
        np.testing.assert_array_equal(
            stacked["observation.neighbor_mask"],
            np.concatenate([f["observation.neighbor_mask"] for f in expected_window]),
        )
        np.testing.assert_array_equal(
            stacked["observation.environment_state"], frames[-1]["observation.environment_state"]
        )
        np.testing.assert_array_equal(
            stacked["observation.state"],
            np.concatenate([f["observation.state"] for f in expected_window]),
        )

    def test_reset_clears_history_and_warm_up_zero_fills_again(self) -> None:
        horizon = 3
        buffer = ObservationHistoryBuffer(observation_horizon=horizon, num_robots=1)
        for i in range(horizon + 2):
            buffer.append_and_stack(0, _make_frame(i))

        buffer.reset()
        new_frame = _make_frame(999)
        stacked = buffer.append_and_stack(0, new_frame)

        np.testing.assert_array_equal(
            stacked["observation.neighbor_state"],
            np.concatenate(
                [np.zeros_like(new_frame["observation.neighbor_state"])] * (horizon - 1)
                + [new_frame["observation.neighbor_state"]]
            ),
        )
        np.testing.assert_array_equal(
            stacked["observation.neighbor_mask"],
            np.concatenate(
                [np.zeros_like(new_frame["observation.neighbor_mask"])] * (horizon - 1)
                + [new_frame["observation.neighbor_mask"]]
            ),
        )
        np.testing.assert_array_equal(
            stacked["observation.state"],
            np.concatenate(
                [np.zeros_like(new_frame["observation.state"])] * (horizon - 1) + [new_frame["observation.state"]]
            ),
        )

    def test_online_stack_matches_offline_format_sample_for_policy(self) -> None:
        simulator = _build_multi_robot_simulator()
        horizon = 3
        buffer = ObservationHistoryBuffer(observation_horizon=horizon, num_robots=1)
        per_robot_action_dim = simulator.nu // simulator.num_robots

        past_samples: list[dict[str, np.ndarray]] = []
        for step in range(horizon + 2):
            frame = _make_frame(step)
            sample = dict(frame)
            sample["action"] = (np.arange(per_robot_action_dim, dtype=np.float32) + step)

            online_stacked = buffer.append_and_stack(0, frame)
            offline_obs, _ = format_sample_for_policy(
                sample=sample,
                simulator=simulator,
                observation_horizon=horizon,
                past_samples=list(past_samples),
            )

            for field_name, online_value in online_stacked.items():
                np.testing.assert_allclose(
                    online_value,
                    offline_obs[field_name].numpy(),
                    err_msg=f"online/offline mismatch in '{field_name}' at step {step}",
                )

            past_samples.append(sample)


class CollateBatchDatasetHistoryTests(unittest.TestCase):
    """Regression coverage for the dataset-backed history path (_bulk_past_samples).

    The online/offline equivalence test above passes ``past_samples`` directly,
    so it never exercises the dataset lookup, episode-boundary filtering, or
    bulk-fetch logic in _bulk_past_samples/_fetch_samples. This drives that
    path through the real collate_batch_for_policy entry point instead.
    """

    def test_history_never_crosses_episode_boundary_and_zero_pads_missing_leading_frames(self) -> None:
        simulator = _build_multi_robot_simulator()
        horizon = 3
        # Episode 0: absolute indices 0, 1, 2. Episode 1: absolute indices 3, 4, 5.
        dataset = _ScalarIndexOnlyDataset(
            [_make_dataset_row(i, episode_index=0) for i in range(3)]
            + [_make_dataset_row(i, episode_index=1) for i in range(3, 6)]
        )
        # index 3 is episode 1's first frame (no real predecessor at all);
        # index 4 is its second frame (exactly one real predecessor, index 3).
        batch = [dataset[3], dataset[4]]

        observations, _ = collate_batch_for_policy(
            batch=batch,
            simulator=simulator,
            observation_horizon=horizon,
            dataset=dataset,
        )

        neighbor_state = observations["observation.neighbor_state"]
        neighbor_mask = observations["observation.neighbor_mask"]
        self.assertEqual(tuple(neighbor_state.shape), (2, horizon * 2))
        self.assertEqual(tuple(neighbor_mask.shape), (2, horizon))

        # Row 0 (index 3): candidate history indices 1 and 2 both belong to
        # episode 0, so neither qualifies -- the whole window is zero-padded.
        np.testing.assert_allclose(neighbor_state[0].numpy(), [0.0, 0.0, 0.0, 0.0, 303.0, 304.0])
        np.testing.assert_allclose(neighbor_mask[0].numpy(), [0.0, 0.0, 1.0])

        # Row 1 (index 4): candidate index 2 belongs to episode 0 (excluded),
        # candidate index 3 belongs to episode 1 (kept) -- one padded slot
        # followed by two real, correctly-ordered frames.
        np.testing.assert_allclose(neighbor_state[1].numpy(), [0.0, 0.0, 303.0, 304.0, 304.0, 305.0])
        np.testing.assert_allclose(neighbor_mask[1].numpy(), [0.0, 1.0, 1.0])

        # environment_state is never history-stacked -- always the current
        # (most recent) frame, regardless of how much history exists.
        np.testing.assert_allclose(observations["observation.environment_state"][0].numpy(), [103.0])
        np.testing.assert_allclose(observations["observation.environment_state"][1].numpy(), [104.0])

        # state IS history-stacked (same episode-boundary/zero-pad rules as
        # neighbor_state above): row 0's candidates 1,2 are both excluded
        # (episode 0), row 1's candidate 2 is excluded but 3 is kept.
        np.testing.assert_allclose(observations["observation.state"][0].numpy(), [0.0, 0.0, 0.0, 0.0, 203.0, 204.0])
        np.testing.assert_allclose(observations["observation.state"][1].numpy(), [0.0, 0.0, 203.0, 204.0, 204.0, 205.0])


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


class ObservationHistoryCacheTests(unittest.TestCase):
    """Coverage for build_observation_history_cache, the once-per-round replacement
    for _bulk_past_samples' every-batch, every-epoch past-frame refetching.
    """

    def test_matches_uncached_bulk_past_samples_path(self) -> None:
        simulator = _build_multi_robot_simulator()
        dataset = _episodes_dataset()
        for observation_horizon in (1, 2, 5):
            cache = build_observation_history_cache(dataset, observation_horizon)
            for name in _HISTORY_STACKED_FIELDS:
                field_dim = dataset[0][name].shape[0]
                self.assertEqual(tuple(cache[name].shape), (10, observation_horizon * field_dim))
            for index in range(10):
                sample = dataset[index]
                past_samples = _bulk_past_samples(
                    batch=[sample], dataset=dataset, observation_horizon=observation_horizon,
                )[0]
                uncached_obs, _ = format_sample_for_policy(
                    sample=sample,
                    simulator=simulator,
                    observation_horizon=observation_horizon,
                    past_samples=past_samples,
                )
                for name in _HISTORY_STACKED_FIELDS:
                    np.testing.assert_allclose(
                        cache[name][index].numpy(),
                        uncached_obs[name].numpy(),
                        err_msg=f"cache mismatch in '{name}' at index={index}, observation_horizon={observation_horizon}",
                    )

    def test_episode_boundary_zero_pads_without_crossing_into_previous_episode(self) -> None:
        dataset = _episodes_dataset()
        cache = build_observation_history_cache(dataset, observation_horizon=4)

        # Index 4 (episode 0's last frame): window [1,2,3,4], all within episode 0 -- no padding.
        expected_neighbor_state_4 = np.concatenate([dataset[i]["observation.neighbor_state"] for i in (1, 2, 3, 4)])
        np.testing.assert_allclose(cache["observation.neighbor_state"][4].numpy(), expected_neighbor_state_4)
        np.testing.assert_allclose(cache["observation.neighbor_mask"][4].numpy(), [1.0, 1.0, 1.0, 1.0])

        # Index 5 (episode 1's first frame): window [2,3,4,5] all belong to episode 0
        # except position 5 itself -- must ZERO-pad, never fall back to a repeat.
        expected_neighbor_state_5 = np.concatenate(
            [np.zeros(2, dtype=np.float32)] * 3 + [dataset[5]["observation.neighbor_state"]]
        )
        np.testing.assert_allclose(cache["observation.neighbor_state"][5].numpy(), expected_neighbor_state_5)
        np.testing.assert_allclose(cache["observation.neighbor_mask"][5].numpy(), [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            cache["observation.state"][5].numpy(),
            np.concatenate([np.zeros(2, dtype=np.float32)] * 3 + [dataset[5]["observation.state"]]),
        )
        np.testing.assert_allclose(cache["observation.state_mask"][5].numpy(), [0.0, 0.0, 0.0, 1.0])

        # Index 6 (episode 1's second frame): window [3,4,5,6] -- positions 3,4 belong
        # to episode 0 (zero), 5,6 are real episode-1 frames.
        expected_neighbor_state_6 = np.concatenate(
            [np.zeros(2, dtype=np.float32)] * 2
            + [dataset[5]["observation.neighbor_state"], dataset[6]["observation.neighbor_state"]]
        )
        np.testing.assert_allclose(cache["observation.neighbor_state"][6].numpy(), expected_neighbor_state_6)
        np.testing.assert_allclose(cache["observation.neighbor_mask"][6].numpy(), [0.0, 0.0, 1.0, 1.0])

        # Index 0 (dataset's very first frame): window [-3,-2,-1,0] -- everything
        # before position 0 is invalid (dataset start coincides with episode start
        # here), only the current frame itself is real.
        expected_neighbor_state_0 = np.concatenate(
            [np.zeros(2, dtype=np.float32)] * 3 + [dataset[0]["observation.neighbor_state"]]
        )
        np.testing.assert_allclose(cache["observation.neighbor_state"][0].numpy(), expected_neighbor_state_0)
        np.testing.assert_allclose(cache["observation.neighbor_mask"][0].numpy(), [0.0, 0.0, 0.0, 1.0])

    def test_scalar_index_only_dataset_supported(self) -> None:
        # build_observation_history_cache only ever calls dataset[int]; it must
        # work against a backend that rejects list-style bulk indexing, exactly
        # like a real LeRobotDataset does (see _ScalarIndexOnlyDataset above).
        dataset = _episodes_dataset()
        cache = build_observation_history_cache(dataset, observation_horizon=3)
        for name in _HISTORY_STACKED_FIELDS:
            field_dim = dataset[0][name].shape[0]
            self.assertEqual(tuple(cache[name].shape), (10, 3 * field_dim))

    def test_observation_horizon_one_is_just_the_current_frame(self) -> None:
        dataset = _episodes_dataset()
        cache = build_observation_history_cache(dataset, observation_horizon=1)
        for index in range(10):
            for name in _HISTORY_STACKED_FIELDS:
                np.testing.assert_allclose(cache[name][index].numpy(), dataset[index][name])

    def test_collate_batch_for_policy_uses_cache_when_given(self) -> None:
        simulator = _build_multi_robot_simulator()
        dataset = _episodes_dataset()
        observation_horizon = 4
        cache = build_observation_history_cache(dataset, observation_horizon)
        batch = [dataset[4], dataset[5]]

        observations, _ = collate_batch_for_policy(
            batch=batch,
            simulator=simulator,
            observation_horizon=observation_horizon,
            dataset=dataset,
            observation_history_cache=cache,
        )
        for name in _HISTORY_STACKED_FIELDS:
            np.testing.assert_allclose(observations[name][0].numpy(), cache[name][4].numpy())
            np.testing.assert_allclose(observations[name][1].numpy(), cache[name][5].numpy())

    def test_non_positive_observation_horizon_rejected(self) -> None:
        # format_sample_for_policy already rejects this; the cache must too,
        # rather than silently returning a cache shaped for horizon=1
        # regardless of what its caller actually asked for.
        dataset = _episodes_dataset()
        for observation_horizon in (0, -1):
            with self.assertRaises(ValueError):
                build_observation_history_cache(dataset, observation_horizon)


if __name__ == "__main__":
    unittest.main()
