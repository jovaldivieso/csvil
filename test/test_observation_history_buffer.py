from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.factory import DynamicsFactory
from learning.dagger import ObservationHistoryBuffer
from learning.data_utils import collate_batch_for_policy, format_sample_for_policy


def _make_frame(step: int) -> dict[str, np.ndarray]:
    """A distinguishable synthetic observation frame for a given rollout step."""
    return {
        "observation.environment_state": np.array([100.0 + step], dtype=np.float32),
        "observation.state": np.array([200.0 + step, 201.0 + step], dtype=np.float32),
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
        np.testing.assert_array_equal(stacked["observation.state"], frame0["observation.state"])

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
        np.testing.assert_array_equal(stacked["observation.state"], frames[-1]["observation.state"])

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

        # environment_state/state are never history-stacked -- always the
        # current (most recent) frame, regardless of how much history exists.
        np.testing.assert_allclose(observations["observation.environment_state"][0].numpy(), [103.0])
        np.testing.assert_allclose(observations["observation.environment_state"][1].numpy(), [104.0])
        np.testing.assert_allclose(observations["observation.state"][0].numpy(), [203.0, 204.0])
        np.testing.assert_allclose(observations["observation.state"][1].numpy(), [204.0, 205.0])


if __name__ == "__main__":
    unittest.main()
