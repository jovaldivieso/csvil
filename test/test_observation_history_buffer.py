from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.factory import DynamicsFactory
from learning.dagger import ObservationHistoryBuffer
from learning.data_utils import format_sample_for_policy


def _make_frame(step: int) -> dict[str, np.ndarray]:
    """A distinguishable synthetic observation frame for a given rollout step."""
    return {
        "observation.environment_state": np.array([100.0 + step], dtype=np.float32),
        "observation.state": np.array([200.0 + step, 201.0 + step], dtype=np.float32),
        "observation.neighbor_state": np.array([300.0 + step, 301.0 + step], dtype=np.float32),
        "observation.neighbor_mask": np.array([1.0], dtype=np.float32),
    }


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
    def test_warm_up_padding_repeats_earliest_frame(self) -> None:
        buffer = ObservationHistoryBuffer(observation_horizon=3, num_robots=1)
        frame0 = _make_frame(0)

        stacked = buffer.append_and_stack(0, frame0)

        np.testing.assert_array_equal(
            stacked["observation.neighbor_state"],
            np.concatenate([frame0["observation.neighbor_state"]] * 3),
        )
        np.testing.assert_array_equal(
            stacked["observation.neighbor_mask"],
            np.concatenate([frame0["observation.neighbor_mask"]] * 3),
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

    def test_reset_clears_history_and_warm_up_repeats_again(self) -> None:
        horizon = 3
        buffer = ObservationHistoryBuffer(observation_horizon=horizon, num_robots=1)
        for i in range(horizon + 2):
            buffer.append_and_stack(0, _make_frame(i))

        buffer.reset()
        new_frame = _make_frame(999)
        stacked = buffer.append_and_stack(0, new_frame)

        np.testing.assert_array_equal(
            stacked["observation.neighbor_state"],
            np.concatenate([new_frame["observation.neighbor_state"]] * horizon),
        )
        np.testing.assert_array_equal(
            stacked["observation.neighbor_mask"],
            np.concatenate([new_frame["observation.neighbor_mask"]] * horizon),
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


if __name__ == "__main__":
    unittest.main()
