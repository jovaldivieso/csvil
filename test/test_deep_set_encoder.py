from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.factory import DynamicsFactory
from learning.models.deepset_encoder import DeepSetEncoder
from learning.models.encoder import ObservationEncoder, EncoderFactory


class DeepSetEncoderTests(unittest.TestCase):
    def test_encoder_factory_and_interface(self) -> None:
        encoder = EncoderFactory.create(
            "deepset",
            state_dim=3,
            neighbor_feature_dim=2,
            neighbor_slots=0,
            phi_dims=[8],
            rho_dims=[4],
        )
        self.assertIsInstance(encoder, ObservationEncoder)
        self.assertEqual(encoder.out_dim, 7)

        with self.assertRaises(ValueError):
            EncoderFactory.create("unknown", state_dim=3, neighbor_feature_dim=2, neighbor_slots=0)

    def test_zero_neighbor_input_returns_zero_embedding(self) -> None:
        encoder = DeepSetEncoder(
            state_dim=3,
            neighbor_feature_dim=2,
            neighbor_slots=0,
            phi_dims=[8, 8],
            rho_dims=[4],
            pool_type="max",
        )

        x = torch.zeros((3, 0, 2), dtype=torch.float32)
        mask = torch.zeros((3, 0, 1), dtype=torch.float32)
        ego = torch.zeros((3, 3), dtype=torch.float32)

        output = encoder({
            "observation.environment_state": ego,
            "observation.state": torch.zeros((3, 0)),
            "observation.neighbor_state": x.reshape(3, -1),
            "observation.neighbor_mask": mask.reshape(3, -1),
        })

        self.assertEqual(tuple(output.shape), (3, 7))
        self.assertTrue(torch.allclose(output[:, 3:], torch.zeros_like(output[:, 3:])))

    def test_rejects_featurewise_masks(self) -> None:
        encoder = DeepSetEncoder(state_dim=9, neighbor_feature_dim=2, neighbor_slots=2, phi_dims=[8], rho_dims=[4])
        x = torch.zeros((2, 3, 2), dtype=torch.float32)
        invalid_mask = torch.ones((2, 3, 2), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, r"Canonical neighbor tensors"):
            encoder({
                "observation.environment_state": torch.zeros((2, 1)),
                "observation.state": torch.zeros((2, 2)),
                "observation.neighbor_state": x.reshape(2, -1),
                "observation.neighbor_mask": invalid_mask,
            })

    def test_mask_distinguishes_collision_from_invisible_slot(self) -> None:
        torch.manual_seed(7)
        encoder = DeepSetEncoder(
            state_dim=9,
            neighbor_feature_dim=2,
            neighbor_slots=2,
            phi_dims=[8, 8],
            rho_dims=[4],
            pool_type="sum",
        )

        x = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        visible_mask = torch.tensor([[[1.0], [1.0]]], dtype=torch.float32)
        hidden_mask = torch.tensor([[[1.0], [0.0]]], dtype=torch.float32)

        ego = torch.zeros((1, 3), dtype=torch.float32)
        visible_input = {
            "observation.environment_state": ego,
            "observation.state": torch.zeros((1, 0)),
            "observation.neighbor_state": x.reshape(1, -1),
            "observation.neighbor_mask": visible_mask.reshape(1, -1),
        }
        hidden_input = {**visible_input, "observation.neighbor_mask": hidden_mask.reshape(1, -1)}
        visible_output = encoder(visible_input)
        hidden_output = encoder(hidden_input)

        self.assertFalse(torch.allclose(visible_output, hidden_output))

    def test_all_masked_rows_return_zero_context_without_nan_gradients(self) -> None:
        torch.manual_seed(11)
        encoder = DeepSetEncoder(
            state_dim=11,
            neighbor_feature_dim=2,
            neighbor_slots=3,
            phi_dims=[8, 8],
            rho_dims=[4],
            pool_type="mean",
        )
        x = torch.randn((2, 3, 2), requires_grad=True)
        mask = torch.tensor(
            [
                [[0.0], [0.0], [0.0]],
                [[1.0], [0.0], [0.0]],
            ]
        )

        ego = torch.randn((2, 2))
        output = encoder({
            "observation.environment_state": ego,
            "observation.state": torch.zeros((2, 0)),
            "observation.neighbor_state": x.reshape(2, -1),
            "observation.neighbor_mask": mask.reshape(2, -1),
        })
        self.assertTrue(torch.equal(output[0, 2:], torch.zeros_like(output[0, 2:])))
        self.assertTrue(torch.isfinite(output).all())

        output.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_encoder_accepts_more_runtime_neighbors_than_configured(self) -> None:
        encoder = DeepSetEncoder(
            state_dim=9,
            neighbor_feature_dim=2,
            neighbor_slots=1,
            phi_dims=[8],
            rho_dims=[4],
            pool_type="sum",
        )
        output = encoder(
            {
                "observation.environment_state": torch.zeros((2, 3)),
                "observation.state": torch.zeros((2, 3)),
                "observation.neighbor_state": torch.zeros((2, 6)),
                "observation.neighbor_mask": torch.ones((2, 3)),
            }
        )
        self.assertEqual(tuple(output.shape), (2, 10))

    def test_stacked_neighbor_history_is_not_scrambled_across_time(self) -> None:
        """Regression guard: DeepSet must reshape via the shared, time-major-aware helper
        and pool over neighbors using only the current-timestep mask, not a naive
        view() that would mix different neighbors' per-timestep features together."""
        neighbor_slots, observation_horizon = 2, 2
        neighbor_feature_dim = 1 * observation_horizon

        # Time-major flat layout (oldest frame first, neighbor-minor within each frame):
        # frame0 = [neighbor0=10.0, neighbor1=20.0], frame1 = [neighbor0=11.0, neighbor1=21.0]
        raw_neighbor_state = torch.tensor([[10.0, 20.0, 11.0, 21.0]])
        # neighbor0 visible at both frames; neighbor1 visible only at the earlier
        # frame -- invisible at the current (most recent) frame.
        raw_neighbor_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        neighbor_obs, neighbor_mask = ObservationEncoder._split_neighbor_tensors(
            raw_neighbor_state, raw_neighbor_mask, neighbor_feature_dim, observation_horizon
        )
        torch.testing.assert_close(neighbor_obs[0, 0], torch.tensor([10.0, 11.0]))
        torch.testing.assert_close(neighbor_obs[0, 1], torch.tensor([20.0, 21.0]))
        torch.testing.assert_close(neighbor_mask[0, 0], torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(neighbor_mask[0, 1], torch.tensor([1.0, 0.0]))

        state_dim = 2 + neighbor_slots * (neighbor_feature_dim + observation_horizon)
        encoder = DeepSetEncoder(
            state_dim=state_dim,
            neighbor_feature_dim=neighbor_feature_dim,
            neighbor_slots=neighbor_slots,
            observation_horizon=observation_horizon,
            phi_dims=[8, 8],
            rho_dims=[4],
            pool_type="max",
        )
        observation = {
            "observation.environment_state": torch.zeros(1, 1),
            "observation.state": torch.zeros(1, 1),
            "observation.neighbor_state": raw_neighbor_state,
            "observation.neighbor_mask": raw_neighbor_mask,
        }
        out = encoder(observation)
        self.assertEqual(tuple(out.shape), (1, encoder.out_dim))
        self.assertFalse(torch.isnan(out).any())

        # neighbor1 is masked out at the current timestep, so its entire packed
        # (feature, mask) history must be excluded from max-pooling regardless of
        # its value -- changing it must not move the output at all.
        alternate_neighbor_state = raw_neighbor_state.clone()
        alternate_neighbor_state[0, 1] = 999.0
        alternate_neighbor_state[0, 3] = 999.0
        alternate_out = encoder({**observation, "observation.neighbor_state": alternate_neighbor_state})
        torch.testing.assert_close(out, alternate_out)


class MultiRobotMaskSemanticsTests(unittest.TestCase):
    def test_fleet_of_one_uses_empty_neighbor_contract(self) -> None:
        simulator = DynamicsFactory.create(
            system_name="double_integrator",
            config={
                "dt": 0.05,
                "goal": [0.0, 0.0],
                "randomize_goal": False,
            },
        )
        state = simulator.random_initial_state(np.random.default_rng(3))
        observation = simulator.observe(state)
        local_observation = simulator.decentralized_policy_observation(observation)
        features = simulator.get_dataset_features()

        self.assertEqual(simulator.num_robots, 1)
        self.assertEqual(tuple(local_observation["observation.neighbor_state"].shape), (0,))
        self.assertEqual(tuple(local_observation["observation.neighbor_mask"].shape), (0,))
        self.assertEqual(features["observation.neighbor_state"]["shape"], (0,))
        self.assertEqual(features["observation.neighbor_mask"]["shape"], (0,))

    def test_decentralized_observation_distinguishes_zero_distance_from_invisible_neighbor(self) -> None:
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": 0.05,
                "d_safe": 0.1,
                "inter_robot_visibility_radius": 0.1,
                "robots": [
                    {
                        "system": "double_integrator",
                        "config": {
                            "dt": 0.05,
                            "goal": [0.0, 0.0],
                            "randomize_goal": False,
                        },
                    },
                    {
                        "system": "double_integrator",
                        "config": {
                            "dt": 0.05,
                            "goal": [1.0, 1.0],
                            "randomize_goal": False,
                        },
                    },
                ],
            },
        )

        colliding_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        invisible_state = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0], dtype=float)

        colliding_obs = simulator.observe(colliding_state)
        invisible_obs = simulator.observe(invisible_state)

        colliding_robot_obs = simulator.decentralized_policy_observation(colliding_obs, 0)
        invisible_robot_obs = simulator.decentralized_policy_observation(invisible_obs, 0)

        self.assertEqual(tuple(colliding_robot_obs["observation.neighbor_state"].shape), (2,))
        self.assertEqual(tuple(colliding_robot_obs["observation.neighbor_mask"].shape), (1,))
        self.assertEqual(tuple(invisible_robot_obs["observation.neighbor_state"].shape), (2,))
        self.assertEqual(tuple(invisible_robot_obs["observation.neighbor_mask"].shape), (1,))
        np.testing.assert_allclose(colliding_robot_obs["observation.neighbor_state"], np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(invisible_robot_obs["observation.neighbor_state"], np.zeros(2, dtype=np.float32))
        self.assertEqual(float(colliding_robot_obs["observation.neighbor_mask"][0]), 1.0)
        self.assertEqual(float(invisible_robot_obs["observation.neighbor_mask"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()