from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.factory import DynamicsFactory
from learning.models.deep_set_encoder import DeepSetEncoder


class DeepSetEncoderTests(unittest.TestCase):
    def test_zero_neighbor_input_returns_zero_embedding(self) -> None:
        encoder = DeepSetEncoder(
            in_features=2,
            phi_dims=[8, 8],
            rho_dims=[4],
            pool_type="max",
        )

        x = torch.zeros((3, 0, 2), dtype=torch.float32)
        mask = torch.zeros((3, 0, 1), dtype=torch.float32)

        output = encoder(x, mask)

        self.assertEqual(tuple(output.shape), (3, 4))
        self.assertTrue(torch.allclose(output, torch.zeros_like(output)))

    def test_rejects_featurewise_masks(self) -> None:
        encoder = DeepSetEncoder(in_features=2, phi_dims=[8], rho_dims=[4])
        x = torch.zeros((2, 3, 2), dtype=torch.float32)
        invalid_mask = torch.ones((2, 3, 2), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, r"mask.*\(B, K, 1\)"):
            encoder(x, invalid_mask)

    def test_mask_distinguishes_collision_from_invisible_slot(self) -> None:
        torch.manual_seed(7)
        encoder = DeepSetEncoder(
            in_features=2,
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

        visible_output = encoder(x, visible_mask)
        hidden_output = encoder(x, hidden_mask)

        self.assertFalse(torch.allclose(visible_output, hidden_output))

    def test_all_masked_rows_return_zero_context_without_nan_gradients(self) -> None:
        torch.manual_seed(11)
        encoder = DeepSetEncoder(
            in_features=2,
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

        output = encoder(x, mask)
        self.assertTrue(torch.equal(output[0], torch.zeros_like(output[0])))
        self.assertTrue(torch.isfinite(output).all())

        output.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


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
        observation = simulator.observe(state, validate=False)
        local_observation = simulator.decentralized_policy_observation(observation)
        features = simulator.get_decentralized_dataset_features()

        self.assertEqual(simulator.num_robots, 1)
        self.assertEqual(tuple(local_observation["neighbor_obs"].shape), (0, 2))
        self.assertEqual(tuple(local_observation["neighbor_mask"].shape), (0, 1))
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

        colliding_obs = simulator.observe(colliding_state, validate=False)
        invisible_obs = simulator.observe(invisible_state, validate=False)

        colliding_robot_obs = simulator.decentralized_policy_observation(colliding_obs, 0)
        invisible_robot_obs = simulator.decentralized_policy_observation(invisible_obs, 0)

        self.assertEqual(tuple(colliding_robot_obs["neighbor_obs"].shape), (1, 2))
        self.assertEqual(tuple(colliding_robot_obs["neighbor_mask"].shape), (1, 1))
        self.assertEqual(tuple(invisible_robot_obs["neighbor_obs"].shape), (1, 2))
        self.assertEqual(tuple(invisible_robot_obs["neighbor_mask"].shape), (1, 1))
        np.testing.assert_allclose(colliding_robot_obs["neighbor_obs"], np.zeros((1, 2), dtype=np.float32))
        np.testing.assert_allclose(invisible_robot_obs["neighbor_obs"], np.zeros((1, 2), dtype=np.float32))
        self.assertEqual(float(colliding_robot_obs["neighbor_mask"][0, 0]), 1.0)
        self.assertEqual(float(invisible_robot_obs["neighbor_mask"][0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()