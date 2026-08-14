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


class MultiRobotMaskSemanticsTests(unittest.TestCase):
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