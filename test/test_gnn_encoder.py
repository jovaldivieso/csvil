from __future__ import annotations

import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.models.encoder import EncoderFactory, ObservationEncoder
from learning.models.gnn_encoder import GNNEncoder


class GNNEncoderTests(unittest.TestCase):
    def test_encoder_factory_and_interface(self) -> None:
        encoder = EncoderFactory.create(
            "gnn",
            state_dim=8,
            neighbor_feature_dim=2,
            neighbor_slots=1,
            hidden_dim=8,
            num_layers=1,
        )
        self.assertIsInstance(encoder, ObservationEncoder)
        self.assertIsInstance(encoder, GNNEncoder)
        self.assertEqual(encoder.ego_dim, 5)
        self.assertEqual(encoder.out_dim, 5 + 8)

    def test_forward_runs_with_default_horizon_one(self) -> None:
        neighbor_slots, neighbor_feature_dim = 1, 4
        state_dim = 5 + neighbor_slots * (neighbor_feature_dim + 1)
        encoder = EncoderFactory.create(
            "gnn",
            state_dim=state_dim,
            neighbor_feature_dim=neighbor_feature_dim,
            neighbor_slots=neighbor_slots,
            hidden_dim=8,
        )
        batch = 3
        observation = {
            "observation.environment_state": torch.randn(batch, 2),
            "observation.state": torch.randn(batch, 3),
            "observation.neighbor_state": torch.randn(batch, neighbor_slots * neighbor_feature_dim),
            "observation.neighbor_mask": torch.ones(batch, neighbor_slots),
        }
        out = encoder(observation)
        self.assertEqual(tuple(out.shape), (batch, encoder.out_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_forward_runs_with_observation_horizon_greater_than_one(self) -> None:
        neighbor_slots, observation_horizon, per_frame_dim = 2, 3, 2
        neighbor_feature_dim = per_frame_dim * observation_horizon
        ego_dim = 6
        state_dim = ego_dim + neighbor_slots * (neighbor_feature_dim + observation_horizon)

        encoder = EncoderFactory.create(
            "gnn",
            state_dim=state_dim,
            neighbor_feature_dim=neighbor_feature_dim,
            neighbor_slots=neighbor_slots,
            observation_horizon=observation_horizon,
            hidden_dim=8,
        )
        self.assertEqual(encoder.ego_dim, ego_dim)

        batch = 4
        observation = {
            "observation.environment_state": torch.randn(batch, 3),
            "observation.state": torch.randn(batch, 3),
            "observation.neighbor_state": torch.randn(batch, neighbor_slots * neighbor_feature_dim),
            "observation.neighbor_mask": torch.randint(0, 2, (batch, neighbor_slots * observation_horizon)).float(),
        }
        out = encoder(observation)
        self.assertEqual(tuple(out.shape), (batch, encoder.out_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_stacked_neighbor_history_is_not_scrambled_across_time(self) -> None:
        """Regression guard: GNN must reshape via the shared, time-major-aware helper,
        not a naive `.view()` that would mix different neighbors' features together
        once observation_horizon > 1."""
        neighbor_slots, observation_horizon = 2, 2
        neighbor_feature_dim = 1 * observation_horizon

        # Time-major flat layout (oldest frame first, neighbor-minor within each frame):
        # frame0 = [neighbor0=10.0, neighbor1=20.0], frame1 = [neighbor0=11.0, neighbor1=21.0]
        raw_neighbor_state = torch.tensor([[10.0, 20.0, 11.0, 21.0]])
        # neighbor0 visible at both frames; neighbor1 visible only at the earlier frame.
        raw_neighbor_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        neighbor_obs, neighbor_mask = ObservationEncoder._split_neighbor_tensors(
            raw_neighbor_state, raw_neighbor_mask, neighbor_feature_dim, observation_horizon
        )

        torch.testing.assert_close(neighbor_obs[0, 0], torch.tensor([10.0, 11.0]))
        torch.testing.assert_close(neighbor_obs[0, 1], torch.tensor([20.0, 21.0]))
        torch.testing.assert_close(neighbor_mask[0, 0], torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(neighbor_mask[0, 1], torch.tensor([1.0, 0.0]))

        # GNNEncoder.forward() must gate visibility on the most recent frame only.
        state_dim = 2 + neighbor_slots * (neighbor_feature_dim + observation_horizon)
        encoder = EncoderFactory.create(
            "gnn",
            state_dim=state_dim,
            neighbor_feature_dim=neighbor_feature_dim,
            neighbor_slots=neighbor_slots,
            observation_horizon=observation_horizon,
            hidden_dim=4,
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

        # neighbor1 is masked out at the current timestep, so it must never be
        # gathered into a message at all -- changing its packed history must
        # not move the output.
        alternate_neighbor_state = raw_neighbor_state.clone()
        alternate_neighbor_state[0, 1] = 999.0
        alternate_neighbor_state[0, 3] = 999.0
        alternate_out = encoder({**observation, "observation.neighbor_state": alternate_neighbor_state})
        torch.testing.assert_close(out, alternate_out)


if __name__ == "__main__":
    unittest.main()
