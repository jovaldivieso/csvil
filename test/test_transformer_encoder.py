from __future__ import annotations

import os
import sys
import unittest

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.models.encoder import EncoderFactory, ObservationEncoder
from learning.models.transformer_encoder import TransformerEncoder


class TransformerEncoderTests(unittest.TestCase):
    def test_encoder_factory_and_interface(self) -> None:
        encoder = EncoderFactory.create(
            "transformer",
            state_dim=8,
            neighbor_feature_dim=2,
            neighbor_slots=1,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
        )
        self.assertIsInstance(encoder, ObservationEncoder)
        self.assertIsInstance(encoder, TransformerEncoder)
        self.assertEqual(encoder.ego_dim, 6)
        self.assertEqual(encoder.out_dim, 6 + 8)

    def test_forward_runs(self) -> None:
        neighbor_slots, neighbor_feature_dim = 1, 4
        state_dim = 6 + neighbor_slots * neighbor_feature_dim
        encoder = EncoderFactory.create(
            "transformer",
            state_dim=state_dim,
            neighbor_feature_dim=neighbor_feature_dim,
            neighbor_slots=neighbor_slots,
            hidden_dim=8,
            num_heads=2,
        )
        batch = 3
        observation = {
            "observation.environment_state": torch.randn(batch, 2),
            "observation.state": torch.randn(batch, 4),
            "observation.neighbor_state": torch.randn(batch, neighbor_slots * neighbor_feature_dim),
            "observation.neighbor_mask": torch.ones(batch, neighbor_slots),
        }
        out = encoder(observation)
        self.assertEqual(tuple(out.shape), (batch, encoder.out_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_neighbor_slots_are_not_scrambled(self) -> None:
        """Regression guard: Transformer must reshape neighbor-major via the
        shared helper and gate attention using each neighbor's own mask, not
        a naive view() that would mix different neighbors' features
        together."""
        torch.manual_seed(0)
        neighbor_slots, neighbor_feature_dim = 2, 1

        raw_neighbor_state = torch.tensor([[10.0, 20.0]])
        # neighbor0 visible; neighbor1 invisible.
        raw_neighbor_mask = torch.tensor([[1.0, 0.0]])

        neighbor_obs, neighbor_mask = ObservationEncoder._split_neighbor_tensors(
            raw_neighbor_state, raw_neighbor_mask, neighbor_feature_dim
        )
        torch.testing.assert_close(neighbor_obs[0, 0], torch.tensor([10.0]))
        torch.testing.assert_close(neighbor_obs[0, 1], torch.tensor([20.0]))
        torch.testing.assert_close(neighbor_mask[0], torch.tensor([1.0, 0.0]))

        state_dim = 1 + neighbor_slots * neighbor_feature_dim
        encoder = EncoderFactory.create(
            "transformer",
            state_dim=state_dim,
            neighbor_feature_dim=neighbor_feature_dim,
            neighbor_slots=neighbor_slots,
            hidden_dim=8,
            num_heads=2,
        )
        encoder.eval()
        observation = {
            "observation.environment_state": torch.zeros(1, 1),
            "observation.state": torch.zeros(1, 0),
            "observation.neighbor_state": raw_neighbor_state,
            "observation.neighbor_mask": raw_neighbor_mask,
        }
        with torch.no_grad():
            out = encoder(observation)
        self.assertEqual(tuple(out.shape), (1, encoder.out_dim))
        self.assertFalse(torch.isnan(out).any())

        # neighbor1 is masked out, so the attention padding mask must
        # exclude it entirely -- changing its value must not move the
        # pooled output at all.
        alternate_neighbor_state = raw_neighbor_state.clone()
        alternate_neighbor_state[0, 1] = 999.0
        with torch.no_grad():
            alternate_out = encoder({**observation, "observation.neighbor_state": alternate_neighbor_state})
        torch.testing.assert_close(out, alternate_out)


if __name__ == "__main__":
    unittest.main()
