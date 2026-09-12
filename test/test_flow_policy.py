from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.models.encoder import EncoderFactory
from learning.models.flow_policy import FlowPolicy


def _build_policy(action_scale=None) -> FlowPolicy:
    encoder = EncoderFactory.create(
        "deepset", state_dim=6, neighbor_feature_dim=4, neighbor_slots=0,
        observation_horizon=1, phi_dims=[8], rho_dims=[4],
    )
    # These are correctness checks on the normalization math, not on
    # torch.compile's own codegen -- patched out to avoid depending on
    # Inductor's (unrelated, environment-specific) async compile workers.
    with mock.patch("torch.compile", side_effect=lambda module, **kwargs: module):
        return FlowPolicy(
            action_dim=2, obs_encoder=encoder, hidden_dims=[16], prediction_horizon=3,
            num_inference_steps=4, action_scale=action_scale,
        )


def _fake_observation(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "observation.environment_state": torch.zeros(batch_size, 2),
        "observation.state": torch.zeros(batch_size, 3),
        "observation.state_mask": torch.ones(batch_size, 1),
        "observation.neighbor_state": torch.zeros(batch_size, 0),
        "observation.neighbor_mask": torch.zeros(batch_size, 0),
    }


class ActionScaleValidationTests(unittest.TestCase):
    def test_wrong_length_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _build_policy(action_scale=[1.0, 2.0, 3.0])

    def test_non_positive_entry_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _build_policy(action_scale=[1.0, 0.0])
        with self.assertRaises(ValueError):
            _build_policy(action_scale=[-1.0, 1.0])

    def test_omitted_defaults_to_all_ones(self) -> None:
        policy = _build_policy(action_scale=None)
        torch.testing.assert_close(policy.action_scale, torch.ones(2))


class ComputeLossActionScaleInvarianceTests(unittest.TestCase):
    """compute_loss divides actions by action_scale before ever touching the
    x_0 ~ N(0, 1) noise prior (see FlowPolicy's own docstring for why: an
    isotropic prior trained against anisotropic raw action bounds lets
    whichever dimension has the largest physical range dominate the loss,
    regardless of task relevance). That means feeding in actions already
    pre-multiplied by a given scale, under that same scale, must reduce to
    exactly the unscaled case: both compute x_1 = raw_actions internally.
    """

    def test_prescaled_actions_reproduce_the_unscaled_loss(self) -> None:
        policy = _build_policy(action_scale=None)
        obs = _fake_observation(batch_size=5)
        raw_actions = torch.randn(5, policy.prediction_horizon, policy.action_dim)

        torch.manual_seed(0)
        baseline_loss = policy.compute_loss(obs, raw_actions)

        scale = torch.tensor([2.0, 5.0])
        with torch.no_grad():
            policy.action_scale.copy_(scale)
        scaled_actions = raw_actions * scale

        torch.manual_seed(0)
        rescaled_loss = policy.compute_loss(obs, scaled_actions)

        torch.testing.assert_close(rescaled_loss, baseline_loss)

    def test_skipping_normalization_gives_a_different_loss(self) -> None:
        """Sanity check that the test above isn't vacuously true -- omitting
        the division by action_scale for the same prescaled input must land
        on a genuinely different loss value, confirming the test can fail.
        """
        policy = _build_policy(action_scale=None)
        obs = _fake_observation(batch_size=5)
        raw_actions = torch.randn(5, policy.prediction_horizon, policy.action_dim)

        torch.manual_seed(0)
        baseline_loss = policy.compute_loss(obs, raw_actions)

        scale = torch.tensor([2.0, 5.0])
        scaled_actions = raw_actions * scale  # deliberately NOT re-pointing action_scale

        torch.manual_seed(0)
        unnormalized_loss = policy.compute_loss(obs, scaled_actions)

        self.assertFalse(torch.allclose(unnormalized_loss, baseline_loss))


class SelectActionRescalesToPhysicalUnitsTests(unittest.TestCase):
    """select_action integrates entirely in the normalized space compute_loss
    trains in, then must rescale back to physical action units before
    returning -- every downstream consumer (apply_execution_noise,
    simulator.step, ...) expects physical units, not the network's internal
    representation.
    """

    def test_zero_network_output_returns_exactly_the_scaled_noise_draw(self) -> None:
        # With every weight and bias zeroed, _predict_velocity is
        # identically 0 regardless of input (Linear(0)=0, Mish(0)=0
        # propagates through every layer), so the Euler loop's
        # x.add_(0, alpha=dt) never changes x away from its initial
        # torch.randn draw -- the only transformation left to verify is
        # select_action's final "* action_scale" rescale.
        scale = [1.0, 100.0]
        policy = _build_policy(action_scale=scale)
        with torch.no_grad():
            for param in policy.net.parameters():
                param.zero_()

        torch.manual_seed(42)
        expected_x0 = torch.randn(1, policy.prediction_horizon, policy.action_dim)

        torch.manual_seed(42)
        obs = _fake_observation(batch_size=1)
        action = policy.select_action(obs)

        torch.testing.assert_close(action, expected_x0 * torch.tensor(scale))


if __name__ == "__main__":
    unittest.main()
