from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.config_loaders import EncoderConfig, FlowConfig
from learning.train_dagger import DaggerConfig, DaggerTrainer


def _minimal_dagger_config(**overrides: object) -> DaggerConfig:
    """Build a DaggerConfig with every required field filled by an inert placeholder.

    start_with_aggregation=True skips the on-disk dataset_root existence check, so
    none of these values need to point at anything real -- only the fields under
    test (round_seeds / restart_round_seed / dagger_iterations here) matter.
    """
    defaults: dict[str, object] = dict(
        system="single_integrator",
        experiment_config={},
        repo_id="test/repo",
        dataset_root=Path("unused"),
        start_with_aggregation=True,
        planner_name="casadi",
        dagger_iterations=3,
        trajectories_per_iteration=[10],
        steps_per_trajectory=50,
        action_noise_std=0.0,
        expert_mix_beta_start=0.8,
        expert_mix_beta_end=0.0,
        expert_mix_beta_decay_rate=None,
        expert_mix_decay_after_eval_success=None,
        adaptive_beta_recovery=False,
        expert_mix_beta_recovery=1.0,
        expert_mix_beta_recovery_increment=1.0,
        target_epochs_per_round=[10.0],
        eval_episodes=0,
        eval_steps=None,
        eval_seed_start=0,
        eval_action_noise_std=0.0,
        batch_size=64,
        learning_rate=1e-3,
        mlp_hidden_dims=(64,),
        prediction_horizon=1,
        observation_horizon=1,
        encoder_config=EncoderConfig(encoder_type="deepset", kwargs={}),
        policy_type="mlp",
        flow_config=FlowConfig(),
        checkpoint_dir=Path("unused"),
        seed=0,
        max_train_steps=None,
    )
    defaults.update(overrides)
    return DaggerConfig(**defaults)


class DaggerConfigRoundSeedsTests(unittest.TestCase):
    def test_round_seeds_matching_dagger_iterations_is_accepted(self) -> None:
        cfg = _minimal_dagger_config(dagger_iterations=3, round_seeds=[0, 0, 1])
        self.assertEqual(cfg.round_seeds, [0, 0, 1])

    def test_round_seeds_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _minimal_dagger_config(dagger_iterations=3, round_seeds=[0, 1])

    def test_round_seeds_and_restart_default_to_unset(self) -> None:
        cfg = _minimal_dagger_config()
        self.assertIsNone(cfg.round_seeds)
        self.assertFalse(cfg.restart_round_seed)


class DaggerConfigBetaRecoveryTests(unittest.TestCase):
    def test_default_beta_recovery_matches_expert_only_behavior(self) -> None:
        cfg = _minimal_dagger_config()
        self.assertEqual(cfg.expert_mix_beta_recovery, 1.0)
        self.assertEqual(cfg.expert_mix_beta_recovery_increment, 1.0)

    def test_beta_recovery_out_of_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _minimal_dagger_config(expert_mix_beta_recovery=1.5)
        with self.assertRaises(ValueError):
            _minimal_dagger_config(expert_mix_beta_recovery=-0.1)

    def test_non_positive_beta_recovery_increment_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _minimal_dagger_config(expert_mix_beta_recovery_increment=0.0)
        with self.assertRaises(ValueError):
            _minimal_dagger_config(expert_mix_beta_recovery_increment=-0.25)

    def test_valid_gradual_recovery_schedule_is_accepted(self) -> None:
        cfg = _minimal_dagger_config(
            expert_mix_beta_recovery=0.75,
            expert_mix_beta_recovery_increment=0.25,
        )
        self.assertEqual(cfg.expert_mix_beta_recovery, 0.75)
        self.assertEqual(cfg.expert_mix_beta_recovery_increment, 0.25)


class DaggerConfigEvalToleranceOverridesTests(unittest.TestCase):
    def test_defaults_to_none_and_is_independent_of_tolerance_overrides(self) -> None:
        cfg = _minimal_dagger_config(tolerance_overrides={"pos_tol": 0.1})
        self.assertIsNone(cfg.eval_tolerance_overrides)
        self.assertEqual(cfg.tolerance_overrides, {"pos_tol": 0.1})

    def test_can_differ_from_tolerance_overrides(self) -> None:
        cfg = _minimal_dagger_config(
            tolerance_overrides={"pos_tol": 0.1},
            eval_tolerance_overrides={"pos_tol": 0.2, "theta_tol": 1.1},
        )
        self.assertEqual(cfg.tolerance_overrides, {"pos_tol": 0.1})
        self.assertEqual(cfg.eval_tolerance_overrides, {"pos_tol": 0.2, "theta_tol": 1.1})

    def test_non_positive_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _minimal_dagger_config(eval_tolerance_overrides={"pos_tol": 0.0})
        with self.assertRaises(ValueError):
            _minimal_dagger_config(eval_tolerance_overrides={"pos_tol": -0.1})


class DaggerTrainerSchedulesRoundSeedsTests(unittest.TestCase):
    def test_length_one_round_seeds_broadcasts_to_every_round(self) -> None:
        _, _, _, round_seeds = DaggerTrainer.schedules(
            trajectories=[10],
            epochs=[10.0],
            rounds=3,
            round_seeds=[5],
        )
        self.assertEqual(round_seeds, [5, 5, 5])

    def test_exact_length_round_seeds_is_preserved(self) -> None:
        _, _, _, round_seeds = DaggerTrainer.schedules(
            trajectories=[10],
            epochs=[10.0],
            rounds=3,
            round_seeds=[0, 0, 1],
        )
        self.assertEqual(round_seeds, [0, 0, 1])

    def test_mismatched_round_seeds_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            DaggerTrainer.schedules(
                trajectories=[10],
                epochs=[10.0],
                rounds=3,
                round_seeds=[0, 1],
            )

    def test_omitted_round_seeds_stays_none(self) -> None:
        _, _, _, round_seeds = DaggerTrainer.schedules(
            trajectories=[10],
            epochs=[10.0],
            rounds=3,
        )
        self.assertIsNone(round_seeds)


if __name__ == "__main__":
    unittest.main()
