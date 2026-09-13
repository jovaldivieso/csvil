from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.config_loaders import DEFAULT_DAGGER_TRAINING_CONFIG, load_dagger_training_config


def _write_policy_config(training_yaml_lines: list[str]) -> Path:
    """Write a minimal policy config with the given 'training:' body to a
    uniquely-named temp file (load_dagger_training_config's yaml cache keys
    off the path, so each test needs its own file to avoid a stale hit)."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    body = "model:\n  policy_type: mlp\ntraining:\n" + "\n".join(
        f"  {line}" for line in training_yaml_lines
    )
    with os.fdopen(fd, "w") as f:
        f.write(body)
    return Path(path)


class LoadDaggerTrainingConfigTests(unittest.TestCase):
    """Regression coverage for the 'training:' section's key allowlist --
    DEFAULT_DAGGER_TRAINING_CONFIG is the single source of truth for which
    keys a policy YAML may set; DaggerConfig/collect_dagger_rollouts tests
    never exercise this loader directly, so a field added everywhere else
    but here still fails at CLI startup with 'Unknown key(s) in policy
    config' -- exactly what happened before this file existed.
    """

    def test_every_dagger_config_field_used_by_train_dagger_is_a_known_key(self) -> None:
        # Mirrors the exact fields learning/train_dagger.py's CLI option
        # parsing reads out of this dict via option(...). If a name is added
        # there without a matching DEFAULT_DAGGER_TRAINING_CONFIG entry, the
        # yaml loader rejects it as "unknown" before it ever reaches
        # train_dagger.py -- this test would catch that mismatch here
        # instead of at CLI runtime.
        expected_keys = {
            "planner", "dagger_iterations", "trajectories_per_iteration",
            "steps_per_trajectory", "action_noise_std", "training_curriculum",
            "round_seeds", "restart_round_seed", "initial_states", "goal_states",
            "workspace_bounds", "tolerance_overrides", "eval_tolerance_overrides",
            "expert_mix_beta_start",
            "expert_mix_beta_end", "expert_mix_beta_decay_rate",
            "expert_mix_decay_after_eval_success", "adaptive_beta_recovery",
            "expert_mix_beta_recovery", "expert_mix_beta_recovery_increment",
            "target_epochs_per_round", "eval_episodes", "eval_steps",
            "eval_seed_start", "eval_action_noise_std", "batch_size",
            "learning_rate", "seed", "max_train_steps",
        }
        self.assertEqual(expected_keys, set(DEFAULT_DAGGER_TRAINING_CONFIG))

    def test_beta_recovery_fields_are_accepted_from_a_policy_yaml(self) -> None:
        path = _write_policy_config([
            "expert_mix_beta_recovery: 0.75",
            "expert_mix_beta_recovery_increment: 0.25",
        ])
        try:
            config = load_dagger_training_config(path)
        finally:
            path.unlink()
        self.assertEqual(config["expert_mix_beta_recovery"], 0.75)
        self.assertEqual(config["expert_mix_beta_recovery_increment"], 0.25)

    def test_beta_recovery_fields_default_to_expert_only_when_omitted(self) -> None:
        path = _write_policy_config(["dagger_iterations: 3"])
        try:
            config = load_dagger_training_config(path)
        finally:
            path.unlink()
        self.assertEqual(config["expert_mix_beta_recovery"], 1.0)
        self.assertEqual(config["expert_mix_beta_recovery_increment"], 1.0)

    def test_eval_tolerance_overrides_is_accepted_and_independent_of_tolerance_overrides(self) -> None:
        path = _write_policy_config([
            "tolerance_overrides:",
            "  pos_tol: 0.1",
            "eval_tolerance_overrides:",
            "  pos_tol: 0.2",
            "  theta_tol: 1.1",
        ])
        try:
            config = load_dagger_training_config(path)
        finally:
            path.unlink()
        self.assertEqual(config["tolerance_overrides"], {"pos_tol": 0.1})
        self.assertEqual(config["eval_tolerance_overrides"], {"pos_tol": 0.2, "theta_tol": 1.1})

    def test_eval_tolerance_overrides_defaults_to_none_when_omitted(self) -> None:
        path = _write_policy_config(["dagger_iterations: 3"])
        try:
            config = load_dagger_training_config(path)
        finally:
            path.unlink()
        self.assertIsNone(config["eval_tolerance_overrides"])

    def test_unknown_key_is_still_rejected(self) -> None:
        path = _write_policy_config(["not_a_real_field: 1"])
        try:
            with self.assertRaises(ValueError):
                load_dagger_training_config(path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
