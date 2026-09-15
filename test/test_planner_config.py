from __future__ import annotations

import os
import sys
import unittest

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import ConfigurationError, validate_system_config

CONFIG_PATH = os.path.join(PROJECT_ROOT, "test", "config", "multi_unicycle2_casadi_config.yaml")


def _load_raw_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class ProjectorTuningKeysReachValidatedConfigTests(unittest.TestCase):
    """Regression coverage for a real gap: CasadiTrajectoryProjector reads
    terminal_velocity_weight/fallback_terminal_velocity_tol/
    max_consecutive_fallbacks off its planner_config (see
    planning/casadi_projector.py), but validate_system_config's key
    allowlist never listed them, so any expert YAML setting one was
    silently either rejected outright or (if it slipped past that) just
    never made it into the dict PolicyFactory hands the projector -- the
    projector always fell back to its own hardcoded defaults regardless of
    what the config said.
    """

    def test_overridden_values_are_accepted_and_pass_through(self) -> None:
        raw_config = _load_raw_config()
        raw_config["terminal_velocity_weight"] = 42.0
        raw_config["fallback_terminal_velocity_tol"] = 0.007
        raw_config["max_consecutive_fallbacks"] = 3

        validated = validate_system_config(system_name="multi_robot", raw_config=raw_config)

        self.assertEqual(validated["terminal_velocity_weight"], 42.0)
        self.assertEqual(validated["fallback_terminal_velocity_tol"], 0.007)
        self.assertEqual(validated["max_consecutive_fallbacks"], 3)

    def test_omitted_values_default_to_the_projectors_own_hardcoded_fallbacks(self) -> None:
        # These three literals must stay in sync with
        # CasadiTrajectoryProjector.__init__'s own config.get(..., <default>)
        # calls -- a config that omits these keys must behave identically to
        # before they were configurable.
        raw_config = _load_raw_config()

        validated = validate_system_config(system_name="multi_robot", raw_config=raw_config)

        self.assertEqual(validated["terminal_velocity_weight"], 1.0)
        self.assertEqual(validated["fallback_terminal_velocity_tol"], 1e-2)
        self.assertEqual(validated["max_consecutive_fallbacks"], 5)

    def test_non_positive_terminal_velocity_weight_is_rejected(self) -> None:
        raw_config = _load_raw_config()
        raw_config["terminal_velocity_weight"] = 0.0
        with self.assertRaises(ConfigurationError):
            validate_system_config(system_name="multi_robot", raw_config=raw_config)

    def test_negative_fallback_terminal_velocity_tol_is_rejected(self) -> None:
        raw_config = _load_raw_config()
        raw_config["fallback_terminal_velocity_tol"] = -0.01
        with self.assertRaises(ConfigurationError):
            validate_system_config(system_name="multi_robot", raw_config=raw_config)

    def test_non_positive_max_consecutive_fallbacks_is_rejected(self) -> None:
        raw_config = _load_raw_config()
        raw_config["max_consecutive_fallbacks"] = 0
        with self.assertRaises(ConfigurationError):
            validate_system_config(system_name="multi_robot", raw_config=raw_config)


if __name__ == "__main__":
    unittest.main()
