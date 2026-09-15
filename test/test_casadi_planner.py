from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from learning.dagger.rollouts import apply_execution_noise
from planning.casadi_planner import PlannerSolveError

CONFIG_PATH = os.path.join(PROJECT_ROOT, "test", "config", "multi_unicycle2_casadi_config.yaml")


class MpcWarmStartFailureRecoveryTests(unittest.TestCase):
    """Regression test for a real training failure: adding a hard d_collision
    floor beneath the soft d_safe buffer (see casadi_planner.py's collision
    constraints) gave CasadiPlanner's MPC mode a genuinely non-convex hard
    "stay outside this disk" constraint it didn't have before. Its shifted
    warm start (from the previous solve, see __call__) can leave IPOPT stuck
    at a locally infeasible point relative to that constraint once noisy
    execution has pushed the actual state -- often at near-saturated
    velocity -- away from what the previous solve predicted, even though a
    feasible trajectory exists (confirmed by solving the exact same x0/goal
    cold, which succeeds immediately). Before the hard floor, collision
    avoidance was purely soft (slack-relaxable), so this failure mode could
    not occur at all.

    This exact scenario (config, initial state, goal, noise seed) is a
    verbatim reproduction of a real `train_dagger.py` DAgger-collection
    failure: solving open-loop from this initial state with only execution
    noise (no DAgger policy-mixing at all -- irrelevant to triggering this)
    reliably fails at step 36 without the cold-restart recovery below.
    """

    def test_warm_start_failure_recovers_via_cold_restart_instead_of_raising(self) -> None:
        with open(CONFIG_PATH) as f:
            raw_config = yaml.safe_load(f)
        validated = validate_system_config(system_name="multi_robot", raw_config=raw_config)
        simulator = DynamicsFactory.create(system_name="multi_robot", config=validated)
        planner = PlannerFactory.create(planner_name="casadi", simulator=simulator, config=dict(validated))

        initial_state = np.array(
            [-2.099, 1.016878, -1.217284, 0.0, 0.0, 1.590059, 1.925185, -1.838818, 0.0, 0.0]
        )
        goal_state_full = np.array(
            [2.887915, -2.448935, 0.840153, 0.0, 0.0, -0.085179, -2.539448, 1.413332, 0.0, 0.0]
        )
        simulator.set_goal(np.concatenate([goal_state_full[0:3], goal_state_full[5:8]]))

        rng = np.random.default_rng(16927820532609799510)
        state = simulator.reset(initial_state)
        planner.reset()

        # The bare reproduction fails at step 36; 40 leaves a small margin
        # while keeping the test's real IPOPT solve count (and runtime) low.
        for step in range(40):
            obs = simulator.observe(state)
            try:
                action = planner(obs)
            except PlannerSolveError as exc:
                self.fail(f"Planner failed at step={step} instead of recovering via cold restart: {exc}")
            executed_action = apply_execution_noise(simulator, action, action_noise_std=0.03, rng=rng)
            state = simulator.step(state, executed_action)
            if simulator.should_terminate_rollout(state):
                break


if __name__ == "__main__":
    unittest.main()
