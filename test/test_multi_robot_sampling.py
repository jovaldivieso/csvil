from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.factory import DynamicsFactory

DT = 0.05


class MixedFixedAndRandomizedGoalSamplingTests(unittest.TestCase):
    """Regression test: randomize_goal_for_reset used to retry every robot
    in fleet order, including fixed-goal ones (randomize_goal=False), for
    which randomize_goal_for_reset() is a no-op. If an earlier *randomized*
    robot's draw happened to land within d_safe of a later robot's fixed
    goal, every retry of that later, fixed robot was identical and the
    method exhausted its attempts and raised -- even though simply
    redrawing the earlier randomized robot would trivially have produced a
    valid layout. Fixed goals are now committed before any randomized one
    is drawn, so only the randomized robot is ever retried.
    """

    def test_randomized_robot_is_redrawn_instead_of_raising_on_a_fixed_goal_conflict(self) -> None:
        # Seed 5 is pinned deliberately: np.random.default_rng(5)'s first
        # uniform(-1, 1, size=2) draw is (0.610, 0.616), within d_safe=0.3 of
        # the fixed goal (0.5, 0.5) below; its second draw, (0.031, -0.428),
        # is not. Robot 0 (randomized) is processed before robot 1 (fixed)
        # consumes any RNG calls, so this is also the *first* draw the
        # sampler itself makes -- reproducing the exact ordering conflict
        # this test guards against, deterministically.
        fixed_goal = [0.5, 0.5]
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": DT,
                "d_safe": 0.3,
                "robots": [
                    {"system": "single_integrator", "config": {
                        "dt": DT, "max_vel": 1.0, "randomize_goal": True, "goal": [0.0, 0.0],
                    }},
                    {"system": "single_integrator", "config": {
                        "dt": DT, "max_vel": 1.0, "randomize_goal": False, "goal": fixed_goal,
                    }},
                ],
            },
        )

        rng = np.random.default_rng(5)
        simulator.randomize_goal_for_reset(rng)

        np.testing.assert_allclose(simulator.simulators[1].goal_state, fixed_goal)
        np.testing.assert_allclose(simulator.simulators[0].goal_state, [0.03065112, -0.42839724], atol=1e-6)
        distance = float(np.linalg.norm(simulator.simulators[0].goal_state - np.asarray(fixed_goal)))
        self.assertGreaterEqual(distance, simulator.d_safe)

    def test_conflicting_fixed_goals_raise_immediately(self) -> None:
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": DT,
                "d_safe": 0.3,
                "robots": [
                    {"system": "single_integrator", "config": {
                        "dt": DT, "max_vel": 1.0, "randomize_goal": False, "goal": [0.0, 0.0],
                    }},
                    {"system": "single_integrator", "config": {
                        "dt": DT, "max_vel": 1.0, "randomize_goal": False, "goal": [0.1, 0.0],
                    }},
                ],
            },
        )

        with self.assertRaises(RuntimeError):
            simulator.randomize_goal_for_reset(np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
