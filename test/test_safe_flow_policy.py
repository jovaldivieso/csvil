from __future__ import annotations

import os
import sys
import unittest
import warnings

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.factory import DynamicsFactory
from learning.models.encoder import EncoderFactory
from learning.models.policy import PolicyFactory
from learning.models.safe_flow_policy import SafeFlowMPCPolicy
from planning.casadi_planner import PlannerSolveError
from planning.casadi_projector import CasadiTrajectoryProjector

DT = 0.05


def _build_two_robot_simulator(goal0: list[float], goal1: list[float]):
    """Two unicycle2 robots with explicit, non-randomized starts/goals."""
    return DynamicsFactory.create(
        system_name="multi_robot",
        config={
            "dt": DT,
            "d_safe": 0.1,
            "robots": [
                {
                    "system": "unicycle2",
                    "config": {
                        "dt": DT, "max_accel": 2.0, "max_omega": 2.0, "max_speed": 2.0,
                        "start": [0.0, 0.0, 0.0, 0.0, 0.0], "goal": goal0, "randomize_goal": False,
                    },
                },
                {
                    "system": "unicycle2",
                    "config": {
                        "dt": DT, "max_accel": 2.0, "max_omega": 2.0, "max_speed": 2.0,
                        "start": [3.0, 3.0, 0.0, 0.0, 0.0], "goal": goal1, "randomize_goal": False,
                    },
                },
            ],
        },
    )


def _build_safe_flow_policy(simulator) -> SafeFlowMPCPolicy:
    encoder = EncoderFactory.create(
        "deepset", state_dim=22, neighbor_feature_dim=8, neighbor_slots=1,
        observation_horizon=2, phi_dims=[8], rho_dims=[4],
    )
    policy = PolicyFactory.create(
        "safeflow",
        action_dim=2, obs_encoder=encoder, hidden_dims=[16], prediction_horizon=3, num_inference_steps=2,
        simulator=simulator, planner_config={},
    )
    assert isinstance(policy, SafeFlowMPCPolicy)
    return policy


class GoalAnchorIndependenceTests(unittest.TestCase):
    """Root-cause fix for the same staleness issue a since-removed
    sync_simulator() hook used to patch around: PolicyFactory.create now
    gives SafeFlowMPCPolicy its own private, goal-zeroed dynamics objects
    instead of live references into whatever simulator built it, so there
    is nothing left to keep in sync with rollout collection's or
    evaluate_current_policy's own separately-constructed simulators.

    This is correct, not just convenient, because every constraint/cost
    CasadiTrajectoryProjector enforces -- dynamics, action/velocity bounds,
    pairwise neighbor distances, terminal velocity -- is equivariant under a
    per-robot rigid transform of invert_obs's reconstructed frame; the goal
    used there is never anything more than a fixed, self-consistent local
    anchor. See learning/models/policy.py's comment for the one way this
    could stop being true (a future world-frame-referencing term).
    """

    def test_local_sims_have_zeroed_goal_regardless_of_source_simulator(self) -> None:
        simulator = _build_two_robot_simulator(goal0=[5.0, 0.0, 0.7], goal1=[-5.0, 3.0, -0.2])
        policy = _build_safe_flow_policy(simulator)

        np.testing.assert_allclose(policy.local_sims[0].goal, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(policy.local_sims[1].goal, [0.0, 0.0, 0.0])
        # projectors[i].sim and local_sims[i] must still be the same object
        # (both are PolicyFactory.create's own copies) -- a projector never
        # needs a goal of its own, separate from local_sims'.
        self.assertIs(policy.projectors[0].sim, policy.local_sims[0])
        self.assertIs(policy.projectors[1].sim, policy.local_sims[1])

    def test_construction_does_not_mutate_the_source_simulator(self) -> None:
        simulator = _build_two_robot_simulator(goal0=[5.0, 0.0, 0.7], goal1=[-5.0, 3.0, -0.2])
        _build_safe_flow_policy(simulator)

        # The deep copy must fully isolate local_sims from the simulator
        # that built them -- a caller that passes the same simulator object
        # for both policy construction and its own rollout (some
        # evaluate_policy.py call sites do) must not find its goal silently
        # zeroed out from under it.
        np.testing.assert_allclose(simulator.simulators[0].goal, [5.0, 0.0, 0.7])
        np.testing.assert_allclose(simulator.simulators[1].goal, [-5.0, 3.0, -0.2])

    def test_projected_action_is_invariant_to_which_wrong_goal_local_sims_hold(self) -> None:
        # The actual mathematical property the whole redesign relies on,
        # checked directly at the projector level: invert_obs's goal is a
        # "gauge" the projector's output must not depend on, as long as it's
        # applied consistently within one call. If this ever fails, zeroing
        # the goal in PolicyFactory.create is no longer a safe substitute
        # for the real one, and this whole approach needs revisiting.
        real_goal = [5.0, 3.0, 0.9]
        state = np.array([1.2, -0.4, 0.3, 0.6, -0.2])
        real_sim = DynamicsFactory.create(
            system_name="unicycle2",
            config={"dt": DT, "max_accel": 2.0, "max_omega": 2.0, "max_speed": 2.0,
                    "goal": real_goal, "randomize_goal": False},
        )
        obs = real_sim.observe(state)
        # horizon must be long enough that the hard terminal-velocity==0
        # constraint is actually reachable from state's v=0.6 within
        # max_accel=2.0, dt=DT: decelerating needs >= 0.6 / (2.0 * DT) = 6
        # steps; 10 leaves margin.
        rng = np.random.default_rng(0)
        u_ref = rng.uniform(-1.0, 1.0, size=(2, 10))

        def project_with_anchor(anchor_goal: list[float]) -> np.ndarray:
            sim = DynamicsFactory.create(
                system_name="unicycle2",
                config={"dt": DT, "max_accel": 2.0, "max_omega": 2.0, "max_speed": 2.0,
                        "goal": anchor_goal, "randomize_goal": False},
            )
            projector = CasadiTrajectoryProjector(sim, {}, neighbor_slots=0, horizon=10, robot_index=0)
            return projector.project(obs, u_ref)

        u_safe_zeroed = project_with_anchor([0.0, 0.0, 0.0])
        u_safe_other = project_with_anchor([-7.0, 2.0, -1.4])
        np.testing.assert_allclose(u_safe_zeroed, u_safe_other, atol=1e-6)


class ProjectorFailClosedFallbackTests(unittest.TestCase):
    """Copilot review: returning the unprojected u_ref on any solve failure
    bypasses every constraint the projector exists to enforce. The fix falls
    back to the last known-safe trajectory (SafeFlowMPC Theorem 2) whenever
    one exists. On a true cold-start failure there is no such trajectory --
    exactly the situation the paper's Assumption 2 assumes away -- so this
    raises PlannerSolveError instead of fabricating a fallback with no
    safety basis; see PolicySolveFailureRecoveryTests for how callers are
    expected to handle that.
    """

    def setUp(self) -> None:
        self.simulator = _build_two_robot_simulator(goal0=[5.0, 0.0, 0.0], goal1=[-5.0, 0.0, 0.0])
        self.projector = CasadiTrajectoryProjector(
            self.simulator.simulators[0], {}, neighbor_slots=0, horizon=3, robot_index=0
        )
        self.state0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.obs0 = self.simulator.simulators[0].observe(self.state0)
        # Deliberately out of bounds (max_accel=max_omega=2.0): any solved
        # trajectory is guaranteed to differ from this, so a test asserting
        # "the fallback is NOT u_ref" can't pass by coincidence.
        self.u_ref = np.full((2, 3), 999.0)

    def test_falls_back_to_last_safe_trajectory_when_one_exists(self) -> None:
        self.projector.project(self.obs0, self.u_ref)
        self.assertIsNotNone(self.projector._prev_U_sol)
        expected_fallback = np.hstack(
            [self.projector._prev_U_sol[:, 1:], self.projector._prev_U_sol[:, -1:]]
        )[:, :3]

        self.projector.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        perturbed_obs = self.simulator.simulators[0].observe(self.state0 + 0.01)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback_action = self.projector.project(perturbed_obs, self.u_ref)

        np.testing.assert_allclose(fallback_action, expected_fallback)
        self.assertFalse(np.allclose(fallback_action, self.u_ref), "must not silently fall back to u_ref")
        self.assertTrue(any("last known-safe" in str(w.message) for w in caught))

    def test_raises_on_cold_start_failure_instead_of_returning_u_ref(self) -> None:
        # No previously-solved safe trajectory exists (first call ever, or
        # right after reset()) -- SafeFlowMPC's safety guarantee (Theorem 2)
        # assumes one always exists (Assumption 2), so there is no
        # paper-faithful fallback action here. Must fail loudly rather than
        # silently returning the unverified, unprojected u_ref.
        self.projector.reset()
        self.projector.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        with self.assertRaises(PlannerSolveError):
            self.projector.project(self.obs0, self.u_ref)


class PolicySolveFailureRecoveryTests(unittest.TestCase):
    """A raised PlannerSolveError from a policy's projector must not crash a
    whole rollout/evaluation run over one cold-start hiccup with no safe
    trajectory to fall back on. Callers treat it as an episode-ending
    failure, exactly like an expert-side PlannerSolveError -- verified here
    against the generic evaluation-side entry point (rollout_policy_with_
    action_fn); the collection-side entry point (collect_dagger_rollouts)
    falls back to that step's already-computed expert action instead, since
    unlike evaluation it always has the expert available.
    """

    def test_rollout_policy_with_action_fn_ends_episode_on_policy_solve_failure(self) -> None:
        from learning.dagger.rollouts import rollout_policy_with_action_fn

        simulator = _build_two_robot_simulator(goal0=[5.0, 0.0, 0.0], goal1=[-5.0, 0.0, 0.0])
        initial_state = simulator.reset_random()

        def failing_action_fn(observation: np.ndarray) -> np.ndarray:
            raise PlannerSolveError("forced failure for test")

        reached_goal, steps_taken = rollout_policy_with_action_fn(
            simulator=simulator,
            initial_state=initial_state,
            num_steps=10,
            action_fn=failing_action_fn,
        )
        self.assertFalse(reached_goal)
        # failing_action_fn raises on the very first call, before
        # simulator.step() ever runs -- zero steps were actually executed.
        self.assertEqual(steps_taken, 0)


if __name__ == "__main__":
    unittest.main()
