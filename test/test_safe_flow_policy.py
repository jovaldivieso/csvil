from __future__ import annotations

import os
import sys
import unittest
import warnings

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.factory import DynamicsFactory
from learning.dagger import ObservationHistoryBuffer
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
                        "dt": DT, "max_linear_accel": 2.0, "max_angular_accel": 2.0, "max_angular_vel": 2.0, "max_linear_vel": 2.0,
                        "start": [0.0, 0.0, 0.0, 0.0, 0.0], "goal": goal0, "randomize_goal": False,
                    },
                },
                {
                    "system": "unicycle2",
                    "config": {
                        "dt": DT, "max_linear_accel": 2.0, "max_angular_accel": 2.0, "max_angular_vel": 2.0, "max_linear_vel": 2.0,
                        "start": [3.0, 3.0, 0.0, 0.0, 0.0], "goal": goal1, "randomize_goal": False,
                    },
                },
            ],
        },
    )


def _build_safe_flow_policy(simulator) -> SafeFlowMPCPolicy:
    # For 2 unicycle2 robots at observation_horizon=2: ego packs
    # environment_state(4, single-frame) + state(2*2=4, stacked) +
    # state_mask(1*2=2, stacked) = 10. DeepSetEncoder derives ego_dim as
    # state_dim - neighbor_slots*(neighbor_feature_dim + observation_horizon)
    # = state_dim - 1*(8+2), so state_dim must be 20 for ego_dim to equal
    # the actual 10 -- a mismatch here would only surface once something
    # actually calls select_action with this policy, not at construction.
    encoder = EncoderFactory.create(
        "deepset", state_dim=20, neighbor_feature_dim=8, neighbor_slots=1,
        observation_horizon=2, phi_dims=[8], rho_dims=[4],
    )
    policy = PolicyFactory.create(
        "safeflow",
        action_dim=2, obs_encoder=encoder, hidden_dims=[16], prediction_horizon=22, num_inference_steps=2,
        simulator=simulator, planner_config={},
    )
    assert isinstance(policy, SafeFlowMPCPolicy)
    return policy


class ProjectorInheritsFleetCollisionDistancesTests(unittest.TestCase):
    """d_safe/d_collision are attributes of the fleet (MultiRobotSim), not of
    the single-robot local_sims each projector actually holds -- so
    CasadiTrajectoryProjector's own config.get("d_safe", getattr(self.sim,
    "d_safe", 0.0)) fallback could never reach the fleet's real value on its
    own. Whenever planner_config didn't separately repeat d_safe/d_collision
    (as {} does here), every projector silently settled on 0.0, i.e. no
    collision avoidance at all. PolicyFactory.create now fills both in from
    the fleet simulator before constructing each projector, whenever
    planner_config doesn't already carry an explicit value of its own.
    """

    def test_empty_planner_config_still_inherits_fleet_d_safe_and_d_collision(self) -> None:
        simulator = _build_two_robot_simulator(goal0=[5.0, 0.0, 0.0], goal1=[-5.0, 0.0, 0.0])
        policy = _build_safe_flow_policy(simulator)

        for projector in policy.projectors:
            self.assertEqual(projector.d_safe, simulator.d_safe)
            self.assertEqual(projector.d_collision, simulator.d_collision)
        self.assertGreater(simulator.d_safe, 0.0)  # sanity: the fleet's own value isn't itself 0

    def test_explicit_planner_config_override_still_wins(self) -> None:
        simulator = _build_two_robot_simulator(goal0=[5.0, 0.0, 0.0], goal1=[-5.0, 0.0, 0.0])
        encoder = EncoderFactory.create(
            "deepset", state_dim=20, neighbor_feature_dim=8, neighbor_slots=1,
            observation_horizon=2, phi_dims=[8], rho_dims=[4],
        )
        policy = PolicyFactory.create(
            "safeflow",
            action_dim=2, obs_encoder=encoder, hidden_dims=[16], prediction_horizon=22, num_inference_steps=2,
            simulator=simulator, planner_config={"d_safe": 0.05, "d_collision": 0.05},
        )

        for projector in policy.projectors:
            self.assertEqual(projector.d_safe, 0.05)
            self.assertEqual(projector.d_collision, 0.05)


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
            config={"dt": DT, "max_linear_accel": 2.0, "max_angular_accel": 2.0, "max_angular_vel": 2.0, "max_linear_vel": 2.0,
                    "goal": real_goal, "randomize_goal": False},
        )
        obs = real_sim.observe(state)
        rng = np.random.default_rng(0)
        u_ref = rng.uniform(-1.0, 1.0, size=(2, 22))

        def project_with_anchor(anchor_goal: list[float]) -> np.ndarray:
            sim = DynamicsFactory.create(
                system_name="unicycle2",
                config={"dt": DT, "max_linear_accel": 2.0, "max_angular_accel": 2.0, "max_angular_vel": 2.0, "max_linear_vel": 2.0,
                        "goal": anchor_goal, "randomize_goal": False},
            )
            projector = CasadiTrajectoryProjector(sim, {}, neighbor_slots=0, horizon=22, robot_index=0)
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
        # An amplified terminal_velocity_weight (default is 1.0) makes the
        # projector actually settle near rest for this test's constant,
        # in-bounds u_ref: the terminal-rest condition is a soft cost now,
        # not a hard constraint, so with the default weight a merely
        # in-bounds (not adversarially extreme) reference still mostly wins
        # out over resting -- these tests need a genuinely near-rest cached
        # trajectory to fall back on, which real (trained, not synthetic)
        # u_ref values get "for free" simply by not asking for sustained
        # motion right up to the horizon's edge.
        self.projector = CasadiTrajectoryProjector(
            self.simulator.simulators[0], {"terminal_velocity_weight": 500.0},
            neighbor_slots=0, horizon=22, robot_index=0,
        )
        self.state0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.obs0 = self.simulator.simulators[0].observe(self.state0)
        # A realistic, in-bounds constant (not the old out-of-bounds 999):
        # an out-of-bounds/extreme reference dominates the (now merely soft)
        # terminal cost by construction, at any finite weight, so the
        # projector no longer reliably settles near rest under one -- not a
        # bug, just incompatible with a soft cost. The solved trajectory
        # still ends up clearly different from this reference (the terminal
        # cost pulls it back toward ~0, nowhere near the constant 1.0), so
        # "solved trajectory != u_ref" remains verified below.
        self.u_ref = np.full((2, 22), 1.0)

    def test_falls_back_to_last_safe_trajectory_when_one_exists(self) -> None:
        self.projector.project(self.obs0, self.u_ref)
        self.assertIsNotNone(self.projector._prev_U_sol)
        # Matches project()'s own fallback padding: the shifted plan's
        # extra final slot is zero action (holds the certified-at-rest
        # terminal state), not a duplicate of the last applied action.
        expected_fallback = np.hstack(
            [self.projector._prev_U_sol[:, 1:], np.zeros((self.projector.sim.nu, 1))]
        )[:, :22]

        self.projector.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        # 0.005, not fallback_terminal_velocity_tol's own 0.01: the soft
        # terminal cost only pulls the cached solve *close to* rest, not to
        # literal machine-precision zero the way the old hard equality
        # constraint did, so perturbing by exactly the tolerance would make
        # the residual's sign (which side of the boundary) coincidental.
        perturbed_obs = self.simulator.simulators[0].observe(self.state0 + 0.005)
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

    def test_consecutive_fallbacks_stay_at_rest_instead_of_drifting_away(self) -> None:
        # Regression test for padding the fallback's extra control slot with
        # a duplicate of the *last applied* action (whatever nonzero
        # deceleration drove velocity to zero) instead of zero (the action
        # that actually sustains the certified at-rest terminal state).
        # Reapplying a stale nonzero action to an already-at-rest state
        # would accelerate it right back away from rest -- exactly the
        # "failure recovery does not establish an invariant resting
        # trajectory" gap. A run of consecutive failures across genuinely
        # different ticks (a new x0 each time, so each hits the shift-and-
        # pad branch, not the same-tick reuse-as-is branch) must keep the
        # fallback's own terminal velocity bounded, not drifting further
        # away with each additional fallback.
        self.projector.project(self.obs0, self.u_ref)
        self.projector.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(1, 6):
                # Perturb position/theta only, leaving velocity exactly at
                # the certified-zero terminal value -- isolates the padding
                # recipe itself from _revalidate_fallback's separate,
                # already-tested tolerance for genuine x0 velocity drift.
                perturbed_state = self.state0.copy()
                perturbed_state[:3] += 0.001 * i
                obs_i = self.simulator.simulators[0].observe(perturbed_state)
                self.projector.project(obs_i, self.u_ref)
        terminal_velocity = self.projector._prev_X_sol[
            list(self.projector.velocity_idx), self.projector.N
        ]
        # Not 0.0 to machine precision: the terminal-rest condition is a
        # soft cost now, so the cached solve only ever settles *close to*
        # rest, never exactly -- fallback_terminal_velocity_tol is precisely
        # the bound _revalidate_fallback itself already enforces on this
        # quantity every time it accepts a fallback, so bounding by it here
        # is the actual safety-relevant assertion, not an arbitrary
        # tolerance choice.
        np.testing.assert_allclose(
            terminal_velocity, 0.0, atol=self.projector.fallback_terminal_velocity_tol
        )

    def test_exceeding_max_consecutive_fallbacks_raises_instead_of_coasting_indefinitely(self) -> None:
        # A fallback accepted within fallback_terminal_velocity_tol is
        # coasting at a small residual velocity, not genuinely at rest (it's
        # re-simulated under zero action, which doesn't decay that residual
        # further); each individual reuse is only checked through its own
        # horizon. An unbroken run of them would otherwise let that drift
        # keep accumulating for as long as real solves keep failing.
        # max_consecutive_fallbacks bounds it: one more consecutive fallback
        # than the default (5) must raise rather than be accepted.
        self.assertEqual(self.projector.max_consecutive_fallbacks, 5)
        self.projector.project(self.obs0, self.u_ref)
        self.projector.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(1, 6):
                perturbed_state = self.state0.copy()
                perturbed_state[:3] += 0.001 * i
                obs_i = self.simulator.simulators[0].observe(perturbed_state)
                self.projector.project(obs_i, self.u_ref)

            sixth_state = self.state0.copy()
            sixth_state[:3] += 0.006
            sixth_obs = self.simulator.simulators[0].observe(sixth_state)
            with self.assertRaises(PlannerSolveError):
                self.projector.project(sixth_obs, self.u_ref)

    def test_repeated_same_tick_calls_do_not_each_consume_the_fallback_budget(self) -> None:
        # select_action() calls project() once per flow/Euler denoising step
        # -- several times per real control tick, all at the *same* x0 (only
        # u_ref changes as the flow network refines its guess). None of
        # those represent additional elapsed time or drift, so repeating
        # far more of them than max_consecutive_fallbacks allows must not
        # raise, as long as it's genuinely the same tick throughout.
        self.assertEqual(self.projector.max_consecutive_fallbacks, 5)
        self.projector.project(self.obs0, self.u_ref)
        self.projector.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(20):  # far more than max_consecutive_fallbacks (5)
                self.projector.project(self.obs0, self.u_ref)
        self.assertEqual(self.projector._consecutive_fallback_count, 0)


class NeighborActiveGateTests(unittest.TestCase):
    """A masked-out neighbor slot was represented purely by moving
    neighbor_trajs to a numerically-far-away sentinel (1e6, 1e6), so the
    hard/soft collision constraints stayed structurally present in the NLP
    and were only satisfied because that position happens to be far outside
    any realistic workspace. A misconfigured (or, in principle, a
    fixed-state/workspace value that happened to coincide with it) scenario
    could make an absent neighbor register as a real, unavoidable obstacle.
    neighbor_active now gates the constraint threshold itself (0 -> squared
    distance >= 0, trivially true for *any* position) so masking no longer
    depends on where neighbor_trajs happens to point.
    """

    def setUp(self) -> None:
        self.sim = DynamicsFactory.create(
            system_name="unicycle2",
            config={"dt": DT, "max_linear_accel": 2.0, "max_angular_accel": 2.0, "max_angular_vel": 2.0, "max_linear_vel": 2.0,
                    "goal": [2.0, 0.0, 0.0], "randomize_goal": False},
        )
        self.projector = CasadiTrajectoryProjector(
            self.sim, {"d_safe": 0.5, "d_collision": 0.3}, neighbor_slots=1, horizon=22, robot_index=0,
        )
        self.state0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.obs0 = self.sim.observe(self.state0)
        # Symmetric bang-bang reference (accelerate then decelerate back to
        # zero over the 22-step horizon) -- unlike u_ref=0, this actually
        # gives the optimizer somewhere to go, so the phantom neighbor below
        # can meaningfully be "in the way" or not.
        self.u_ref = np.zeros((2, 22))
        self.u_ref[0, :11] = 2.0
        self.u_ref[0, 11:] = -2.0
        # A "neighbor" sitting almost exactly on that unconstrained path
        # (probed empirically: x ~= 0.42 at k=14) for the whole horizon --
        # well within d_collision of it unless actively avoided.
        self.neighbor_trajs = np.tile(np.array([[0.42], [0.0]]), (1, 1, 23))

    def _min_distance_to_phantom_neighbor(self, u_safe: np.ndarray) -> float:
        state = self.state0.copy()
        positions = [state[:2]]
        for k in range(u_safe.shape[1]):
            state = self.sim.predict_next_state(state, u_safe[:, k], validate=False)
            positions.append(state[:2].copy())
        positions = np.array(positions)
        return float(np.min(np.linalg.norm(positions - np.array([0.42, 0.0]), axis=1)))

    def test_masked_neighbor_is_ignored_regardless_of_its_position(self) -> None:
        u_safe = self.projector.project(
            self.obs0, self.u_ref, self.neighbor_trajs, neighbor_active=np.array([0.0]),
        )
        # Free to pass right by (0.42, 0) since the slot is inactive --
        # tracks u_ref's own unconstrained path through that point.
        self.assertLess(self._min_distance_to_phantom_neighbor(u_safe), 0.3)

    def test_active_neighbor_at_the_same_position_is_avoided(self) -> None:
        self.projector.reset()
        u_safe = self.projector.project(
            self.obs0, self.u_ref, self.neighbor_trajs, neighbor_active=np.array([1.0]),
        )
        self.assertGreaterEqual(self._min_distance_to_phantom_neighbor(u_safe), 0.3 - 1e-6)

    def test_omitting_neighbor_active_defaults_to_all_active(self) -> None:
        # Backward compatibility: a caller that doesn't pass neighbor_active
        # (the pre-existing signature) must still get full constraint
        # enforcement, not silently-disabled ones.
        u_safe = self.projector.project(self.obs0, self.u_ref, self.neighbor_trajs)
        self.assertGreaterEqual(self._min_distance_to_phantom_neighbor(u_safe), 0.3 - 1e-6)


class TerminalConditionSupportTests(unittest.TestCase):
    """First-order systems need no terminal constraint at all: Assumption 1
    (a controller exists with a control-invariant safety set encompassing
    the terminal set) is trivially satisfied everywhere for a driftless
    first-order system (u=0 is a fixed point at any state), not skipped as
    a deviation from the paper. Velocity-having systems get a soft terminal-
    velocity cost instead of a hard equality constraint: a hard constraint
    is only satisfiable when the horizon leaves no margin beyond the
    system's own worst-case braking distance for anything else (u_ref
    tracking, collision avoidance) sharing that same horizon, which made
    the projector stall rather than progress even in ordinary scenarios --
    so a too-short horizon here just weakens the soft cost, it doesn't risk
    outright infeasibility.
    """

    def test_first_order_system_constructs_and_solves_without_a_terminal_constraint(self) -> None:
        sim = DynamicsFactory.create(
            system_name="single_integrator",
            config={"dt": DT, "max_vel": 1.0, "goal": [1.0, 1.0], "randomize_goal": False},
        )
        projector = CasadiTrajectoryProjector(sim, {}, neighbor_slots=0, horizon=3, robot_index=0)
        self.assertEqual(projector.velocity_idx, ())

        state0 = np.array([0.0, 0.0])
        obs0 = sim.observe(state0)
        # Deliberately large: with no terminal constraint to fight, the
        # solver should simply saturate toward it every step.
        u_ref = np.full((2, 3), 999.0)
        u_safe = projector.project(obs0, u_ref)
        np.testing.assert_allclose(u_safe, sim.max_action, atol=1e-6)

    def test_short_horizon_for_velocity_having_system_constructs_and_solves(self) -> None:
        # A horizon far shorter than the braking distance at max speed
        # (2.0 / (2.0 * DT) = 20 steps) must not raise: the terminal
        # velocity term is a soft cost, so it's always solvable, just
        # weakly enforced from far-from-rest states.
        sim = DynamicsFactory.create(
            system_name="unicycle2",
            config={"dt": DT, "max_linear_accel": 2.0, "max_angular_accel": 2.0, "max_angular_vel": 2.0, "max_linear_vel": 2.0,
                    "goal": [1.0, 0.0, 0.0], "randomize_goal": False},
        )
        projector = CasadiTrajectoryProjector(sim, {}, neighbor_slots=0, horizon=3, robot_index=0)
        state0 = np.array([0.0, 0.0, 0.0, 2.0, 0.0])
        obs0 = sim.observe(state0)
        u_ref = np.zeros((2, 3))
        projector.project(obs0, u_ref)


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

        reached_goal, steps_taken, failure_reason = rollout_policy_with_action_fn(
            simulator=simulator,
            initial_state=initial_state,
            num_steps=10,
            action_fn=failing_action_fn,
        )
        self.assertFalse(reached_goal)
        # failing_action_fn raises on the very first call, before
        # simulator.step() ever runs -- zero steps were actually executed.
        self.assertEqual(steps_taken, 0)
        self.assertEqual(failure_reason, "solve_failure")


class FirstOrderMultiRobotRejectionTests(unittest.TestCase):
    """A velocity-less system (single_integrator, unicycle1) gives
    _build_neighbor_trajectories no state from which to recover the ego's
    own previous absolute position, so it could only ever report every
    neighbor as momentarily stationary -- never bounding an actually
    approaching neighbor's motion, regardless of how close or fast it's
    closing. That's a silent, unconditional safety gap for the entire
    episode, not a one-tick warm-up artifact, so SafeFlowMPCPolicy rejects
    this combination outright at construction time rather than shipping a
    policy whose neighbor forecast cannot back its own hard d_collision
    constraint. A lone first-order robot (no neighbors to forecast at all)
    is unaffected.
    """

    def test_multi_robot_first_order_fleet_is_rejected_at_construction(self) -> None:
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": DT,
                "d_safe": 0.1,
                "robots": [
                    {"system": "single_integrator", "config": {
                        "dt": DT, "max_vel": 5.0, "goal": [0.0, 0.0], "randomize_goal": False,
                    }},
                    {"system": "single_integrator", "config": {
                        "dt": DT, "max_vel": 5.0, "goal": [0.0, 0.0], "randomize_goal": False,
                    }},
                ],
            },
        )
        encoder = EncoderFactory.create(
            "deepset", state_dim=10, neighbor_feature_dim=4, neighbor_slots=1,
            observation_horizon=2, phi_dims=[8], rho_dims=[4],
        )

        with self.assertRaises(ValueError):
            PolicyFactory.create(
                "safeflow",
                action_dim=2, obs_encoder=encoder, hidden_dims=[16], prediction_horizon=3,
                num_inference_steps=2, simulator=simulator, planner_config={},
            )

    def test_single_first_order_robot_is_unaffected(self) -> None:
        # Exercises SafeFlowMPCPolicy.__init__'s guard directly (bypassing
        # PolicyFactory.create, which would otherwise need a real FlowPolicy
        # + neighbor-aware encoder just to reach it) since DeepSetEncoder
        # itself always requires neighbor_feature_dim > 0, independent of
        # this policy's own neighbor_slots -- an unrelated constraint that a
        # single-robot config wouldn't pair with "deepset" in practice.
        sim = DynamicsFactory.create(
            system_name="single_integrator",
            config={"dt": DT, "max_vel": 5.0, "goal": [0.0, 0.0], "randomize_goal": False},
        )
        projector = CasadiTrajectoryProjector(sim, {}, neighbor_slots=0, horizon=3, robot_index=0)

        policy = SafeFlowMPCPolicy(inner_policy=None, projectors=[projector], local_sims=[sim])
        self.assertEqual(policy.neighbor_slots, 0)

    def test_multi_robot_fleet_with_single_observation_frame_is_rejected(self) -> None:
        # A velocity-having fleet hits the exact same gap if observation_
        # horizon == 1: _build_neighbor_trajectories can only estimate a
        # neighbor's velocity by differencing two consecutive frames, so one
        # frame alone forecasts every neighbor as stationary regardless of
        # this robot's own velocity_state_indices.
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": DT,
                "d_safe": 0.1,
                "robots": [
                    {"system": "unicycle2", "config": {
                        "dt": DT, "max_linear_accel": 2.0, "max_angular_accel": 2.0, "max_angular_vel": 2.0, "max_linear_vel": 2.0,
                        "goal": [0.0, 0.0, 0.0], "randomize_goal": False,
                    }},
                    {"system": "unicycle2", "config": {
                        "dt": DT, "max_linear_accel": 2.0, "max_angular_accel": 2.0, "max_angular_vel": 2.0, "max_linear_vel": 2.0,
                        "goal": [0.0, 0.0, 0.0], "randomize_goal": False,
                    }},
                ],
            },
        )
        encoder = EncoderFactory.create(
            "deepset", state_dim=10, neighbor_feature_dim=4, neighbor_slots=1,
            observation_horizon=1, phi_dims=[8], rho_dims=[4],
        )

        with self.assertRaises(ValueError):
            PolicyFactory.create(
                "safeflow",
                action_dim=2, obs_encoder=encoder, hidden_dims=[16], prediction_horizon=22,
                num_inference_steps=2, simulator=simulator, planner_config={},
            )


class OneRobotMultiRobotFleetUnwrappingTests(unittest.TestCase):
    """A one-robot multi_robot configuration is valid (num_robots=1 is not
    rejected by config validation), but PolicyFactory.create used to decide
    whether to unwrap the fleet's sole sub-simulator based on num_robots > 1
    rather than on whether `simulator` *is* a fleet wrapper at all. For
    num_robots == 1 that kept the MultiRobotSimulator wrapper itself as
    local_sims[0] -- which reports empty velocity_state_indices regardless
    of the wrapped system, silently disabling the terminal-rest cost
    entirely for a system that actually has one.
    """

    def test_single_wrapped_velocity_having_robot_unwraps_to_its_own_sim(self) -> None:
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": DT,
                "d_safe": 0.1,
                "robots": [
                    {"system": "unicycle2", "config": {
                        "dt": DT, "max_linear_accel": 2.0, "max_angular_accel": 2.0, "max_angular_vel": 2.0, "max_linear_vel": 2.0,
                        "goal": [1.0, 0.0, 0.0], "randomize_goal": False,
                    }},
                ],
            },
        )
        self.assertEqual(simulator.num_robots, 1)
        encoder = EncoderFactory.create(
            "deepset", state_dim=10, neighbor_feature_dim=4, neighbor_slots=1,
            observation_horizon=2, phi_dims=[8], rho_dims=[4],
        )

        policy = PolicyFactory.create(
            "safeflow",
            action_dim=2, obs_encoder=encoder, hidden_dims=[16], prediction_horizon=22, num_inference_steps=2,
            simulator=simulator, planner_config={},
        )

        from systems.unicycle2 import Unicycle2

        self.assertIsInstance(policy.local_sims[0], Unicycle2)
        self.assertEqual(policy.local_sims[0].velocity_state_indices, (3, 4))


class VelocityHavingNeighborVelocityUnderAccelerationTests(unittest.TestCase):
    """Regression test: for velocity-having systems (double_integrator,
    unicycle2), _build_neighbor_trajectories previously back-propagated the
    ego's own previous position/heading using its *current* frame's
    velocity/turn-rate, e.g. pos_prev = pos_now - dt * v_now for
    double_integrator. DoubleIntegrator.predict_next_state actually advances
    position via next_pos = pos + v*dt + 0.5*a*dt**2 using the *previous*
    frame's velocity, so whenever the ego genuinely accelerates between
    frames, that shortcut folds part of the ego's own acceleration into a
    supposedly-stationary neighbor's estimated velocity. Fixed by reading
    the actual previous frame from the already-stacked observation.state
    instead of re-deriving it from the current one.
    """

    def test_stationary_neighbor_velocity_unbiased_by_egos_own_acceleration(self) -> None:
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": DT,
                "d_safe": 0.1,
                "robots": [
                    {"system": "double_integrator", "config": {
                        "dt": DT, "max_accel": 10.0, "goal": [0.0, 0.0], "randomize_goal": False,
                    }},
                    {"system": "double_integrator", "config": {
                        "dt": DT, "max_accel": 10.0, "goal": [0.0, 0.0], "randomize_goal": False,
                    }},
                ],
            },
        )
        encoder = EncoderFactory.create(
            "deepset", state_dim=10, neighbor_feature_dim=4, neighbor_slots=1,
            observation_horizon=2, phi_dims=[8], rho_dims=[4],
        )
        policy = PolicyFactory.create(
            "safeflow",
            action_dim=2, obs_encoder=encoder, hidden_dims=[16], prediction_horizon=3, num_inference_steps=2,
            simulator=simulator, planner_config={},
        )

        history_buffer = ObservationHistoryBuffer(2, 2)

        def observe_and_stack(ego_state: np.ndarray, neighbor_state: np.ndarray):
            state = np.concatenate([ego_state, neighbor_state])
            simulator.reset(state)
            full_obs = simulator.observe(state)
            return [
                history_buffer.append_and_stack(r, simulator.decentralized_policy_observation(full_obs, r))
                for r in range(2)
            ]

        # Tick 1: ego (robot 0) at (-1, 0) moving at (2, 0). Applying a
        # genuine acceleration of (4, 0) for one step -- computed via the
        # real dynamics, not by hand, so tick 2 is exactly consistent with
        # predict_next_state's own update rule -- takes it to tick 2 with a
        # *different* velocity (2.2, 0), not merely a different position.
        # The neighbor (robot 1) stays fixed at (3, 0), v=(0, 0) throughout.
        ego_tick1 = np.array([-1.0, 0.0, 2.0, 0.0])
        ego_tick2 = simulator.simulators[0].predict_next_state(
            ego_tick1, np.array([4.0, 0.0]), validate=False
        )
        neighbor_state = np.array([3.0, 0.0, 0.0, 0.0])

        observe_and_stack(ego_tick1, neighbor_state)
        stacked = observe_and_stack(ego_tick2, neighbor_state)

        observation_dict = {
            name: torch.as_tensor(np.stack([stacked[r][name] for r in range(2)]), dtype=torch.float32)
            for name in stacked[0]
        }
        ego_obs_np = policy._extract_ego_observation(observation_dict)
        x0_batch = np.stack([policy.local_sims[b].invert_obs(ego_obs_np[b]) for b in range(2)])

        neighbor_trajs, _ = policy._build_neighbor_trajectories(observation_dict, x0_batch)

        # Robot 0 (the accelerating one) sees a genuinely stationary
        # neighbor -- its extrapolated trajectory must stay at (3, 0)
        # throughout, not drift at ~0.5*a*dt = 0.1 m/s the way back-
        # propagating with the current (not previous) frame's velocity
        # would have produced.
        robot0_neighbor_traj = neighbor_trajs[0, 0]
        np.testing.assert_allclose(robot0_neighbor_traj[0], 3.0, atol=1e-9)
        np.testing.assert_allclose(robot0_neighbor_traj[1], 0.0, atol=1e-9)


class HeterogeneousFleetNeighborForwardSimulationTests(unittest.TestCase):
    """Regression test: the unicycle-shaped (arc) neighbor-forecast branch of
    _build_neighbor_trajectories always forward-simulated every neighbor
    with local_sims[0] -- robot 0's own sim object -- regardless of which
    fleet robot that neighbor slot actually was. The fleet contract only
    requires equal type/dimensions across robots, not equal numeric
    dynamics parameters, so a neighbor with a higher max_linear_vel than
    robot 0 had its forecast velocity silently clipped down to robot 0's own
    limit, understating how fast it can actually close distance.
    """

    def test_faster_neighbors_own_max_linear_vel_is_used_not_robot_zeros(self) -> None:
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": DT,
                "d_safe": 0.1,
                "robots": [
                    {"system": "unicycle2", "config": {
                        "dt": DT, "max_linear_accel": 2.0, "max_angular_accel": 2.0, "max_angular_vel": 2.0, "max_linear_vel": 2.0,
                        "goal": [0.0, 0.0, 0.0], "randomize_goal": False,
                    }},
                    {"system": "unicycle2", "config": {
                        "dt": DT, "max_linear_accel": 2.0, "max_angular_accel": 2.0, "max_angular_vel": 2.0, "max_linear_vel": 10.0,
                        "goal": [0.0, 0.0, 0.0], "randomize_goal": False,
                    }},
                ],
            },
        )
        encoder = EncoderFactory.create(
            "deepset", state_dim=20, neighbor_feature_dim=8, neighbor_slots=1,
            observation_horizon=2, phi_dims=[8], rho_dims=[4],
        )
        # Generously long: robot 1's max_linear_vel=10.0 is much faster than
        # robot 0's, and this test isolates neighbor-forecast correctness,
        # not horizon sizing.
        policy = PolicyFactory.create(
            "safeflow",
            action_dim=2, obs_encoder=encoder, hidden_dims=[16], prediction_horizon=100, num_inference_steps=2,
            simulator=simulator, planner_config={},
        )

        history_buffer = ObservationHistoryBuffer(2, 2)

        def observe_and_stack(robot0_state: np.ndarray, robot1_state: np.ndarray):
            state = np.concatenate([robot0_state, robot1_state])
            simulator.reset(state)
            full_obs = simulator.observe(state)
            return [
                history_buffer.append_and_stack(r, simulator.decentralized_policy_observation(full_obs, r))
                for r in range(2)
            ]

        # Robot 0 (the observer, max_linear_vel=2.0) sits still at the origin
        # the whole time, isolating this from its own pos_prev
        # reconstruction. Robot 1 (max_linear_vel=10.0) travels in a
        # straight line at v=8.0 -- only possible under its *own* limit,
        # never robot 0's.
        robot0_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        robot1_tick1 = np.array([-2.0, -5.0, 0.0, 8.0, 0.0])
        robot1_tick2 = simulator.simulators[1].predict_next_state(
            robot1_tick1, np.array([0.0, 0.0]), validate=False
        )

        observe_and_stack(robot0_state, robot1_tick1)
        stacked = observe_and_stack(robot0_state, robot1_tick2)

        observation_dict = {
            name: torch.as_tensor(np.stack([stacked[r][name] for r in range(2)]), dtype=torch.float32)
            for name in stacked[0]
        }
        ego_obs_np = policy._extract_ego_observation(observation_dict)
        x0_batch = np.stack([policy.local_sims[b].invert_obs(ego_obs_np[b]) for b in range(2)])

        neighbor_trajs, _ = policy._build_neighbor_trajectories(observation_dict, x0_batch)

        # Robot 0's forecast of robot 1 must keep advancing at v=8.0 (its
        # own max_linear_vel) for the whole horizon -- forward-simulating
        # with robot 0's sim (max_linear_vel=2.0) would clip it down to 2.0
        # after the very first step.
        robot0s_view_of_robot1 = neighbor_trajs[0, 0, 0, :]  # x-coordinate over the horizon
        step_size = robot0s_view_of_robot1[2] - robot0s_view_of_robot1[1]
        np.testing.assert_allclose(step_size, 8.0 * DT, atol=1e-6)


class EgoRotationNeighborVelocityConsistencyTests(unittest.TestCase):
    """_build_neighbor_trajectories reconstructs each neighbor's absolute
    position at both history instants by rotating that instant's ego-relative
    reading through *that instant's own* ego heading (theta_prev for the
    older frame, theta_now for the newer one) before differencing. Every
    other neighbor-forecast test in this file holds the ego's heading fixed
    (theta stays 0 throughout), so none of them can catch a regression that
    collapses back to a single shared heading for both instants -- that
    shortcut is invisible whenever the ego doesn't actually turn between
    frames, since theta_prev == theta_now trivially, and would otherwise
    leak the ego's own rotation into every neighbor's estimated velocity as
    a spurious tangential component proportional to the neighbor's distance.
    """

    def test_stationary_neighbor_stays_stationary_while_ego_only_rotates(self) -> None:
        simulator = _build_two_robot_simulator(goal0=[0.0, 0.0, 0.0], goal1=[0.0, 0.0, 0.0])
        policy = _build_safe_flow_policy(simulator)
        history_buffer = ObservationHistoryBuffer(2, 2)

        def observe_and_stack(ego_state: np.ndarray, neighbor_state: np.ndarray):
            state = np.concatenate([ego_state, neighbor_state])
            simulator.reset(state)
            full_obs = simulator.observe(state)
            return [
                history_buffer.append_and_stack(r, simulator.decentralized_policy_observation(full_obs, r))
                for r in range(2)
            ]

        # Ego (robot 0) spins in place -- v=0 throughout, so its position
        # never moves, but omega=2.0 turns its heading by omega*dt=0.1 rad
        # between the two frames. The neighbor (robot 1) is genuinely
        # stationary at (5, 0). Any use of a single shared heading to
        # interpret both frames' ego-relative readings would misattribute
        # that 0.1 rad of pure ego rotation to neighbor motion.
        ego_tick1 = np.array([0.0, 0.0, 0.0, 0.0, 2.0])
        ego_tick2 = simulator.simulators[0].predict_next_state(
            ego_tick1, np.array([0.0, 0.0]), validate=False
        )
        neighbor_state = np.array([5.0, 0.0, 0.0, 0.0, 0.0])

        observe_and_stack(ego_tick1, neighbor_state)
        stacked = observe_and_stack(ego_tick2, neighbor_state)

        observation_dict = {
            name: torch.as_tensor(np.stack([stacked[r][name] for r in range(2)]), dtype=torch.float32)
            for name in stacked[0]
        }
        ego_obs_np = policy._extract_ego_observation(observation_dict)
        x0_batch = np.stack([policy.local_sims[b].invert_obs(ego_obs_np[b]) for b in range(2)])

        neighbor_trajs, _ = policy._build_neighbor_trajectories(observation_dict, x0_batch)

        # atol reflects ObservationHistoryBuffer's float32 storage (this
        # test's nonzero heading puts real cos/sin roundoff in play, unlike
        # the theta=0 cases elsewhere in this file where cos(0)/sin(0) are
        # exact) -- several orders of magnitude tighter than the >0.1 m
        # error a lost-rotation regression would produce here.
        robot0_neighbor_traj = neighbor_trajs[0, 0]
        np.testing.assert_allclose(robot0_neighbor_traj[0], 5.0, atol=1e-4)
        np.testing.assert_allclose(robot0_neighbor_traj[1], 0.0, atol=1e-4)

    def test_moving_neighbors_constant_global_velocity_survives_ego_turn(self) -> None:
        simulator = _build_two_robot_simulator(goal0=[0.0, 0.0, 0.0], goal1=[0.0, 0.0, 0.0])
        policy = _build_safe_flow_policy(simulator)
        history_buffer = ObservationHistoryBuffer(2, 2)

        def observe_and_stack(ego_state: np.ndarray, neighbor_state: np.ndarray):
            state = np.concatenate([ego_state, neighbor_state])
            simulator.reset(state)
            full_obs = simulator.observe(state)
            return [
                history_buffer.append_and_stack(r, simulator.decentralized_policy_observation(full_obs, r))
                for r in range(2)
            ]

        # Ego (robot 0) both moves and turns: v=1.0, omega=2.0 advances its
        # heading by 0.1 rad and its position along the arc between frames.
        # The neighbor (robot 1) drives in a straight line at a known global
        # velocity (1.2, 0.9) m/s -- comfortably under both robots'
        # max_linear_vel=2.0 fixture limit, since exceeding it would have
        # the forward-simulation loop's own predict_next_state call clip the
        # speed on its very first step (correct, physical behavior -- see
        # HeterogeneousFleetNeighborForwardSimulationTests -- but a
        # confound for this test, which isolates rotation/timing instead).
        # omega=0 keeps its own heading/speed constant, so real unicycle
        # dynamics via predict_next_state (not a hand-picked position) give
        # a per-step ground-truth displacement of exactly (1.2, 0.9) * DT
        # regardless of what the ego is doing.
        ego_tick1 = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        ego_tick2 = simulator.simulators[0].predict_next_state(
            ego_tick1, np.array([0.0, 0.0]), validate=False
        )
        neighbor_heading = np.arctan2(0.9, 1.2)
        neighbor_speed = float(np.hypot(1.2, 0.9))
        neighbor_tick1 = np.array([10.0, 10.0, neighbor_heading, neighbor_speed, 0.0])
        neighbor_tick2 = simulator.simulators[1].predict_next_state(
            neighbor_tick1, np.array([0.0, 0.0]), validate=False
        )

        observe_and_stack(ego_tick1, neighbor_tick1)
        stacked = observe_and_stack(ego_tick2, neighbor_tick2)

        observation_dict = {
            name: torch.as_tensor(np.stack([stacked[r][name] for r in range(2)]), dtype=torch.float32)
            for name in stacked[0]
        }
        ego_obs_np = policy._extract_ego_observation(observation_dict)
        x0_batch = np.stack([policy.local_sims[b].invert_obs(ego_obs_np[b]) for b in range(2)])

        neighbor_trajs, _ = policy._build_neighbor_trajectories(observation_dict, x0_batch)

        robot0s_view_of_robot1 = neighbor_trajs[0, 0]  # (2, horizon + 1): x row, y row
        # "Now" (index 0) must already land on the neighbor's true tick-2
        # position, and every subsequent step must keep advancing by the
        # same known ground-truth displacement -- straight-line motion, so
        # the arc-forecast collapses to a constant step exactly.
        np.testing.assert_allclose(robot0s_view_of_robot1[:, 0], [10.06, 10.045], atol=1e-4)
        step = robot0s_view_of_robot1[:, 2] - robot0s_view_of_robot1[:, 1]
        np.testing.assert_allclose(step, [1.2 * DT, 0.9 * DT], atol=1e-4)


class SelectActionBatchSizeGuardTests(unittest.TestCase):
    """select_action indexes self.local_sims[b]/self.projectors[b] directly
    by batch position, assuming it *is* the fleet's own robot ordering (see
    build_decentralized_joint_action, the only real caller, which always
    builds exactly one row per robot in that order). A batch of any other
    size must be rejected explicitly rather than silently applying the
    wrong robot's projector/weights (too small a batch) or crashing with an
    opaque IndexError (too large).
    """

    @staticmethod
    def _build_observation_dict(simulator) -> dict[str, torch.Tensor]:
        history_buffer = ObservationHistoryBuffer(2, 2)
        state = simulator.reset_random()
        full_obs = simulator.observe(state)
        stacked = [
            history_buffer.append_and_stack(r, simulator.decentralized_policy_observation(full_obs, r))
            for r in range(2)
        ]
        return {
            name: torch.as_tensor(np.stack([stacked[r][name] for r in range(2)]), dtype=torch.float32)
            for name in stacked[0]
        }

    def test_full_fleet_batch_succeeds(self) -> None:
        simulator = _build_two_robot_simulator(goal0=[5.0, 0.0, 0.0], goal1=[-5.0, 0.0, 0.0])
        policy = _build_safe_flow_policy(simulator)
        observation_dict = self._build_observation_dict(simulator)

        action = policy.select_action(observation_dict)
        self.assertEqual(action.shape[0], 2)

    def test_mismatched_batch_size_raises_instead_of_silently_misapplying_projectors(self) -> None:
        simulator = _build_two_robot_simulator(goal0=[5.0, 0.0, 0.0], goal1=[-5.0, 0.0, 0.0])
        policy = _build_safe_flow_policy(simulator)
        observation_dict = self._build_observation_dict(simulator)
        truncated = {name: tensor[:1] for name, tensor in observation_dict.items()}

        with self.assertRaises(ValueError):
            policy.select_action(truncated)


if __name__ == "__main__":
    unittest.main()
