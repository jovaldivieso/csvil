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
        # horizon must be long enough for the projector's hard
        # terminal-velocity==0 constraint to be feasible from this
        # simulator's own worst-case velocity (max_speed/max_omega=2.0) at
        # max_accel=2.0, dt=DT=0.05: >= 2.0 / (2.0 * 0.05) = 20 steps.
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
            config={"dt": DT, "max_accel": 2.0, "max_omega": 2.0, "max_speed": 2.0,
                    "goal": real_goal, "randomize_goal": False},
        )
        obs = real_sim.observe(state)
        # horizon must clear CasadiTrajectoryProjector's own construction-time
        # feasibility check for the hard terminal-velocity==0 constraint,
        # which is based on this simulator's worst-case state bound
        # (max_speed=max_omega=2.0), not the actual state's v=0.6: needs
        # >= 2.0 / (2.0 * DT) = 20 steps; 22 leaves margin.
        rng = np.random.default_rng(0)
        u_ref = rng.uniform(-1.0, 1.0, size=(2, 22))

        def project_with_anchor(anchor_goal: list[float]) -> np.ndarray:
            sim = DynamicsFactory.create(
                system_name="unicycle2",
                config={"dt": DT, "max_accel": 2.0, "max_omega": 2.0, "max_speed": 2.0,
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
        # horizon must clear CasadiTrajectoryProjector's own construction-time
        # feasibility check for the hard terminal-velocity==0 constraint
        # (based on max_speed=max_omega=2.0, max_accel=2.0, dt=DT): needs
        # >= 2.0 / (2.0 * DT) = 20 steps; 22 leaves margin.
        self.projector = CasadiTrajectoryProjector(
            self.simulator.simulators[0], {}, neighbor_slots=0, horizon=22, robot_index=0
        )
        self.state0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.obs0 = self.simulator.simulators[0].observe(self.state0)
        # Deliberately out of bounds (max_accel=max_omega=2.0): any solved
        # trajectory is guaranteed to differ from this, so a test asserting
        # "the fallback is NOT u_ref" can't pass by coincidence.
        self.u_ref = np.full((2, 22), 999.0)

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
        # fallback's own terminal velocity at zero throughout.
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
        np.testing.assert_allclose(terminal_velocity, 0.0, atol=1e-9)


class TerminalConditionSupportTests(unittest.TestCase):
    """First-order systems need no terminal constraint at all: Assumption 1
    (a controller exists with a control-invariant safety set encompassing
    the terminal set) is trivially satisfied everywhere for a driftless
    first-order system (u=0 is a fixed point at any state), not skipped as
    a deviation from the paper. A too-short horizon for a velocity-having
    system, on the other hand, is caught at construction time rather than
    surfacing only as an opaque runtime NLP infeasibility.
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

    def test_too_short_horizon_for_velocity_having_system_raises_at_construction(self) -> None:
        sim = DynamicsFactory.create(
            system_name="unicycle2",
            config={"dt": DT, "max_accel": 2.0, "max_omega": 2.0, "max_speed": 2.0,
                    "goal": [1.0, 0.0, 0.0], "randomize_goal": False},
        )
        # 2.0 / (2.0 * DT) = 20 steps needed; 3 is far short.
        with self.assertRaises(ValueError):
            CasadiTrajectoryProjector(sim, {}, neighbor_slots=0, horizon=3, robot_index=0)


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

        neighbor_trajs = policy._build_neighbor_trajectories(observation_dict, x0_batch)

        # Robot 0 (the accelerating one) sees a genuinely stationary
        # neighbor -- its extrapolated trajectory must stay at (3, 0)
        # throughout, not drift at ~0.5*a*dt = 0.1 m/s the way back-
        # propagating with the current (not previous) frame's velocity
        # would have produced.
        robot0_neighbor_traj = neighbor_trajs[0, 0]
        np.testing.assert_allclose(robot0_neighbor_traj[0], 3.0, atol=1e-9)
        np.testing.assert_allclose(robot0_neighbor_traj[1], 0.0, atol=1e-9)


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
