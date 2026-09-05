import os
import sys
import unittest
import casadi as ca
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from core.factory import DynamicsFactory
from core.config import ConfigurationError, validate_system_config
from planning.casadi_planner import CasadiPlanner
from systems.dynamics import DynamicsProtocol
from systems.multi_robot import MultiRobotSimulator


def _build_simulators() -> dict[str, DynamicsProtocol]:
    simulators: dict[str, DynamicsProtocol] = {}

    simulators["single_integrator"] = DynamicsFactory.create(
        system_name="single_integrator",
        config={
            "dt": 0.05,
            "goal": [0.0, 0.0],
            "randomize_goal": False,
        },
    )
    simulators["double_integrator"] = DynamicsFactory.create(
        system_name="double_integrator",
        config={
            "dt": 0.05,
            "goal": [0.0, 0.0],
            "randomize_goal": False,
        },
    )
    simulators["unicycle1"] = DynamicsFactory.create(
        system_name="unicycle1",
        config={
            "dt": 0.05,
            "goal": [0.0, 0.0, 0.0],
            "randomize_goal": False,
        },
    )
    simulators["unicycle2"] = DynamicsFactory.create(
        system_name="unicycle2",
        config={
            "dt": 0.05,
            "goal": [0.0, 0.0, 0.0],
            "randomize_goal": False,
            "randomize_initial_velocity": False,
        },
    )
    simulators["multi_robot"] = DynamicsFactory.create(
        system_name="multi_robot",
        config={
            "dt": 0.05,
            "d_safe": 0.1,
            "robots": [
                {
                    "system": "double_integrator",
                    "config": {
                        "dt": 0.05,
                        "goal": [0.0, 0.0],
                        "randomize_goal": False,
                    },
                },
                {
                    "system": "double_integrator",
                    "config": {
                        "dt": 0.05,
                        "goal": [2.0, -1.0],
                        "randomize_goal": False,
                    },
                },
            ],
        },
    )

    return simulators


def _assert_partition(slices: list[slice], total_size: int) -> None:
    assert len(slices) > 0

    covered = np.zeros(total_size, dtype=bool)
    for idx, slc in enumerate(slices):
        assert slc.start is not None and slc.stop is not None
        assert 0 <= slc.start < slc.stop <= total_size, f"invalid slice {idx}: {slc}"
        assert not covered[slc.start:slc.stop].any(), f"overlap detected in slice {idx}: {slc}"
        covered[slc.start:slc.stop] = True

    assert covered.all(), "slices must cover the full vector range"

    sorted_slices = sorted(slices, key=lambda s: int(s.start))
    cursor = 0
    for slc in sorted_slices:
        assert slc.start == cursor, "slices must be contiguous without gaps"
        cursor = int(slc.stop)
    assert cursor == total_size


class SimulatorContractTests(unittest.TestCase):
    def test_predict_next_state_is_side_effect_free(self) -> None:
        simulators = _build_simulators()
        for name, simulator in simulators.items():
            state = simulator.reset(simulator.random_initial_state(np.random.default_rng(11)))
            action = np.zeros(int(simulator.nu), dtype=float)
            time_before = simulator.time
            stored_state_before = None if simulator.state is None else simulator.state.copy()

            predicted = simulator.predict_next_state(state, action)

            self.assertEqual(simulator.time, time_before, name)
            np.testing.assert_array_equal(simulator.state, stored_state_before, err_msg=name)
            np.testing.assert_allclose(predicted, simulator.step(state, action), err_msg=name)

    def test_unicycle1_step_wraps_theta_and_observe_invert_obs_are_so2_consistent(self) -> None:
        simulator = _build_simulators()["unicycle1"]
        state = simulator.reset(np.array([0.0, 0.0, np.pi - 0.01], dtype=float))
        action = np.array([0.0, 1.0], dtype=float)

        next_state = simulator.step(state, action)
        expected_unwrapped_theta = float(state[2] + action[1] * simulator.dt)

        self.assertGreaterEqual(next_state[2], -np.pi)
        self.assertLessEqual(next_state[2], np.pi)
        np.testing.assert_allclose(
            [np.sin(next_state[2]), np.cos(next_state[2])],
            [np.sin(expected_unwrapped_theta), np.cos(expected_unwrapped_theta)],
            atol=1e-9,
        )

        observation = simulator.observe(next_state)
        recovered_state = simulator.invert_obs(observation)
        np.testing.assert_allclose(recovered_state[:2], next_state[:2], atol=1e-9)
        np.testing.assert_allclose(
            [np.sin(recovered_state[2]), np.cos(recovered_state[2])],
            [np.sin(next_state[2]), np.cos(next_state[2])],
            atol=1e-9,
        )

    def test_fleet_contract_invariants(self) -> None:
        simulators = _build_simulators()
        for name, simulator in simulators.items():
            self.assertEqual(simulator.num_robots, len(simulator.simulators), name)
            self.assertEqual(len(simulator.robot_state_slices), simulator.num_robots, name)
            self.assertEqual(len(simulator.robot_action_slices), simulator.num_robots, name)

            _assert_partition(simulator.robot_state_slices, int(simulator.nx))
            _assert_partition(simulator.robot_action_slices, int(simulator.nu))

    def test_dataset_schema_matches_formatted_frames(self) -> None:
        simulators = _build_simulators()
        rng = np.random.default_rng(7)

        for name, simulator in simulators.items():
            initial_state = simulator.random_initial_state(rng)
            state = simulator.reset(initial_state)
            observation = simulator.observe(state)
            action = np.zeros(int(simulator.nu), dtype=np.float32)

            features = simulator.get_dataset_features()
            frames = simulator.format_dataset_frame(observation, action)
            self.assertTrue(isinstance(frames, list) and frames, name)
            frame = frames[0]

            obs_feature_dim = 0
            action_feature_dim = 0

            for feature_name, feature_info in features.items():
                self.assertIn(feature_name, frame, f"{name}: missing feature '{feature_name}' in formatted frame")

                feature_shape = feature_info.get("shape")
                self.assertTrue(isinstance(feature_shape, tuple) and len(feature_shape) == 1)
                expected_dim = int(feature_shape[0])

                value = frame[feature_name]
                value_arr = np.asarray(value)
                self.assertEqual(value_arr.ndim, 1)
                self.assertEqual(
                    value_arr.shape[0],
                    expected_dim,
                    f"{name}: feature '{feature_name}' expected dim {expected_dim}, got {value_arr.shape[0]}",
                )

                feature_names = feature_info.get("names")
                if feature_names is not None:
                    self.assertEqual(
                        len(feature_names),
                        expected_dim,
                        f"{name}: feature '{feature_name}' names length mismatch",
                    )

                if feature_name.startswith("observation."):
                    obs_feature_dim += expected_dim
                elif feature_name == "action" or feature_name.endswith(".action"):
                    action_feature_dim += expected_dim

            expected_obs_dim = int(simulator.obs_dim)
            expected_action_dim = int(simulator.nu)
            if simulator.num_robots > 1:
                expected_obs_dim = int(simulator.robot_observation_slices[0].stop)
                expected_action_dim = int(simulator.robot_action_slices[0].stop)
            self.assertEqual(
                obs_feature_dim,
                expected_obs_dim,
                f"{name}: observation feature sum {obs_feature_dim} != expected {expected_obs_dim}",
            )
            self.assertEqual(
                action_feature_dim,
                expected_action_dim,
                f"{name}: action feature sum {action_feature_dim} != expected {expected_action_dim}",
            )

    def test_geometry_metadata_contract(self) -> None:
        simulators = _build_simulators()
        expected_is_euclidean = {
            "single_integrator": True,
            "double_integrator": True,
            "unicycle1": False,
            "unicycle2": False,
            "multi_robot": True,
        }

        for name, expected in expected_is_euclidean.items():
            self.assertIn(name, simulators)
            simulator = simulators[name]
            self.assertEqual(
                bool(simulator.is_euclidean),
                expected,
                f"{name}: is_euclidean mismatch",
            )

    def test_angular_state_indices_contract(self) -> None:
        simulators = _build_simulators()
        expected_angular_state_indices = {
            "single_integrator": (),
            "double_integrator": (),
            "unicycle1": (2,),
            "unicycle2": (2,),
            "multi_robot": (),
        }

        for name, expected in expected_angular_state_indices.items():
            self.assertIn(name, simulators)
            simulator = simulators[name]
            self.assertEqual(
                tuple(simulator.angular_state_indices),
                expected,
                f"{name}: angular_state_indices mismatch",
            )

    def test_multi_robot_casadi_dynamics_maps_homogeneous_fleet(self) -> None:
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": 0.05,
                "robots": [
                    {
                        "system": "double_integrator",
                        "config": {"dt": 0.05, "goal": [0.0, 0.0], "randomize_goal": False},
                    },
                    {
                        "system": "double_integrator",
                        "config": {"dt": 0.05, "goal": [1.0, 1.0], "randomize_goal": False},
                    },
                ],
            },
        )
        state = np.arange(simulator.nx, dtype=float) / 10.0
        action = np.arange(simulator.nu, dtype=float) / 20.0

        mapped_result = np.asarray(
            simulator.casadi_dynamics(ca.DM(state), ca.DM(action))
        ).reshape(-1)
        loop_result = np.concatenate(
            [
                np.asarray(sim.casadi_dynamics(ca.DM(state[state_slice]), ca.DM(action[action_slice]))).reshape(-1)
                for sim, state_slice, action_slice in zip(
                    simulator.simulators,
                    simulator.robot_state_slices,
                    simulator.robot_action_slices,
                )
            ]
        )

        np.testing.assert_allclose(mapped_result, loop_result)
        mapped_function = simulator._casadi_mapped_dynamics
        self.assertIsNotNone(mapped_function)
        simulator.casadi_dynamics(ca.DM(state), ca.DM(action))
        self.assertIs(simulator._casadi_mapped_dynamics, mapped_function)

    def test_multi_robot_non_euclidean_geometry_metadata(self) -> None:
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": 0.05,
                "d_safe": 0.1,
                "robots": [
                    {
                        "system": "unicycle2",
                        "config": {
                            "dt": 0.05,
                            "goal": [0.0, 0.0, 0.0],
                            "randomize_goal": False,
                            "randomize_initial_velocity": False,
                        },
                    },
                    {
                        "system": "unicycle2",
                        "config": {
                            "dt": 0.05,
                            "goal": [1.0, -1.0, 0.0],
                            "randomize_goal": False,
                            "randomize_initial_velocity": False,
                        },
                    },
                ],
            },
        )

        self.assertFalse(bool(simulator.is_euclidean))
        self.assertEqual(tuple(simulator.angular_state_indices), (2, 7))

    def test_is_collision_gates_on_d_collision_not_d_safe(self) -> None:
        """d_safe is the planner's soft avoidance buffer; d_collision is the physical
        threshold is_collision() must use. A distance strictly between the two must
        not be reported as a collision, even though the planner would avoid it."""
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": 0.05,
                "d_safe": 0.3,
                "d_collision": 0.1,
                "robots": [
                    {"system": "double_integrator", "config": {"dt": 0.05, "goal": [0.0, 0.0], "randomize_goal": False}},
                    {"system": "double_integrator", "config": {"dt": 0.05, "goal": [2.0, -1.0], "randomize_goal": False}},
                ],
            },
        )

        # 0.2 apart: inside the soft buffer (d_safe=0.3) but outside the physical
        # collision radius (d_collision=0.1) -- not a real collision.
        near_state = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0])
        self.assertFalse(simulator.is_collision(near_state))
        self.assertIsNone(simulator.collision_details(near_state))

        # 0.05 apart: inside the physical collision radius.
        colliding_state = np.array([0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
        self.assertTrue(simulator.is_collision(colliding_state))
        details = simulator.collision_details(colliding_state)
        self.assertIsNotNone(details)
        self.assertAlmostEqual(details["distance"], 0.05, places=6)
        self.assertEqual(details["threshold"], 0.1)

    def test_multi_robot_neighbor_observation_includes_relative_heading(self) -> None:
        simulator = DynamicsFactory.create(
            system_name="multi_robot",
            config={
                "dt": 0.05,
                "robots": [
                    {
                        "system": "unicycle2",
                        "config": {
                            "dt": 0.05,
                            "goal": [0.0, 0.0, 0.0],
                            "randomize_goal": False,
                            "randomize_initial_velocity": False,
                        },
                    },
                    {
                        "system": "unicycle2",
                        "config": {
                            "dt": 0.05,
                            "goal": [1.0, 0.0, 0.0],
                            "randomize_goal": False,
                            "randomize_initial_velocity": False,
                        },
                    },
                ],
            },
        )
        state = simulator.reset(
            np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 1.0, 0.0, -np.pi / 2, 0.0, 0.0])
        )

        neighbor_state = simulator.decentralized_policy_observation(simulator.observe(state))["observation.neighbor_state"]
        np.testing.assert_allclose(neighbor_state, [0.0, -1.0, 0.0, -1.0], atol=1e-9)
        self.assertEqual(
            simulator.get_dataset_features()["observation.neighbor_state"]["names"],
            ["neighbor_0_x", "neighbor_0_y", "neighbor_0_sin_rel_theta", "neighbor_0_cos_rel_theta"],
        )

    def test_multi_robot_neighbor_observation_supports_three_dimensional_position(self) -> None:
        class ThreeDimensionalSimulator:
            dt = 0.05
            nx = 3
            nu = 1
            max_action = 1.0
            is_euclidean = True
            angular_state_indices = ()
            position_indices = (0, 1, 2)

            def get_dataset_features(self) -> dict[str, object]:
                return {
                    "observation.environment_state": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
                    "observation.state": {"dtype": "float32", "shape": (0,), "names": []},
                    "observation.neighbor_state": {"dtype": "float32", "shape": (0,), "names": []},
                    "observation.neighbor_mask": {"dtype": "float32", "shape": (0,), "names": []},
                    "action": {"dtype": "float32", "shape": (1,), "names": ["action"]},
                }

            def reset(self, state: np.ndarray) -> np.ndarray:
                return np.asarray(state, dtype=float)

            def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray:
                return np.asarray(state, dtype=float)

            def global_vector_to_ego(self, vector: np.ndarray, state: np.ndarray) -> np.ndarray:
                return np.asarray(vector, dtype=float)

        simulator = MultiRobotSimulator([ThreeDimensionalSimulator(), ThreeDimensionalSimulator()])
        state = simulator.reset(np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0]))
        neighbor_state = simulator.decentralized_policy_observation(simulator.observe(state))["observation.neighbor_state"]

        np.testing.assert_allclose(neighbor_state, [1.0, 2.0, 3.0])
        self.assertEqual(
            simulator.get_dataset_features()["observation.neighbor_state"]["names"],
            ["neighbor_0_x", "neighbor_0_y", "neighbor_0_z"],
        )

    def test_heading_system_boundary_roundtrip_preserves_so2_state(self) -> None:
        simulators = _build_simulators()
        test_states = {
            "unicycle1": np.array([1.5, -0.25, 3.7], dtype=float),
            "unicycle2": np.array([1.5, -0.25, 3.7, 0.2, -0.1], dtype=float),
        }

        for name, initial_state in test_states.items():
            simulator = simulators[name]
            state = simulator.reset(initial_state)
            observation = simulator.observe(state)
            recovered_state = simulator.invert_obs(observation)

            np.testing.assert_allclose(recovered_state[:2], state[:2], atol=1e-9)
            np.testing.assert_allclose(
                [np.sin(recovered_state[2]), np.cos(recovered_state[2])],
                [np.sin(state[2]), np.cos(state[2])],
                atol=1e-9,
            )

            if state.shape[0] > 3:
                np.testing.assert_allclose(recovered_state[3:], state[3:], atol=1e-9)

    def test_randomize_goal_hook_is_formalized_across_simulators(self) -> None:
        simulators = _build_simulators()

        for name, simulator in simulators.items():
            self.assertTrue(
                callable(getattr(simulator, "randomize_goal_for_reset", None)),
                f"{name}: simulator is missing randomize_goal_for_reset()",
            )

    def test_rollout_termination_api_is_formalized_across_simulators(self) -> None:
        simulators = _build_simulators()

        for name, simulator in simulators.items():
            self.assertTrue(
                callable(getattr(simulator, "reset_rollout_termination", None)),
                f"{name}: simulator is missing reset_rollout_termination()",
            )
            self.assertTrue(
                callable(getattr(simulator, "set_early_rollout_termination", None)),
                f"{name}: simulator is missing set_early_rollout_termination()",
            )
            self.assertTrue(
                callable(getattr(simulator, "should_terminate_rollout", None)),
                f"{name}: simulator is missing should_terminate_rollout()",
            )

    def test_rollout_termination_state_resets_and_honors_hold_steps(self) -> None:
        simulators = _build_simulators()

        for name, simulator in simulators.items():
            goal_state = simulator.goal_state
            self.assertTrue(simulator.is_done(goal_state), f"{name}: goal state should satisfy is_done()")

            hold_steps = int(simulator.config.get("done_hold_steps", 5))
            self.assertGreaterEqual(hold_steps, 1)

            simulator.reset(goal_state)
            for step_idx in range(hold_steps - 1):
                self.assertFalse(
                    simulator.should_terminate_rollout(goal_state),
                    f"{name}: termination triggered too early on step {step_idx + 1}",
                )

            self.assertTrue(
                simulator.should_terminate_rollout(goal_state),
                f"{name}: termination did not trigger on hold step {hold_steps}",
            )

            simulator.reset(goal_state)
            self.assertFalse(
                simulator.should_terminate_rollout(goal_state),
                f"{name}: termination counter was not reset by reset()",
            )

    def test_unicycle2_randomized_goal_uses_normalized_bounds(self) -> None:
        validated = validate_system_config(
            system_name="unicycle2",
            raw_config={
                "dt": 0.05,
                "randomize_goal": True,
                "randomize_initial_velocity": False,
                "max_accel": 0.25,
                "max_speed": 0.5,
                "max_omega": 0.5,
            },
        )

        simulator = DynamicsFactory.create(system_name="unicycle2", config=validated)
        simulator.reset_random()

        self.assertEqual(validated["goal_position_bounds"], [-1.0, 1.0])
        self.assertTrue(np.all(simulator.goal[:2] >= -1.0))
        self.assertTrue(np.all(simulator.goal[:2] <= 1.0))

    def test_planar_start_offset_is_area_uniform_on_annulus(self) -> None:
        simulator = DynamicsFactory.create(
            system_name="single_integrator",
            config={
                "dt": 0.05,
                "goal": [0.0, 0.0],
                "randomize_goal": False,
            },
        )

        radius_min = 0.25
        radius_max = 1.0
        rng = np.random.default_rng(17)
        radii_sq = []
        for _ in range(4000):
            offset = simulator.sample_planar_start_offset(
                rng,
                radius_bounds=(radius_min, radius_max),
                min_goal_distance=radius_min,
            )
            radii_sq.append(float(np.dot(offset, offset)))

        empirical_mean = float(np.mean(np.asarray(radii_sq, dtype=float)))
        expected_mean = 0.5 * (radius_min**2 + radius_max**2)
        self.assertAlmostEqual(empirical_mean, expected_mean, delta=0.02)

    def test_seeded_reset_random_respects_configured_start_region(self) -> None:
        def position_distance_to_goal(simulator: DynamicsProtocol, state: np.ndarray) -> float:
            if simulator.num_robots == 1:
                return float(np.linalg.norm(state[:2] - simulator.goal_state[:2]))

            distances = []
            for sub_sim, state_slice in zip(simulator.simulators, simulator.robot_state_slices):
                robot_state = state[state_slice]
                distances.append(float(np.linalg.norm(robot_state[:2] - sub_sim.goal_state[:2])))
            return min(distances)

        configs = {
            "single_integrator": {
                "dt": 0.05,
                "goal": [0.0, 0.0],
                "randomize_goal": False,
                "initial_position_radius_bounds": [0.0, 1.0],
                "initial_position_min_goal_distance": 0.25,
                "initial_state_seed": 13,
            },
            "double_integrator": {
                "dt": 0.05,
                "goal": [0.0, 0.0],
                "randomize_goal": False,
                "initial_position_radius_bounds": [0.0, 1.0],
                "initial_position_min_goal_distance": 0.25,
                "initial_state_seed": 13,
            },
            "unicycle1": {
                "dt": 0.05,
                "goal": [0.0, 0.0, 0.0],
                "randomize_goal": False,
                "initial_position_radius_bounds": [0.0, 1.0],
                "initial_position_min_goal_distance": 0.25,
                "initial_state_seed": 13,
            },
            "unicycle2": {
                "dt": 0.05,
                "goal": [0.0, 0.0, 0.0],
                "randomize_initial_velocity": False,
                "initial_position_radius_bounds": [0.0, 1.0],
                "initial_position_min_goal_distance": 0.25,
                "initial_state_seed": 13,
            },
            "multi_robot": {
                "dt": 0.05,
                "d_safe": 0.1,
                "initial_state_seed": 13,
                "robots": [
                    {
                        "system": "double_integrator",
                        "config": {
                            "dt": 0.05,
                            "goal": [0.0, 0.0],
                            "randomize_goal": False,
                            "initial_position_radius_bounds": [0.0, 1.0],
                            "initial_position_min_goal_distance": 0.25,
                        },
                    },
                    {
                        "system": "double_integrator",
                        "config": {
                            "dt": 0.05,
                            "goal": [2.0, -1.0],
                            "randomize_goal": False,
                            "initial_position_radius_bounds": [0.0, 1.0],
                            "initial_position_min_goal_distance": 0.25,
                        },
                    },
                ],
            },
        }

        for name, raw_config in configs.items():
            validated = validate_system_config(system_name=name, raw_config=raw_config)
            simulator_a = DynamicsFactory.create(system_name=name, config=validated)
            simulator_b = DynamicsFactory.create(system_name=name, config=validated)

            state_a = simulator_a.reset_random()
            state_b = simulator_b.reset_random()

            np.testing.assert_allclose(state_a, state_b, atol=1e-9)
            self.assertGreater(
                position_distance_to_goal(simulator_a, state_a),
                0.25,
                f"{name}: reset_random sampled inside the configured minimum goal distance",
            )

    def test_multi_robot_per_robot_r_weights(self) -> None:
        raw_config = {
            "dt": 0.05,
            "d_safe": 0.1,
            "horizon": 10,
            "mode": "mpc",
            "Q_diag": [10.0, 10.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0],
            "R_weight_per_robot": [0.1, [0.2, 0.3]],
            "terminal_cost_multiplier": 10.0,
            "error_tolerance": 0.05,
            "robots": [
                {
                    "system": "double_integrator",
                    "config": {
                        "dt": 0.05,
                        "max_accel": 2.0,
                        "goal": [0.0, 0.0],
                        "randomize_goal": False,
                        "error_tolerance": 0.05,
                    },
                },
                {
                    "system": "double_integrator",
                    "config": {
                        "dt": 0.05,
                        "max_accel": 2.0,
                        "goal": [2.0, -1.0],
                        "randomize_goal": False,
                        "error_tolerance": 0.05,
                    },
                },
            ],
        }

        validated = validate_system_config(system_name="multi_robot", raw_config=raw_config)
        self.assertEqual(validated["R_diag"], [0.1, 0.1, 0.2, 0.3])
        self.assertEqual(validated["R_weight_per_robot"], [[0.1, 0.1], [0.2, 0.3]])

        simulator = DynamicsFactory.create(system_name="multi_robot", config=validated)
        planner = CasadiPlanner(simulator=simulator, config=validated)
        np.testing.assert_allclose(np.diag(planner.R), np.asarray(validated["R_diag"], dtype=float))

    def test_multi_robot_per_robot_r_weight_shape_validation(self) -> None:
        raw_config = {
            "dt": 0.05,
            "d_safe": 0.1,
            "horizon": 10,
            "mode": "mpc",
            "Q_diag": [10.0, 10.0, 1.0, 1.0, 10.0, 10.0, 1.0, 1.0],
            "R_weight_per_robot": [0.1],
            "terminal_cost_multiplier": 10.0,
            "error_tolerance": 0.05,
            "robots": [
                {
                    "system": "double_integrator",
                    "config": {
                        "dt": 0.05,
                        "max_accel": 2.0,
                        "goal": [0.0, 0.0],
                        "randomize_goal": False,
                        "error_tolerance": 0.05,
                    },
                },
                {
                    "system": "double_integrator",
                    "config": {
                        "dt": 0.05,
                        "max_accel": 2.0,
                        "goal": [2.0, -1.0],
                        "randomize_goal": False,
                        "error_tolerance": 0.05,
                    },
                },
            ],
        }

        with self.assertRaises(ConfigurationError):
            validate_system_config(system_name="multi_robot", raw_config=raw_config)


if __name__ == "__main__":
    unittest.main()
