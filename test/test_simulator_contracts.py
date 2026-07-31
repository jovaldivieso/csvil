import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from core.factory import DynamicsFactory
from core.config import ConfigurationError, validate_system_config
from planning.casadi_planner import CasadiPlanner
from systems.dynamics import DynamicsProtocol
from systems.state_space_types import SE2PoseState, SO2State


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
    def test_lie_group_primitives(self) -> None:
        source = SO2State.from_angle(0.3)
        target = SO2State.from_angle(-2.8)
        relative = source.between(target)
        self.assertAlmostEqual(relative.log_vee(), source.error_to(target))

        pose = SE2PoseState.from_array(np.array([1.2, -0.7, 0.9]))
        identity = pose.compose(pose.inverse())
        np.testing.assert_allclose(identity.as_matrix(), np.eye(3), atol=1e-9)

        other_pose = SE2PoseState.from_array(np.array([-0.5, 1.8, -1.1]))
        relative_pose = pose.between(other_pose)
        recomposed = relative_pose.compose(pose)
        np.testing.assert_allclose(recomposed.as_matrix(), other_pose.as_matrix(), atol=1e-9)

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
            frame = simulator.format_dataset_frame(observation, action)

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

            self.assertEqual(
                obs_feature_dim,
                int(simulator.obs_dim),
                f"{name}: observation feature sum {obs_feature_dim} != obs_dim {simulator.obs_dim}",
            )
            self.assertEqual(
                action_feature_dim,
                int(simulator.nu),
                f"{name}: action feature sum {action_feature_dim} != nu {simulator.nu}",
            )

    def test_heading_metadata_contract(self) -> None:
        simulators = _build_simulators()
        expected_has_heading = {
            "single_integrator": False,
            "double_integrator": False,
            "unicycle1": True,
            "unicycle2": True,
            "multi_robot": False,
        }

        for name, expected in expected_has_heading.items():
            self.assertIn(name, simulators)
            simulator = simulators[name]
            self.assertEqual(
                bool(simulator.has_heading),
                expected,
                f"{name}: has_heading mismatch",
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
            recovered_orientation = SO2State.from_angle(recovered_state[2])
            original_orientation = SO2State.from_angle(state[2])
            self.assertAlmostEqual(original_orientation.error_to(recovered_orientation), 0.0)

            if state.shape[0] > 3:
                np.testing.assert_allclose(recovered_state[3:], state[3:], atol=1e-9)

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
