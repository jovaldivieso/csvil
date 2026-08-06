from __future__ import annotations

from systems.dynamics import DynamicsProtocol


DEFAULT_SINGLE_ROBOT_SEED = 1
DEFAULT_MULTI_ROBOT_SEED_BASE = 1
DEFAULT_MULTI_ROBOT_SEED_STRIDE = 100


def default_seed_argument_for_simulator(
    simulator: DynamicsProtocol,
) -> list[int] | list[list[int]]:
    if simulator.num_robots <= 1:
        return [DEFAULT_SINGLE_ROBOT_SEED]

    return [
        [DEFAULT_MULTI_ROBOT_SEED_BASE + DEFAULT_MULTI_ROBOT_SEED_STRIDE * robot_idx]
        for robot_idx in range(simulator.num_robots)
    ]