from __future__ import annotations

from typing import Any, Callable, Mapping

from planning.dblacam_planner import DbLacamPlanner
from planning.casadi_planner import CasadiPlanner
from planning.planner import PlannerProtocol
from systems.double_integrator import DoubleIntegrator
from systems.dynamics import DynamicsProtocol
from systems.multi_robot import MultiRobotSimulator
from systems.single_integrator import SingleIntegrator
from systems.unicycle1 import Unicycle1
from systems.unicycle2 import Unicycle2

DynamicsBuilder = Callable[[Mapping[str, Any]], DynamicsProtocol]
PlannerBuilder = Callable[[DynamicsProtocol, Mapping[str, Any]], PlannerProtocol]


SYSTEM_REGISTRY: dict[str, DynamicsBuilder] = {
    "single_integrator": SingleIntegrator,
    "double_integrator": DoubleIntegrator,
    "unicycle1": Unicycle1,
    "unicycle2": Unicycle2,
}

PLANNER_REGISTRY: dict[str, PlannerBuilder] = {
    "casadi": CasadiPlanner,
    "dblacam": DbLacamPlanner,
}

# SE(2) is the 2D rigid-body task space (x, y, theta), enabling orientation-aware arrow plotting.
SE2_SYSTEMS: set[str] = {"unicycle1", "unicycle2"}


class DynamicsFactory:
    """Factory for dynamics simulator instances."""

    @staticmethod
    def _create_multi_robot(config: Mapping[str, Any]) -> DynamicsProtocol:
        robots = config.get("robots")
        if not isinstance(robots, list) or len(robots) == 0:
            raise ValueError(
                "'multi_robot' config must contain a non-empty 'robots' list."
            )

        simulators: list[DynamicsProtocol] = []
        for robot_idx, robot_entry in enumerate(robots):
            if not isinstance(robot_entry, Mapping):
                raise ValueError(
                    f"robots[{robot_idx}] must be a mapping with 'system' and 'config'."
                )

            robot_system = robot_entry.get("system")
            robot_config = robot_entry.get("config", {})
            if not isinstance(robot_system, str):
                raise ValueError(f"robots[{robot_idx}].system must be a string.")
            if not isinstance(robot_config, Mapping):
                raise ValueError(f"robots[{robot_idx}].config must be a mapping.")

            simulators.append(
                DynamicsFactory.create(system_name=robot_system, config=robot_config)
            )

        return MultiRobotSimulator(simulators=simulators, config=config)

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple([*SYSTEM_REGISTRY.keys(), "multi_robot"])

    @staticmethod
    def create(system_name: str, config: Mapping[str, Any]) -> DynamicsProtocol:
        dynamic_registry = {**SYSTEM_REGISTRY, "multi_robot": DynamicsFactory._create_multi_robot}

        try:
            simulator_cls = dynamic_registry[system_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown system '{system_name}'. Available: {sorted(dynamic_registry.keys())}"
            ) from exc
        return simulator_cls(config)


class PlannerFactory:
    """Factory for planner instances."""

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(PLANNER_REGISTRY.keys())

    @staticmethod
    def create(
        planner_name: str,
        simulator: DynamicsProtocol,
        config: Mapping[str, Any],
    ) -> PlannerProtocol:
        try:
            planner_cls = PLANNER_REGISTRY[planner_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown planner '{planner_name}'. Available: {sorted(PLANNER_REGISTRY.keys())}"
            ) from exc
        return planner_cls(simulator, config)
