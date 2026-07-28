from __future__ import annotations

from typing import Any, Callable, Mapping

from planning.casadi_planner import CasadiPlanner
from planning.planner import PlannerProtocol
from systems.double_integrator import DoubleIntegrator
from systems.dynamics import DynamicsProtocol
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
}

# SE(2) is the 2D rigid-body task space (x, y, theta), enabling orientation-aware arrow plotting.
SE2_SYSTEMS: set[str] = {"unicycle1", "unicycle2"}


class DynamicsFactory:
    """Factory for dynamics simulator instances."""

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(SYSTEM_REGISTRY.keys())

    @staticmethod
    def create(system_name: str, config: Mapping[str, Any]) -> DynamicsProtocol:
        try:
            simulator_cls = SYSTEM_REGISTRY[system_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown system '{system_name}'. Available: {sorted(SYSTEM_REGISTRY.keys())}"
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
