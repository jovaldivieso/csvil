from .config import ConfigurationError, load_and_validate_system_config

__all__ = [
    "ConfigurationError",
    "DynamicsFactory",
    "PlannerFactory",
    "SE2_SYSTEMS",
    "load_and_validate_system_config",
]


def __getattr__(name: str):
    if name in {"DynamicsFactory", "PlannerFactory", "SE2_SYSTEMS"}:
        from .factory import DynamicsFactory, PlannerFactory, SE2_SYSTEMS

        return {
            "DynamicsFactory": DynamicsFactory,
            "PlannerFactory": PlannerFactory,
            "SE2_SYSTEMS": SE2_SYSTEMS,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
