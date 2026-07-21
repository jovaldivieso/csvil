from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is missing or malformed."""


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 20
    mode: str = "mpc"
    q_diag: tuple[float, ...] = ()
    r_weight: float = 0.1
    terminal_cost_multiplier: float = 10.0


@dataclass(frozen=True)
class SingleIntegratorSystemConfig:
    dt: float = 0.05
    goal: tuple[float, float] = (0.0, 0.0)
    randomize_goal: bool = True
    max_vel: float = 1.0
    error_tolerance: float = 0.05


@dataclass(frozen=True)
class DoubleIntegratorSystemConfig:
    dt: float = 0.05
    goal: tuple[float, float] = (0.0, 0.0)
    randomize_goal: bool = True
    max_accel: float = 2.0
    error_tolerance: float = 0.05


@dataclass(frozen=True)
class Unicycle1SystemConfig:
    dt: float = 0.05
    goal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize_goal: bool = True
    max_v: float = 2.0
    error_tolerance: float = 0.05


@dataclass(frozen=True)
class Unicycle2SystemConfig:
    dt: float = 0.05
    goal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize_initial_velocity: bool = False
    max_accel: float = 0.25
    max_speed: float = 0.5
    max_omega: float = 0.5
    error_tolerance: float = 0.05


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML into a dictionary and fail with context when invalid."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ConfigurationError(f"Expected YAML mapping in {path}, got {type(data).__name__}.")
    return data


def _float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if not isinstance(value, (int, float)):
        raise ConfigurationError(f"'{key}' must be numeric, got {type(value).__name__}.")
    return float(value)


def _int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if not isinstance(value, int):
        raise ConfigurationError(f"'{key}' must be int, got {type(value).__name__}.")
    return int(value)


def _bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"'{key}' must be bool, got {type(value).__name__}.")
    return value


def _vector(config: Mapping[str, Any], key: str, size: int, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = config.get(key, list(default))
    if not isinstance(raw, list):
        raise ConfigurationError(f"'{key}' must be a list of length {size}.")
    if len(raw) != size:
        raise ConfigurationError(f"'{key}' must have length {size}, got {len(raw)}.")
    out: list[float] = []
    for idx, value in enumerate(raw):
        if not isinstance(value, (int, float)):
            raise ConfigurationError(
                f"'{key}[{idx}]' must be numeric, got {type(value).__name__}."
            )
        out.append(float(value))
    return tuple(out)


def _validate_planner(config: Mapping[str, Any], nx: int) -> PlannerConfig:
    mode = config.get("mode", "mpc")
    if mode not in {"mpc", "open_loop"}:
        raise ConfigurationError("'mode' must be one of {'mpc', 'open_loop'}.")

    planner = PlannerConfig(
        horizon=_int(config, "horizon", 20),
        mode=mode,
        q_diag=_vector(config, "Q_diag", nx, tuple([10.0] * nx)),
        r_weight=_float(config, "R_weight", 0.1),
        terminal_cost_multiplier=_float(config, "terminal_cost_multiplier", 10.0),
    )

    if planner.horizon <= 0:
        raise ConfigurationError("'horizon' must be positive.")
    if planner.r_weight <= 0:
        raise ConfigurationError("'R_weight' must be positive.")
    if planner.terminal_cost_multiplier <= 0:
        raise ConfigurationError("'terminal_cost_multiplier' must be positive.")
    return planner


def load_and_validate_system_config(system_name: str, config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and return a validated config dictionary for a system."""
    raw = load_yaml_config(config_path)
    return validate_system_config(system_name=system_name, raw_config=raw)


def validate_system_config(system_name: str, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate system and planner fields, returning normalized config values."""
    dt = _float(raw_config, "dt", 0.05)
    if dt <= 0:
        raise ConfigurationError("'dt' must be positive.")

    error_tolerance = _float(raw_config, "error_tolerance", 0.05)
    if error_tolerance <= 0:
        raise ConfigurationError("'error_tolerance' must be positive.")

    config_out: dict[str, Any]
    nx: int

    if system_name == "single_integrator":
        system_cfg = SingleIntegratorSystemConfig(
            dt=dt,
            goal=_vector(raw_config, "goal", 2, (0.0, 0.0)),
            randomize_goal=_bool(raw_config, "randomize_goal", "goal" not in raw_config),
            max_vel=_float(raw_config, "max_vel", 1.0),
            error_tolerance=error_tolerance,
        )
        if system_cfg.max_vel <= 0:
            raise ConfigurationError("'max_vel' must be positive.")
        config_out = {
            "dt": system_cfg.dt,
            "goal": list(system_cfg.goal),
            "randomize_goal": system_cfg.randomize_goal,
            "max_vel": system_cfg.max_vel,
            "error_tolerance": system_cfg.error_tolerance,
        }
        nx = 2

    elif system_name == "double_integrator":
        system_cfg = DoubleIntegratorSystemConfig(
            dt=dt,
            goal=_vector(raw_config, "goal", 2, (0.0, 0.0)),
            randomize_goal=_bool(raw_config, "randomize_goal", "goal" not in raw_config),
            max_accel=_float(raw_config, "max_accel", 2.0),
            error_tolerance=error_tolerance,
        )
        if system_cfg.max_accel <= 0:
            raise ConfigurationError("'max_accel' must be positive.")
        config_out = {
            "dt": system_cfg.dt,
            "goal": list(system_cfg.goal),
            "randomize_goal": system_cfg.randomize_goal,
            "max_accel": system_cfg.max_accel,
            "error_tolerance": system_cfg.error_tolerance,
        }
        nx = 4

    elif system_name == "unicycle1":
        system_cfg = Unicycle1SystemConfig(
            dt=dt,
            goal=_vector(raw_config, "goal", 3, (0.0, 0.0, 0.0)),
            randomize_goal=_bool(raw_config, "randomize_goal", "goal" not in raw_config),
            max_v=_float(raw_config, "max_v", 2.0),
            error_tolerance=error_tolerance,
        )
        if system_cfg.max_v <= 0:
            raise ConfigurationError("'max_v' must be positive.")
        config_out = {
            "dt": system_cfg.dt,
            "goal": list(system_cfg.goal),
            "randomize_goal": system_cfg.randomize_goal,
            "max_v": system_cfg.max_v,
            "error_tolerance": system_cfg.error_tolerance,
        }
        nx = 3

    elif system_name == "unicycle2":
        system_cfg = Unicycle2SystemConfig(
            dt=dt,
            goal=_vector(raw_config, "goal", 3, (0.0, 0.0, 0.0)),
            randomize_initial_velocity=_bool(raw_config, "randomize_initial_velocity", False),
            max_accel=_float(raw_config, "max_accel", 0.25),
            max_speed=_float(raw_config, "max_speed", 0.5),
            max_omega=_float(raw_config, "max_omega", 0.5),
            error_tolerance=error_tolerance,
        )
        if system_cfg.max_accel <= 0 or system_cfg.max_speed <= 0 or system_cfg.max_omega <= 0:
            raise ConfigurationError("'max_accel', 'max_speed', and 'max_omega' must be positive.")
        config_out = {
            "dt": system_cfg.dt,
            "goal": list(system_cfg.goal),
            "randomize_initial_velocity": system_cfg.randomize_initial_velocity,
            "max_accel": system_cfg.max_accel,
            "max_speed": system_cfg.max_speed,
            "max_omega": system_cfg.max_omega,
            "error_tolerance": system_cfg.error_tolerance,
        }
        nx = 5

    else:
        raise ConfigurationError(f"Unknown system '{system_name}'.")

    planner_cfg = _validate_planner(raw_config, nx=nx)
    config_out.update(
        {
            "horizon": planner_cfg.horizon,
            "mode": planner_cfg.mode,
            "Q_diag": list(planner_cfg.q_diag),
            "R_weight": planner_cfg.r_weight,
            "terminal_cost_multiplier": planner_cfg.terminal_cost_multiplier,
        }
    )
    return config_out
