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
    r_diag: tuple[float, ...] = ()
    r_weight_per_robot: tuple[tuple[float, ...], ...] = ()
    terminal_cost_multiplier: float = 10.0


@dataclass(frozen=True)
class InitialStateSamplingConfig:
    position_radius_bounds: tuple[float, float]
    min_goal_distance: float
    seed: int | None = None


@dataclass(frozen=True)
class GoalSamplingConfig:
    position_bounds: tuple[float, float] = (-1.0, 1.0)


@dataclass(frozen=True)
class SingleIntegratorSystemConfig:
    dt: float = 0.05
    goal: tuple[float, float] = (0.0, 0.0)
    randomize_goal: bool = True
    max_vel: float = 1.0
    error_tolerance: float = 0.05
    goal_sampling: GoalSamplingConfig = GoalSamplingConfig()
    initial_state_sampling: InitialStateSamplingConfig = InitialStateSamplingConfig(
        position_radius_bounds=(0.05, 1.0),
        min_goal_distance=0.05,
    )


@dataclass(frozen=True)
class DoubleIntegratorSystemConfig:
    dt: float = 0.05
    goal: tuple[float, float] = (0.0, 0.0)
    randomize_goal: bool = True
    max_accel: float = 2.0
    error_tolerance: float = 0.05
    goal_sampling: GoalSamplingConfig = GoalSamplingConfig()
    initial_state_sampling: InitialStateSamplingConfig = InitialStateSamplingConfig(
        position_radius_bounds=(0.05, 1.0),
        min_goal_distance=0.05,
    )


@dataclass(frozen=True)
class Unicycle1SystemConfig:
    dt: float = 0.05
    goal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize_goal: bool = True
    max_v: float = 2.0
    error_tolerance: float = 0.05
    goal_sampling: GoalSamplingConfig = GoalSamplingConfig()
    initial_state_sampling: InitialStateSamplingConfig = InitialStateSamplingConfig(
        position_radius_bounds=(0.05, 1.0),
        min_goal_distance=0.05,
    )


@dataclass(frozen=True)
class Unicycle2SystemConfig:
    dt: float = 0.05
    goal: tuple[float, float, float] = (0.0, 0.0, 0.0)
    randomize_goal: bool = True
    randomize_initial_velocity: bool = False
    max_accel: float = 0.25
    max_speed: float = 0.5
    max_omega: float = 0.5
    error_tolerance: float = 0.05
    goal_sampling: GoalSamplingConfig = GoalSamplingConfig()
    initial_state_sampling: InitialStateSamplingConfig = InitialStateSamplingConfig(
        position_radius_bounds=(0.05, 1.0),
        min_goal_distance=0.05,
    )


@dataclass(frozen=True)
class MultiRobotMemberConfig:
    system: str
    config: dict[str, Any]


@dataclass(frozen=True)
class MultiRobotSystemConfig:
    dt: float = 0.05
    d_safe: float = 0.0
    robots: tuple[MultiRobotMemberConfig, ...] = ()
    inter_robot_visibility_radius: float | tuple[float, ...] = float("inf")
    error_tolerance: float = 0.05
    initial_state_seed: int | None = None


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


def _optional_int(config: Mapping[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"'{key}' must be int when provided, got {type(value).__name__}.")
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


def _positive_vector(raw: Any, key: str, size: int) -> tuple[float, ...]:
    if not isinstance(raw, list):
        raise ConfigurationError(f"'{key}' must be a list of length {size}.")
    if len(raw) != size:
        raise ConfigurationError(f"'{key}' must have length {size}, got {len(raw)}.")

    out: list[float] = []
    for idx, value in enumerate(raw):
        if not isinstance(value, (int, float)):
            raise ConfigurationError(f"'{key}[{idx}]' must be numeric, got {type(value).__name__}.")
        value_f = float(value)
        if value_f <= 0:
            raise ConfigurationError(f"'{key}[{idx}]' must be positive.")
        out.append(value_f)
    return tuple(out)


def _initial_state_sampling(
    raw_config: Mapping[str, Any],
    *,
    error_tolerance: float,
) -> InitialStateSamplingConfig:
    min_goal_distance = _float(
        raw_config,
        "initial_position_min_goal_distance",
        error_tolerance,
    )
    if min_goal_distance < 0:
        raise ConfigurationError("'initial_position_min_goal_distance' must be non-negative.")

    radius_bounds = _vector(
        raw_config,
        "initial_position_radius_bounds",
        2,
        (min_goal_distance, 1.0),
    )
    radius_min, radius_max = radius_bounds
    if radius_min < 0:
        raise ConfigurationError("'initial_position_radius_bounds[0]' must be non-negative.")
    effective_radius_min = max(radius_min, min_goal_distance)
    if radius_max <= effective_radius_min:
        raise ConfigurationError(
            "'initial_position_min_goal_distance' must be smaller than the maximum initial radius."
        )

    return InitialStateSamplingConfig(
        position_radius_bounds=(effective_radius_min, radius_max),
        min_goal_distance=min_goal_distance,
        seed=_optional_int(raw_config, "initial_state_seed"),
    )


def _goal_sampling(raw_config: Mapping[str, Any]) -> GoalSamplingConfig:
    position_bounds = _vector(
        raw_config,
        "goal_position_bounds",
        2,
        (-1.0, 1.0),
    )
    lower, upper = position_bounds
    if lower >= upper:
        raise ConfigurationError("'goal_position_bounds[0]' must be smaller than 'goal_position_bounds[1]'.")
    return GoalSamplingConfig(position_bounds=position_bounds)


def _parse_r_weight_per_robot(
    raw: Any,
    robot_action_dims: tuple[int, ...],
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(raw, list):
        raise ConfigurationError("'R_weight_per_robot' must be a list.")
    if len(raw) != len(robot_action_dims):
        raise ConfigurationError(
            "'R_weight_per_robot' length must match number of robots "
            f"({len(robot_action_dims)}), got {len(raw)}."
        )

    normalized: list[tuple[float, ...]] = []
    for robot_idx, (entry, action_dim) in enumerate(zip(raw, robot_action_dims)):
        if isinstance(entry, (int, float)):
            value = float(entry)
            if value <= 0:
                raise ConfigurationError(
                    f"'R_weight_per_robot[{robot_idx}]' must be positive."
                )
            normalized.append(tuple([value] * action_dim))
            continue

        if isinstance(entry, list):
            normalized.append(
                _positive_vector(
                    entry,
                    key=f"R_weight_per_robot[{robot_idx}]",
                    size=action_dim,
                )
            )
            continue

        raise ConfigurationError(
            f"'R_weight_per_robot[{robot_idx}]' must be numeric or a list of length {action_dim}."
        )

    return tuple(normalized)


def _validate_planner(
    config: Mapping[str, Any],
    nx: int,
    nu: int,
    robot_action_dims: tuple[int, ...] | None = None,
) -> PlannerConfig:
    mode = config.get("mode", "mpc")
    if mode not in {"mpc", "open_loop"}:
        raise ConfigurationError("'mode' must be one of {'mpc', 'open_loop'}.")

    r_weight = _float(config, "R_weight", 0.1)
    if r_weight <= 0:
        raise ConfigurationError("'R_weight' must be positive.")

    raw_r_diag = config.get("R_diag")
    raw_r_per_robot = config.get("R_weight_per_robot")

    if raw_r_diag is not None and raw_r_per_robot is not None:
        raise ConfigurationError("Use either 'R_diag' or 'R_weight_per_robot', not both.")

    r_weight_per_robot: tuple[tuple[float, ...], ...] = ()
    if raw_r_per_robot is not None:
        if robot_action_dims is None:
            raise ConfigurationError("'R_weight_per_robot' is only supported for 'multi_robot' systems.")
        r_weight_per_robot = _parse_r_weight_per_robot(raw_r_per_robot, robot_action_dims)
        r_diag = tuple(value for robot_diag in r_weight_per_robot for value in robot_diag)
        if len(r_diag) != nu:
            raise ConfigurationError(
                "Expanded 'R_weight_per_robot' does not match action dimension "
                f"nu={nu}; got {len(r_diag)} values."
            )
    elif raw_r_diag is not None:
        r_diag = _positive_vector(raw_r_diag, key="R_diag", size=nu)
    else:
        r_diag = tuple([r_weight] * nu)

    planner = PlannerConfig(
        horizon=_int(config, "horizon", 20),
        mode=mode,
        q_diag=_vector(config, "Q_diag", nx, tuple([10.0] * nx)),
        r_weight=r_weight,
        r_diag=r_diag,
        r_weight_per_robot=r_weight_per_robot,
        terminal_cost_multiplier=_float(config, "terminal_cost_multiplier", 10.0),
    )

    if planner.horizon <= 0:
        raise ConfigurationError("'horizon' must be positive.")
    if planner.terminal_cost_multiplier <= 0:
        raise ConfigurationError("'terminal_cost_multiplier' must be positive.")
    return planner

def _preserve_db_lacam_config(raw_config: Mapping[str, Any], config_out: dict[str, Any]) -> None:
    
    db_lacam_config = raw_config.get("db_lacam")

    if db_lacam_config is None:
        return

    if not isinstance(db_lacam_config, Mapping):
        raise ConfigurationError("'db_lacam' must be a mapping.")

    config_out["db_lacam"] = dict(db_lacam_config)
    
def load_and_validate_system_config(system_name: str, config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and return a validated config dictionary for a system."""
    raw = load_yaml_config(config_path)
    return validate_system_config(system_name=system_name, raw_config=raw)


def validate_system_config(system_name: str, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate system and planner fields, returning normalized config values."""
    if system_name == "multi_robot":
        dt = _float(raw_config, "dt", 0.05)
        if dt <= 0:
            raise ConfigurationError("'dt' must be positive.")

        d_safe = _float(raw_config, "d_safe", 0.0)
        if d_safe < 0:
            raise ConfigurationError("'d_safe' must be non-negative.")

        error_tolerance = _float(raw_config, "error_tolerance", 0.05)
        if error_tolerance <= 0:
            raise ConfigurationError("'error_tolerance' must be positive.")

        robots_raw = raw_config.get("robots")
        if not isinstance(robots_raw, list) or len(robots_raw) == 0:
            raise ConfigurationError("'robots' must be a non-empty list of robot configurations.")

        visibility_raw = raw_config.get("inter_robot_visibility_radius", float("inf"))
        visibility_radius: float | tuple[float, ...]
        if isinstance(visibility_raw, (int, float)):
            visibility_value = float(visibility_raw)
            if visibility_value < 0:
                raise ConfigurationError("'inter_robot_visibility_radius' must be non-negative.")
            visibility_radius = visibility_value
        elif isinstance(visibility_raw, list):
            if len(visibility_raw) != len(robots_raw):
                raise ConfigurationError(
                    "When provided as a list, 'inter_robot_visibility_radius' length must match number of robots "
                    f"({len(robots_raw)}), got {len(visibility_raw)}."
                )
            normalized_radii: list[float] = []
            for robot_idx, value in enumerate(visibility_raw):
                if not isinstance(value, (int, float)):
                    raise ConfigurationError(
                        f"'inter_robot_visibility_radius[{robot_idx}]' must be numeric."
                    )
                value_f = float(value)
                if value_f < 0:
                    raise ConfigurationError(
                        f"'inter_robot_visibility_radius[{robot_idx}]' must be non-negative."
                    )
                normalized_radii.append(value_f)
            visibility_radius = tuple(normalized_radii)
        else:
            raise ConfigurationError(
                "'inter_robot_visibility_radius' must be either a number or a per-robot list of numbers."
            )

        members: list[MultiRobotMemberConfig] = []
        total_nx = 0
        robot_action_dims: list[int] = []
        for robot_idx, robot_entry in enumerate(robots_raw):
            if not isinstance(robot_entry, Mapping):
                raise ConfigurationError(
                    f"'robots[{robot_idx}]' must be a mapping with keys 'system' and 'config'."
                )

            robot_system = robot_entry.get("system")
            if not isinstance(robot_system, str):
                raise ConfigurationError(f"'robots[{robot_idx}].system' must be a string.")

            if robot_system == "multi_robot":
                raise ConfigurationError("Nested 'multi_robot' entries are not supported.")

            robot_cfg_raw = robot_entry.get("config", {})
            if not isinstance(robot_cfg_raw, Mapping):
                raise ConfigurationError(f"'robots[{robot_idx}].config' must be a mapping.")

            robot_cfg_with_dt = dict(robot_cfg_raw)
            robot_cfg_with_dt.setdefault("dt", dt)

            validated_robot_cfg = validate_system_config(robot_system, robot_cfg_with_dt)
            members.append(
                MultiRobotMemberConfig(system=robot_system, config=validated_robot_cfg)
            )

            q_diag = validated_robot_cfg.get("Q_diag")
            if not isinstance(q_diag, list):
                raise ConfigurationError(
                    f"Validated config for robot {robot_idx} is missing planner 'Q_diag'."
                )
            r_diag = validated_robot_cfg.get("R_diag")
            if not isinstance(r_diag, list):
                raise ConfigurationError(
                    f"Validated config for robot {robot_idx} is missing planner 'R_diag'."
                )
            total_nx += len(q_diag)
            robot_action_dims.append(len(r_diag))

        fleet_cfg = MultiRobotSystemConfig(
            dt=dt,
            d_safe=d_safe,
            robots=tuple(members),
            inter_robot_visibility_radius=visibility_radius,
            error_tolerance=error_tolerance,
            initial_state_seed=_optional_int(raw_config, "initial_state_seed"),
        )

        config_out: dict[str, Any] = {
            "dt": fleet_cfg.dt,
            "d_safe": fleet_cfg.d_safe,
            "error_tolerance": fleet_cfg.error_tolerance,
            "robots": [
                {
                    "system": member.system,
                    "config": member.config,
                }
                for member in fleet_cfg.robots
            ],
            "inter_robot_visibility_radius": (
                list(fleet_cfg.inter_robot_visibility_radius)
                if isinstance(fleet_cfg.inter_robot_visibility_radius, tuple)
                else fleet_cfg.inter_robot_visibility_radius
            ),
        }
        if fleet_cfg.initial_state_seed is not None:
            config_out["initial_state_seed"] = fleet_cfg.initial_state_seed

        planner_cfg = _validate_planner(
            raw_config,
            nx=total_nx,
            nu=sum(robot_action_dims),
            robot_action_dims=tuple(robot_action_dims),
        )
        config_out.update(
            {
                "horizon": planner_cfg.horizon,
                "mode": planner_cfg.mode,
                "Q_diag": list(planner_cfg.q_diag),
                "R_weight": planner_cfg.r_weight,
                "R_diag": list(planner_cfg.r_diag),
                "terminal_cost_multiplier": planner_cfg.terminal_cost_multiplier,
            }
        )
        if len(planner_cfg.r_weight_per_robot) > 0:
            config_out["R_weight_per_robot"] = [list(values) for values in planner_cfg.r_weight_per_robot]
            
        _preserve_db_lacam_config(raw_config, config_out)
        
        return config_out

    dt = _float(raw_config, "dt", 0.05)
    if dt <= 0:
        raise ConfigurationError("'dt' must be positive.")

    error_tolerance = _float(raw_config, "error_tolerance", 0.05)
    if error_tolerance <= 0:
        raise ConfigurationError("'error_tolerance' must be positive.")

    config_out: dict[str, Any]
    nx: int

    if system_name == "single_integrator":
        goal_sampling = _goal_sampling(raw_config)
        initial_state_sampling = _initial_state_sampling(
            raw_config,
            error_tolerance=error_tolerance,
        )
        system_cfg = SingleIntegratorSystemConfig(
            dt=dt,
            goal=_vector(raw_config, "goal", 2, (0.0, 0.0)),
            randomize_goal=_bool(raw_config, "randomize_goal", "goal" not in raw_config),
            max_vel=_float(raw_config, "max_vel", 1.0),
            error_tolerance=error_tolerance,
            goal_sampling=goal_sampling,
            initial_state_sampling=initial_state_sampling,
        )
        if system_cfg.max_vel <= 0:
            raise ConfigurationError("'max_vel' must be positive.")
        config_out = {
            "dt": system_cfg.dt,
            "goal": list(system_cfg.goal),
            "randomize_goal": system_cfg.randomize_goal,
            "goal_position_bounds": list(system_cfg.goal_sampling.position_bounds),
            "max_vel": system_cfg.max_vel,
            "error_tolerance": system_cfg.error_tolerance,
            "initial_position_min_goal_distance": system_cfg.initial_state_sampling.min_goal_distance,
            "initial_position_radius_bounds": list(system_cfg.initial_state_sampling.position_radius_bounds),
        }
        if system_cfg.initial_state_sampling.seed is not None:
            config_out["initial_state_seed"] = system_cfg.initial_state_sampling.seed
        nx = 2
        nu = 2

    elif system_name == "double_integrator":
        goal_sampling = _goal_sampling(raw_config)
        initial_state_sampling = _initial_state_sampling(
            raw_config,
            error_tolerance=error_tolerance,
        )
        system_cfg = DoubleIntegratorSystemConfig(
            dt=dt,
            goal=_vector(raw_config, "goal", 2, (0.0, 0.0)),
            randomize_goal=_bool(raw_config, "randomize_goal", "goal" not in raw_config),
            max_accel=_float(raw_config, "max_accel", 2.0),
            error_tolerance=error_tolerance,
            goal_sampling=goal_sampling,
            initial_state_sampling=initial_state_sampling,
        )
        if system_cfg.max_accel <= 0:
            raise ConfigurationError("'max_accel' must be positive.")
        config_out = {
            "dt": system_cfg.dt,
            "goal": list(system_cfg.goal),
            "randomize_goal": system_cfg.randomize_goal,
            "goal_position_bounds": list(system_cfg.goal_sampling.position_bounds),
            "max_accel": system_cfg.max_accel,
            "error_tolerance": system_cfg.error_tolerance,
            "initial_position_min_goal_distance": system_cfg.initial_state_sampling.min_goal_distance,
            "initial_position_radius_bounds": list(system_cfg.initial_state_sampling.position_radius_bounds),
        }
        if system_cfg.initial_state_sampling.seed is not None:
            config_out["initial_state_seed"] = system_cfg.initial_state_sampling.seed
        nx = 4
        nu = 2

    elif system_name == "unicycle1":
        goal_sampling = _goal_sampling(raw_config)
        initial_state_sampling = _initial_state_sampling(
            raw_config,
            error_tolerance=error_tolerance,
        )
        system_cfg = Unicycle1SystemConfig(
            dt=dt,
            goal=_vector(raw_config, "goal", 3, (0.0, 0.0, 0.0)),
            randomize_goal=_bool(raw_config, "randomize_goal", "goal" not in raw_config),
            max_v=_float(raw_config, "max_v", 2.0),
            error_tolerance=error_tolerance,
            goal_sampling=goal_sampling,
            initial_state_sampling=initial_state_sampling,
        )
        if system_cfg.max_v <= 0:
            raise ConfigurationError("'max_v' must be positive.")
        config_out = {
            "dt": system_cfg.dt,
            "goal": list(system_cfg.goal),
            "randomize_goal": system_cfg.randomize_goal,
            "goal_position_bounds": list(system_cfg.goal_sampling.position_bounds),
            "max_v": system_cfg.max_v,
            "error_tolerance": system_cfg.error_tolerance,
            "initial_position_min_goal_distance": system_cfg.initial_state_sampling.min_goal_distance,
            "initial_position_radius_bounds": list(system_cfg.initial_state_sampling.position_radius_bounds),
        }
        if system_cfg.initial_state_sampling.seed is not None:
            config_out["initial_state_seed"] = system_cfg.initial_state_sampling.seed
        nx = 3
        nu = 2

    elif system_name == "unicycle2":
        goal_sampling = _goal_sampling(raw_config)
        initial_state_sampling = _initial_state_sampling(
            raw_config,
            error_tolerance=error_tolerance,
        )
        system_cfg = Unicycle2SystemConfig(
            dt=dt,
            goal=_vector(raw_config, "goal", 3, (0.0, 0.0, 0.0)),
            randomize_goal=_bool(raw_config, "randomize_goal", "goal" not in raw_config),
            randomize_initial_velocity=_bool(raw_config, "randomize_initial_velocity", False),
            max_accel=_float(raw_config, "max_accel", 0.25),
            max_speed=_float(raw_config, "max_speed", 0.5),
            max_omega=_float(raw_config, "max_omega", 0.5),
            error_tolerance=error_tolerance,
            goal_sampling=goal_sampling,
            initial_state_sampling=initial_state_sampling,
        )
        if system_cfg.max_accel <= 0 or system_cfg.max_speed <= 0 or system_cfg.max_omega <= 0:
            raise ConfigurationError("'max_accel', 'max_speed', and 'max_omega' must be positive.")
        config_out = {
            "dt": system_cfg.dt,
            "goal": list(system_cfg.goal),
            "randomize_goal": system_cfg.randomize_goal,
            "goal_position_bounds": list(system_cfg.goal_sampling.position_bounds),
            "randomize_initial_velocity": system_cfg.randomize_initial_velocity,
            "max_accel": system_cfg.max_accel,
            "max_speed": system_cfg.max_speed,
            "max_omega": system_cfg.max_omega,
            "error_tolerance": system_cfg.error_tolerance,
            "initial_position_min_goal_distance": system_cfg.initial_state_sampling.min_goal_distance,
            "initial_position_radius_bounds": list(system_cfg.initial_state_sampling.position_radius_bounds),
        }
        if system_cfg.initial_state_sampling.seed is not None:
            config_out["initial_state_seed"] = system_cfg.initial_state_sampling.seed
        nx = 5
        nu = 2

    else:
        raise ConfigurationError(f"Unknown system '{system_name}'.")

    planner_cfg = _validate_planner(raw_config, nx=nx, nu=nu)
    config_out.update(
        {
            "horizon": planner_cfg.horizon,
            "mode": planner_cfg.mode,
            "Q_diag": list(planner_cfg.q_diag),
            "R_weight": planner_cfg.r_weight,
            "R_diag": list(planner_cfg.r_diag),
            "terminal_cost_multiplier": planner_cfg.terminal_cost_multiplier,
        }
    )
    _preserve_db_lacam_config(raw_config, config_out)
    return config_out
