from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


_REPO_ROOT = Path(__file__).resolve().parent.parent


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
    collision_slack_penalty_weight: float = 10000.0


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
    pos_tol: float = 0.05
    theta_tol: float = 0.05
    vel_tol: float = 0.05
    omega_tol: float = 0.05
    goal_sampling: GoalSamplingConfig = GoalSamplingConfig()
    initial_state_sampling: InitialStateSamplingConfig = InitialStateSamplingConfig(
        position_radius_bounds=(0.05, 1.0),
        min_goal_distance=0.05,
    )


@dataclass(frozen=True)
class MultiRobotMemberConfig:
    system: str
    config: dict[str, Any]
    start: tuple[float, ...] | None = None


@dataclass(frozen=True)
class MultiRobotSystemConfig:
    dt: float = 0.05
    d_safe: float = 0.0
    d_collision: float = 0.0
    robots: tuple[MultiRobotMemberConfig, ...] = ()
    inter_robot_visibility_radius: float | tuple[float, ...] = float("inf")
    error_tolerance: float = 0.05
    initial_state_seed: int | None = None


@dataclass(frozen=True)
class MLPArchitectureConfig:
    hidden_dims: tuple[int, ...]

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


def validate_mlp_architecture_config(raw_config: Mapping[str, Any]) -> MLPArchitectureConfig:
    """Validate MLP architecture YAML for custom DAgger training."""
    model_section = raw_config.get("model", raw_config)
    if not isinstance(model_section, Mapping):
        raise ConfigurationError(
            "Policy config must be a mapping with either a top-level 'hidden_dims' key "
            "or a nested 'model.hidden_dims' key."
        )

    hidden_dims_raw = model_section.get("hidden_dims")
    if hidden_dims_raw is None:
        raise ConfigurationError(
            "Missing 'hidden_dims' in policy config. "
            "Expected e.g. model: {hidden_dims: [256, 256, 128]}."
        )
    if not isinstance(hidden_dims_raw, list) or len(hidden_dims_raw) == 0:
        raise ConfigurationError("'hidden_dims' must be a non-empty list of positive integers.")

    hidden_dims: list[int] = []
    for idx, width in enumerate(hidden_dims_raw):
        if isinstance(width, bool) or not isinstance(width, int):
            raise ConfigurationError(
                f"'hidden_dims[{idx}]' must be int, got {type(width).__name__}."
            )
        if width <= 0:
            raise ConfigurationError(f"'hidden_dims[{idx}]' must be positive.")
        hidden_dims.append(int(width))

    return MLPArchitectureConfig(hidden_dims=tuple(hidden_dims))


def load_and_validate_mlp_architecture_config(config_path: str | Path) -> MLPArchitectureConfig:
    """Load and validate custom MLP architecture YAML."""
    raw = load_yaml_config(config_path)
    return validate_mlp_architecture_config(raw)


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


def _optional_positive_int(config: Mapping[str, Any], key: str) -> int | None:
    value = _optional_int(config, key)
    if value is None:
        return None
    if value <= 0:
        raise ConfigurationError(f"'{key}' must be positive when provided.")
    return value


def _optional_non_negative_int(config: Mapping[str, Any], key: str) -> int | None:
    value = _optional_int(config, key)
    if value is None:
        return None
    if value < 0:
        raise ConfigurationError(f"'{key}' must be non-negative when provided.")
    return value


def _resolved_action_noise_seed(config: Mapping[str, Any]) -> int:
    action_noise_seed = _optional_non_negative_int(config, "action_noise_seed")
    if action_noise_seed is not None:
        return action_noise_seed

    initial_state_seed = _optional_non_negative_int(config, "initial_state_seed")
    if initial_state_seed is not None:
        return initial_state_seed

    return 0


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
    default_min_goal_distance: float,
) -> InitialStateSamplingConfig:
    min_goal_distance = _float(
        raw_config,
        "initial_position_min_goal_distance",
        default_min_goal_distance,
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
        seed=_optional_non_negative_int(raw_config, "initial_state_seed"),
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


def _state_dimension_for_system(system_name: str) -> int:
    if system_name == "single_integrator":
        return 2
    if system_name == "double_integrator":
        return 4
    if system_name == "unicycle1":
        return 3
    if system_name == "unicycle2":
        return 5
    raise ConfigurationError(f"Unknown system '{system_name}' when validating robot start state.")


def _require_only_known_keys(raw: Mapping[str, Any], allowed_keys: set[str], *, context: str) -> None:
    unknown_keys = sorted(key for key in raw.keys() if key not in allowed_keys)
    if unknown_keys:
        formatted = ", ".join(f"'{key}'" for key in unknown_keys)
        raise ConfigurationError(f"Unknown keys in {context}: {formatted}.")


def _allowed_system_config_keys(system_name: str) -> set[str]:
    common_keys = {
        "dt",
        "horizon",
        "mode",
        "Q_diag",
        "R_weight",
        "R_diag",
        "terminal_cost_multiplier",
        "collision_slack_penalty_weight",
        "initial_state_seed",
        "action_noise_seed",
        "done_hold_steps",
        "environment",
        "db_lacam",
        "start",
    }

    if system_name == "single_integrator":
        return common_keys | {
            "goal",
            "randomize_goal",
            "max_vel",
            "error_tolerance",
            "goal_position_bounds",
            "initial_position_min_goal_distance",
            "initial_position_radius_bounds",
        }
    if system_name == "double_integrator":
        return common_keys | {
            "goal",
            "randomize_goal",
            "max_accel",
            "error_tolerance",
            "goal_position_bounds",
            "initial_position_min_goal_distance",
            "initial_position_radius_bounds",
        }
    if system_name == "unicycle1":
        return common_keys | {
            "goal",
            "randomize_goal",
            "max_v",
            "error_tolerance",
            "goal_position_bounds",
            "initial_position_min_goal_distance",
            "initial_position_radius_bounds",
        }
    if system_name == "unicycle2":
        return common_keys | {
            "goal",
            "randomize_goal",
            "randomize_initial_velocity",
            "max_accel",
            "max_speed",
            "max_omega",
            "error_tolerance",
            "pos_tol",
            "theta_tol",
            "vel_tol",
            "omega_tol",
            "goal_position_bounds",
            "initial_position_min_goal_distance",
            "initial_position_radius_bounds",
        }
    raise ConfigurationError(f"Unknown system '{system_name}' when validating YAML keys.")


def _allowed_multi_robot_keys() -> set[str]:
    return {
        "dt",
        "d_safe",
        "d_collision",
        "robots",
        "inter_robot_visibility_radius",
        "error_tolerance",
        "initial_state_seed",
        "action_noise_seed",
        "done_hold_steps",
        "horizon",
        "mode",
        "Q_diag",
        "R_weight",
        "R_diag",
        "R_weight_per_robot",
        "terminal_cost_multiplier",
        "collision_slack_penalty_weight",
        "environment",
        "db_lacam",
    }


def _allowed_multi_robot_entry_keys() -> set[str]:
    return {"system", "start", "config"}


def _allowed_homogeneous_fleet_shorthand_keys() -> set[str]:
    return {"num_robots", "system", "config"}


def _expand_homogeneous_fleet_shorthand(robots_raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the {num_robots, system, config} homogeneous-fleet shorthand into a full robots list.

    Lets a config express N identical robots once instead of repeating the same
    {system, config} entry N times; heterogeneous fleets (mixed systems, per-robot
    'start') still use the long list-of-entries form.
    """
    _require_only_known_keys(
        robots_raw,
        _allowed_homogeneous_fleet_shorthand_keys(),
        context="'robots' homogeneous-fleet shorthand",
    )
    if "num_robots" not in robots_raw:
        raise ConfigurationError("'robots' homogeneous-fleet shorthand requires 'num_robots'.")
    num_robots = robots_raw["num_robots"]
    if not isinstance(num_robots, int) or isinstance(num_robots, bool) or num_robots <= 0:
        raise ConfigurationError("'robots.num_robots' must be a positive integer.")
    if "system" not in robots_raw or not isinstance(robots_raw["system"], str):
        raise ConfigurationError("'robots.system' must be a string in the homogeneous-fleet shorthand.")
    shared_config = robots_raw.get("config", {})
    if not isinstance(shared_config, Mapping):
        raise ConfigurationError("'robots.config' must be a mapping in the homogeneous-fleet shorthand.")
    return [
        {"system": robots_raw["system"], "config": dict(shared_config)}
        for _ in range(num_robots)
    ]


def _validate_environment_config(raw_environment: Any, *, key_name: str) -> dict[str, Any]:
    if not isinstance(raw_environment, Mapping):
        raise ConfigurationError(f"'{key_name}' must be a mapping.")

    if "min" not in raw_environment or "max" not in raw_environment:
        raise ConfigurationError(f"'{key_name}' must define both 'min' and 'max'.")

    environment_min = _vector(raw_environment, "min", 2, (-6.0, -6.0))
    environment_max = _vector(raw_environment, "max", 2, (6.0, 6.0))
    if environment_min[0] >= environment_max[0] or environment_min[1] >= environment_max[1]:
        raise ConfigurationError(f"'{key_name}.min' must be strictly smaller than '{key_name}.max'.")

    obstacles = raw_environment.get("obstacles", [])
    if not isinstance(obstacles, list):
        raise ConfigurationError(f"'{key_name}.obstacles' must be a list.")

    validated_environment: dict[str, Any] = {
        "min": list(environment_min),
        "max": list(environment_max),
        "obstacles": list(obstacles),
    }
    return validated_environment


def _resolve_existing_path(
    raw_path: str,
    *,
    field_name: str,
    config_dir: Path | None,
    expect_dir: bool = False,
    expect_file: bool = False,
) -> Path:
    if expect_dir and expect_file:
        raise ConfigurationError(
            f"'{field_name}' cannot require both a file and a directory."
        )

    path = Path(raw_path).expanduser()

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if config_dir is not None:
            candidates.append(config_dir / path)
        candidates.append(_REPO_ROOT / path)
        candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if expect_dir:
            if candidate.is_dir():
                return candidate.resolve()
        elif expect_file:
            if candidate.is_file():
                return candidate.resolve()
        elif candidate.exists():
            return candidate.resolve()

    kind = "directory" if expect_dir else "file" if expect_file else "path"
    raise ConfigurationError(f"'{field_name}' does not exist ({kind}): {raw_path}")


def _validate_db_lacam_config(
    raw_config: Mapping[str, Any],
    *,
    config_dir: Path | None = None,
) -> dict[str, Any] | None:
    raw_db_lacam = raw_config.get("db_lacam")
    if raw_db_lacam is None:
        return None
    if not isinstance(raw_db_lacam, Mapping):
        raise ConfigurationError("'db_lacam' must be a mapping.")

    mode = raw_db_lacam.get("mode", "replan")
    if not isinstance(mode, str):
        raise ConfigurationError("'db_lacam.mode' must be a string.")
    mode = mode.lower()
    if mode not in {"replan", "open_loop"}:
        raise ConfigurationError("'db_lacam.mode' must be one of {'replan', 'open_loop'}.")

    validated_db_lacam: dict[str, Any] = {"mode": mode}

    replan_freq = raw_db_lacam.get("replan_freq", 5)
    if not isinstance(replan_freq, int) or isinstance(replan_freq, bool) or replan_freq <= 0:
        raise ConfigurationError("'db_lacam.replan_freq' must be a positive integer.")
    validated_db_lacam["replan_freq"] = int(replan_freq)

    time_limit_ms = raw_db_lacam.get("time_limit_ms", 60_000)
    if not isinstance(time_limit_ms, int) or isinstance(time_limit_ms, bool) or time_limit_ms <= 0:
        raise ConfigurationError("'db_lacam.time_limit_ms' must be a positive integer.")
    validated_db_lacam["time_limit_ms"] = int(time_limit_ms)

    algorithm_config = raw_db_lacam.get("algorithm_config")
    if not isinstance(algorithm_config, str) or algorithm_config.strip() == "":
        raise ConfigurationError("'db_lacam.algorithm_config' is required and must be a non-empty string.")
    validated_db_lacam["algorithm_config"] = str(
        _resolve_existing_path(
            algorithm_config,
            field_name="db_lacam.algorithm_config",
            config_dir=config_dir,
            expect_file=True,
        )
    )

    executable = raw_db_lacam.get("executable")
    if executable is not None:
        if not isinstance(executable, str) or executable.strip() == "":
            raise ConfigurationError("'db_lacam.executable' must be a non-empty string when provided.")
        validated_db_lacam["executable"] = str(
            _resolve_existing_path(
                executable,
                field_name="db_lacam.executable",
                config_dir=config_dir,
                expect_file=True,
            )
        )

    cwd = raw_db_lacam.get("cwd")
    if cwd is not None:
        if not isinstance(cwd, str) or cwd.strip() == "":
            raise ConfigurationError("'db_lacam.cwd' must be a non-empty string when provided.")
        validated_db_lacam["cwd"] = str(
            _resolve_existing_path(
                cwd,
                field_name="db_lacam.cwd",
                config_dir=config_dir,
                expect_dir=True,
            )
        )

    raise_planning_error = raw_db_lacam.get("raise_planning_error", True)
    if not isinstance(raise_planning_error, bool):
        raise ConfigurationError("'db_lacam.raise_planning_error' must be bool.")
    validated_db_lacam["raise_planning_error"] = raise_planning_error

    if "environment" in raw_db_lacam:
        validated_db_lacam["environment"] = _validate_environment_config(
            raw_db_lacam["environment"],
            key_name="db_lacam.environment",
        )

    return validated_db_lacam


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


def _validate_multi_robot_start(
    robot_idx: int,
    robot_system: str,
    robot_entry: Mapping[str, Any],
    robot_cfg_raw: Mapping[str, Any],
) -> tuple[float, ...] | None:
    expected_dim = _state_dimension_for_system(robot_system)

    start_from_top = robot_entry.get("start")
    start_from_cfg = robot_cfg_raw.get("start")

    if start_from_top is None and start_from_cfg is None:
        return None

    validated_start: tuple[float, ...] | None = None

    if start_from_top is not None:
        validated_start = _vector(robot_entry, "start", expected_dim, tuple([0.0] * expected_dim))

    if start_from_cfg is not None:
        cfg_start = _vector(robot_cfg_raw, "start", expected_dim, tuple([0.0] * expected_dim))
        if validated_start is not None and cfg_start != validated_start:
            raise ConfigurationError(
                f"'robots[{robot_idx}].start' and 'robots[{robot_idx}].config.start' must match when both are provided."
            )
        validated_start = cfg_start

    return validated_start


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
        collision_slack_penalty_weight=_float(config, "collision_slack_penalty_weight", 10000.0),
    )

    if planner.horizon <= 0:
        raise ConfigurationError("'horizon' must be positive.")
    if planner.terminal_cost_multiplier <= 0:
        raise ConfigurationError("'terminal_cost_multiplier' must be positive.")
    if planner.collision_slack_penalty_weight <= 0:
        raise ConfigurationError("'collision_slack_penalty_weight' must be positive.")
    return planner

def load_and_validate_system_config(system_name: str, config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and return a validated config dictionary for a system."""
    raw = load_yaml_config(config_path)
    return validate_system_config(
        system_name=system_name,
        raw_config=raw,
        config_dir=Path(config_path).expanduser().resolve().parent,
    )


def validate_system_config(
    system_name: str,
    raw_config: Mapping[str, Any],
    *,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate system and planner fields, returning normalized config values."""
    if system_name == "multi_robot":
        _require_only_known_keys(raw_config, _allowed_multi_robot_keys(), context="'multi_robot' config")

        dt = _float(raw_config, "dt", 0.05)
        if dt <= 0:
            raise ConfigurationError("'dt' must be positive.")

        d_safe = _float(raw_config, "d_safe", 0.0)
        if d_safe < 0:
            raise ConfigurationError("'d_safe' must be non-negative.")
        d_collision = _float(raw_config, "d_collision", d_safe)
        if d_collision < 0:
            raise ConfigurationError("'d_collision' must be non-negative.")
        if d_collision > d_safe:
            raise ConfigurationError(
                "'d_collision' must not exceed 'd_safe': d_safe is the planner's soft buffer "
                "distance and d_collision is the physical collision threshold, so d_collision "
                "> d_safe would let the expert plan states the simulator immediately flags as "
                "collisions."
            )

        error_tolerance = _float(raw_config, "error_tolerance", 0.05)
        if error_tolerance <= 0:
            raise ConfigurationError("'error_tolerance' must be positive.")

        robots_raw = raw_config.get("robots")
        if isinstance(robots_raw, Mapping):
            robots_raw = _expand_homogeneous_fleet_shorthand(robots_raw)
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

            _require_only_known_keys(robot_entry, _allowed_multi_robot_entry_keys(), context=f"'robots[{robot_idx}]'")

            robot_system = robot_entry.get("system")
            if not isinstance(robot_system, str):
                raise ConfigurationError(f"'robots[{robot_idx}].system' must be a string.")

            if robot_system == "multi_robot":
                raise ConfigurationError("Nested 'multi_robot' entries are not supported.")

            robot_cfg_raw = robot_entry.get("config", {})
            if not isinstance(robot_cfg_raw, Mapping):
                raise ConfigurationError(f"'robots[{robot_idx}].config' must be a mapping.")

            _require_only_known_keys(
                robot_cfg_raw,
                _allowed_system_config_keys(robot_system),
                context=f"'robots[{robot_idx}].config'",
            )

            robot_start = _validate_multi_robot_start(
                robot_idx=robot_idx,
                robot_system=robot_system,
                robot_entry=robot_entry,
                robot_cfg_raw=robot_cfg_raw,
            )

            robot_cfg_with_dt = dict(robot_cfg_raw)
            robot_cfg_with_dt.setdefault("dt", dt)

            validated_robot_cfg = validate_system_config(
                robot_system,
                robot_cfg_with_dt,
                config_dir=config_dir,
            )
            members.append(
                MultiRobotMemberConfig(
                    system=robot_system,
                    config=validated_robot_cfg,
                    start=robot_start,
                )
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
            d_collision=d_collision,
            robots=tuple(members),
            inter_robot_visibility_radius=visibility_radius,
            error_tolerance=error_tolerance,
            initial_state_seed=_optional_non_negative_int(raw_config, "initial_state_seed"),
        )
        action_noise_seed = _resolved_action_noise_seed(raw_config)
        done_hold_steps = _optional_positive_int(raw_config, "done_hold_steps")

        config_out: dict[str, Any] = {
            "dt": fleet_cfg.dt,
            "d_safe": fleet_cfg.d_safe,
            "d_collision": fleet_cfg.d_collision,
            "error_tolerance": fleet_cfg.error_tolerance,
            "action_noise_seed": action_noise_seed,
            "robots": [
                {
                    "system": member.system,
                    "config": member.config,
                    **({"start": list(member.start)} if member.start is not None else {}),
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
        if done_hold_steps is not None:
            config_out["done_hold_steps"] = done_hold_steps

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
                "collision_slack_penalty_weight": planner_cfg.collision_slack_penalty_weight,
            }
        )
        if len(planner_cfg.r_weight_per_robot) > 0:
            config_out["R_weight_per_robot"] = [list(values) for values in planner_cfg.r_weight_per_robot]

        if "environment" in raw_config:
            config_out["environment"] = _validate_environment_config(raw_config["environment"], key_name="environment")

        validated_db_lacam = _validate_db_lacam_config(raw_config, config_dir=config_dir)
        if validated_db_lacam is not None:
            config_out["db_lacam"] = validated_db_lacam
            if "environment" not in config_out and "environment" in validated_db_lacam:
                config_out["environment"] = validated_db_lacam["environment"]
        
        return config_out

    dt = _float(raw_config, "dt", 0.05)
    if dt <= 0:
        raise ConfigurationError("'dt' must be positive.")

    done_hold_steps = _optional_positive_int(raw_config, "done_hold_steps")
    action_noise_seed = _resolved_action_noise_seed(raw_config)

    config_out: dict[str, Any]
    nx: int

    if system_name == "single_integrator":
        _require_only_known_keys(raw_config, _allowed_system_config_keys(system_name), context="'single_integrator' config")

        error_tolerance = _float(raw_config, "error_tolerance", 0.05)
        if error_tolerance <= 0:
            raise ConfigurationError("'error_tolerance' must be positive.")
        goal_sampling = _goal_sampling(raw_config)
        initial_state_sampling = _initial_state_sampling(
            raw_config,
            default_min_goal_distance=error_tolerance,
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
            "action_noise_seed": action_noise_seed,
            "initial_position_min_goal_distance": system_cfg.initial_state_sampling.min_goal_distance,
            "initial_position_radius_bounds": list(system_cfg.initial_state_sampling.position_radius_bounds),
        }
        if system_cfg.initial_state_sampling.seed is not None:
            config_out["initial_state_seed"] = system_cfg.initial_state_sampling.seed
        nx = 2
        nu = 2

    elif system_name == "double_integrator":
        _require_only_known_keys(raw_config, _allowed_system_config_keys(system_name), context="'double_integrator' config")

        error_tolerance = _float(raw_config, "error_tolerance", 0.05)
        if error_tolerance <= 0:
            raise ConfigurationError("'error_tolerance' must be positive.")
        goal_sampling = _goal_sampling(raw_config)
        initial_state_sampling = _initial_state_sampling(
            raw_config,
            default_min_goal_distance=error_tolerance,
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
            "action_noise_seed": action_noise_seed,
            "initial_position_min_goal_distance": system_cfg.initial_state_sampling.min_goal_distance,
            "initial_position_radius_bounds": list(system_cfg.initial_state_sampling.position_radius_bounds),
        }
        if system_cfg.initial_state_sampling.seed is not None:
            config_out["initial_state_seed"] = system_cfg.initial_state_sampling.seed
        nx = 4
        nu = 2

    elif system_name == "unicycle1":
        _require_only_known_keys(raw_config, _allowed_system_config_keys(system_name), context="'unicycle1' config")

        error_tolerance = _float(raw_config, "error_tolerance", 0.05)
        if error_tolerance <= 0:
            raise ConfigurationError("'error_tolerance' must be positive.")
        goal_sampling = _goal_sampling(raw_config)
        initial_state_sampling = _initial_state_sampling(
            raw_config,
            default_min_goal_distance=error_tolerance,
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
            "action_noise_seed": action_noise_seed,
            "initial_position_min_goal_distance": system_cfg.initial_state_sampling.min_goal_distance,
            "initial_position_radius_bounds": list(system_cfg.initial_state_sampling.position_radius_bounds),
        }
        if system_cfg.initial_state_sampling.seed is not None:
            config_out["initial_state_seed"] = system_cfg.initial_state_sampling.seed
        nx = 3
        nu = 2

    elif system_name == "unicycle2":
        _require_only_known_keys(raw_config, _allowed_system_config_keys(system_name), context="'unicycle2' config")

        if "error_tolerance" in raw_config:
            raise ConfigurationError(
                "'error_tolerance' is not supported for 'unicycle2'. "
                "Use 'pos_tol', 'theta_tol', 'vel_tol', and 'omega_tol'."
            )
        pos_tol = _float(raw_config, "pos_tol", 0.05)
        theta_tol = _float(raw_config, "theta_tol", 0.05)
        vel_tol = _float(raw_config, "vel_tol", 0.05)
        omega_tol = _float(raw_config, "omega_tol", 0.05)
        if pos_tol <= 0:
            raise ConfigurationError("'pos_tol' must be positive.")
        if theta_tol <= 0:
            raise ConfigurationError("'theta_tol' must be positive.")
        if vel_tol <= 0:
            raise ConfigurationError("'vel_tol' must be positive.")
        if omega_tol <= 0:
            raise ConfigurationError("'omega_tol' must be positive.")
        goal_sampling = _goal_sampling(raw_config)
        initial_state_sampling = _initial_state_sampling(
            raw_config,
            default_min_goal_distance=pos_tol,
        )
        system_cfg = Unicycle2SystemConfig(
            dt=dt,
            goal=_vector(raw_config, "goal", 3, (0.0, 0.0, 0.0)),
            randomize_goal=_bool(raw_config, "randomize_goal", "goal" not in raw_config),
            randomize_initial_velocity=_bool(raw_config, "randomize_initial_velocity", False),
            max_accel=_float(raw_config, "max_accel", 0.25),
            max_speed=_float(raw_config, "max_speed", 0.5),
            max_omega=_float(raw_config, "max_omega", 0.5),
            pos_tol=pos_tol,
            theta_tol=theta_tol,
            vel_tol=vel_tol,
            omega_tol=omega_tol,
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
            "pos_tol": system_cfg.pos_tol,
            "theta_tol": system_cfg.theta_tol,
            "vel_tol": system_cfg.vel_tol,
            "omega_tol": system_cfg.omega_tol,
            "action_noise_seed": action_noise_seed,
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
            "collision_slack_penalty_weight": planner_cfg.collision_slack_penalty_weight,
        }
    )

    if "environment" in raw_config:
        config_out["environment"] = _validate_environment_config(raw_config["environment"], key_name="environment")

    validated_db_lacam = _validate_db_lacam_config(raw_config, config_dir=config_dir)
    if validated_db_lacam is not None:
        config_out["db_lacam"] = validated_db_lacam
        if "environment" not in config_out and "environment" in validated_db_lacam:
            config_out["environment"] = validated_db_lacam["environment"]

    if done_hold_steps is not None:
        config_out["done_hold_steps"] = done_hold_steps
    return config_out
