from __future__ import annotations

import math
import random
from typing import Mapping

import numpy as np
import torch

from .metrics import DaggerEvalMetrics
from systems.dynamics import DynamicsProtocol
from systems.seed_utils import DEFAULT_MULTI_ROBOT_SEED_STRIDE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def with_seeded_initial_state_config(
    system_name: str,
    config: Mapping[str, object],
    base_seed: int,
) -> dict[str, object]:
    """Ensure all simulator RNG entrypoints get deterministic initial-state seeds."""
    seeded_config = dict(config)
    seeded_config.setdefault("initial_state_seed", int(base_seed))
    if system_name != "multi_robot":
        return seeded_config
    robots_raw = seeded_config.get("robots", [])
    seeded_robots: list[dict[str, object]] = []
    for robot_idx, robot_entry in enumerate(robots_raw):
        if not isinstance(robot_entry, Mapping):
            seeded_robots.append(dict(robot_entry))
            continue
        robot_cfg_raw = robot_entry.get("config", {})
        robot_cfg = dict(robot_cfg_raw) if isinstance(robot_cfg_raw, Mapping) else {}
        robot_cfg.setdefault("initial_state_seed", int(base_seed + 1000 * (robot_idx + 1)))
        seeded_robots.append({"system": robot_entry.get("system"), "config": robot_cfg})
    seeded_config["robots"] = seeded_robots
    return seeded_config


def apply_config_overrides(
    config: Mapping[str, object],
    overrides: Mapping[str, object],
) -> dict[str, object]:
    """Merge flat key/value overrides into a config dict, per-robot for multi_robot fleets."""
    merged_config = dict(config)
    if not overrides:
        return merged_config
    if "robots" not in merged_config:
        merged_config.update(overrides)
        return merged_config
    robots_raw = merged_config.get("robots", [])
    if isinstance(robots_raw, Mapping):
        shared_config = robots_raw.get("config")
        robot_cfg = dict(shared_config) if isinstance(shared_config, Mapping) else {}
        robot_cfg.update(overrides)
        merged_config["robots"] = {**robots_raw, "config": robot_cfg}
        return merged_config
    merged_robots: list[object] = []
    for robot_entry in robots_raw:
        if not isinstance(robot_entry, Mapping):
            merged_robots.append(robot_entry)
            continue
        existing_config = robot_entry.get("config")
        robot_cfg = dict(existing_config) if isinstance(existing_config, Mapping) else {}
        robot_cfg.update(overrides)
        merged_robots.append({**robot_entry, "config": robot_cfg})
    merged_config["robots"] = merged_robots
    return merged_config


def resolve_initial_state_seed(config: Mapping[str, object], fallback_seed: int) -> int:
    return int(config.get("initial_state_seed", fallback_seed))


def resolve_round_steps(
    num_frames: int,
    batch_size: int,
    target_epochs_per_round: float,
    max_train_steps: int | None,
) -> tuple[int, float]:
    if num_frames <= 0:
        raise ValueError("Training dataset must contain at least one frame.")
    if batch_size <= 0:
        raise ValueError("'batch_size' must be positive.")
    if target_epochs_per_round <= 0:
        raise ValueError("'target_epochs_per_round' must be positive.")
    steps = math.ceil(float(target_epochs_per_round) * float(num_frames) / float(batch_size))
    if max_train_steps is not None:
        steps = min(steps, int(max_train_steps))
    if steps <= 0:
        raise ValueError("Per-round training steps must remain positive.")
    return steps, float(steps) * float(batch_size) / float(num_frames)


def print_rollout_metrics(label: str, prefix: str, metrics: DaggerEvalMetrics) -> None:
    print(
        f"{label}: {prefix}_success_rate={100.0 * metrics.success_rate:.1f}% "
        f"{prefix}_mean_steps={metrics.mean_steps:.2f} "
        f"{prefix}_min_steps={metrics.min_steps} {prefix}_max_steps={metrics.max_steps} "
        f"episodes={metrics.num_episodes}"
    )


def evaluation_seed_specs(
    simulator: DynamicsProtocol,
    num_episodes: int,
    seed_start: int,
) -> list[int | list[int]]:
    if num_episodes < 0:
        raise ValueError("'num_episodes' must be non-negative.")
    if simulator.num_robots <= 1:
        return [int(seed_start) + idx for idx in range(num_episodes)]
    return [
        [int(seed_start) + idx + DEFAULT_MULTI_ROBOT_SEED_STRIDE * robot_idx for robot_idx in range(simulator.num_robots)]
        for idx in range(num_episodes)
    ]


def sample_initial_state(simulator: DynamicsProtocol, seed_spec: int | list[int]) -> np.ndarray:
    if isinstance(seed_spec, int):
        rng = np.random.default_rng(int(seed_spec))
        simulator.randomize_goal_for_reset(rng)
        return simulator.random_initial_state(rng)
    sub_simulators = simulator.simulators
    if len(seed_spec) != len(sub_simulators):
        raise ValueError(
            "Per-robot seed specification length must match robot count. "
            f"Got {len(seed_spec)} seeds for {len(sub_simulators)} robots."
        )
    rng = np.random.default_rng(np.random.SeedSequence([int(robot_seed) for robot_seed in seed_spec]))
    simulator.randomize_goal_for_reset(rng)
    return simulator.random_initial_state(rng)
