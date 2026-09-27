from __future__ import annotations

import math
import random
from typing import Mapping

import datasets
import numpy as np
import torch
from lerobot.datasets import compute_stats as _lerobot_compute_stats
from lerobot.datasets import dataset_reader as _lerobot_dataset_reader
from lerobot.datasets import dataset_writer as _lerobot_dataset_writer
from lerobot.datasets import feature_utils as _lerobot_feature_utils

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
    # Only evaluate_policy_rollouts populates this split (curriculum
    # initial_states/goal_states vs simulator RNG fallback); other
    # DaggerEvalMetrics producers (e.g. aggregation's own success
    # accounting) leave both counts at 0, so there's nothing worth a second
    # line for them.
    if metrics.config_num_episodes > 0 or metrics.random_num_episodes > 0:
        config_rate = (
            100.0 * metrics.config_successes / metrics.config_num_episodes
            if metrics.config_num_episodes > 0
            else None
        )
        random_rate = (
            100.0 * metrics.random_successes / metrics.random_num_episodes
            if metrics.random_num_episodes > 0
            else None
        )
        config_display = (
            f"{metrics.config_successes}/{metrics.config_num_episodes} ({config_rate:.1f}%)"
            if config_rate is not None
            else "n/a (0 episodes)"
        )
        random_display = (
            f"{metrics.random_successes}/{metrics.random_num_episodes} ({random_rate:.1f}%)"
            if random_rate is not None
            else "n/a (0 episodes)"
        )
        print(
            f"  {prefix}_success_by_source: config={config_display} random={random_display}"
        )
    total_failures = metrics.collision_failures + metrics.timeout_failures + metrics.solve_failures
    if total_failures > 0:
        print(
            f"  {prefix}_failure_breakdown: collision={metrics.collision_failures} "
            f"timeout={metrics.timeout_failures} solve_failure={metrics.solve_failures} "
            f"(of {total_failures} failures)"
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


def rng_for_seed_spec(simulator: DynamicsProtocol, seed_spec: int | list[int]) -> np.random.Generator:
    """Build the RNG for a seed spec, honoring per-robot seed lists like sample_initial_state does."""
    if isinstance(seed_spec, int):
        return np.random.default_rng(int(seed_spec))
    sub_simulators = simulator.simulators
    if len(seed_spec) != len(sub_simulators):
        raise ValueError(
            "Per-robot seed specification length must match robot count. "
            f"Got {len(seed_spec)} seeds for {len(sub_simulators)} robots."
        )
    return np.random.default_rng(np.random.SeedSequence([int(robot_seed) for robot_seed in seed_spec]))


def sample_initial_state(simulator: DynamicsProtocol, seed_spec: int | list[int]) -> np.ndarray:
    rng = rng_for_seed_spec(simulator, seed_spec)
    simulator.randomize_goal_for_reset(rng)
    return simulator.random_initial_state(rng)


# --- LeRobot compatibility patches, applied below on import ---------------
#
# Some of our systems legitimately report a zero-length dataset feature (e.g.
# observation.state for single_integrator, which has no proprioception
# beyond position -- everything is already captured by the goal-relative
# observation.environment_state). LeRobot's dataset-writing path doesn't
# handle this, in two separate places:
#
# 1. Episode-stats computation: RunningQuantileStats.update() reshapes a
#    batch to (-1, batch.shape[-1]), and when the array is genuinely empty
#    (shape (N, 0)), numpy can't infer the -1 (0 total size / 0 per-row is
#    undefined) and raises ValueError: cannot reshape array of size 0 into
#    shape (0).
# 2. Arrow schema construction: a zero-length 1-D feature becomes
#    datasets.Sequence(length=0, ...), which compiles to a pyarrow
#    FixedSizeListArray of size 0 -- pyarrow rejects this outright
#    (ArrowInvalid: list_size needs to be a strict positive integer) the
#    moment an episode is actually written, regardless of #1.
#
# Both are upstream gaps, not ours to carry a private fork for -- so we patch
# around them here instead. Each patched function is otherwise identical to
# LeRobot's original, with one added branch for the zero-size case.


def _get_feature_stats_zero_size_safe(
    array: np.ndarray,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    quantile_list: list[float] | None = None,
) -> dict[str, np.ndarray]:
    if quantile_list is None:
        quantile_list = _lerobot_compute_stats.DEFAULT_QUANTILES

    original_shape = array.shape
    reshaped, sample_count = _lerobot_compute_stats._prepare_array_for_stats(array, axis)

    if reshaped.shape[0] < 2 or reshaped.size == 0:
        stats = _lerobot_compute_stats._compute_basic_stats(reshaped, sample_count, quantile_list)
    else:
        running_stats = _lerobot_compute_stats.RunningQuantileStats()
        running_stats.update(reshaped)
        stats = running_stats.get_statistics()
        stats["count"] = np.array([sample_count])

    return _lerobot_compute_stats._reshape_stats_by_axis(stats, axis, keepdims, original_shape)


def _get_hf_features_from_features_zero_size_safe(features: dict) -> datasets.Features:
    """Same as LeRobot's original, except a zero-length 1-D feature (e.g.
    observation.state for a system with no proprioception beyond position)
    becomes a variable-length Sequence instead of a fixed-length one.
    Sequence(length=0, ...) compiles to a pyarrow FixedSizeListArray of size
    0, which pyarrow rejects outright (list_size needs to be a strict
    positive integer) the moment an episode with such a feature is written
    -- there is no valid fixed-size encoding for "always empty" in this
    version of pyarrow/datasets. Dropping the fixed length (every row for
    this key is [] anyway) uses a variable-length list instead, which
    pyarrow stores and reads back fine (round-trips to shape (0,), verified
    directly).
    """
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            if ft["shape"][0] == 0:
                hf_features[key] = datasets.Sequence(feature=datasets.Value(dtype=ft["dtype"]))
            else:
                hf_features[key] = datasets.Sequence(
                    length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
                )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")

    return datasets.Features(hf_features)


_lerobot_compute_stats.get_feature_stats = _get_feature_stats_zero_size_safe

# get_hf_features_from_features is imported with `from ... import` in
# several LeRobot modules (a separate name binding each time, not an
# attribute lookup on feature_utils) -- patching feature_utils's own
# attribute alone would silently miss every one of those call sites, so
# each importer's local binding needs patching directly too.
_lerobot_feature_utils.get_hf_features_from_features = _get_hf_features_from_features_zero_size_safe
_lerobot_dataset_writer.get_hf_features_from_features = _get_hf_features_from_features_zero_size_safe
_lerobot_dataset_reader.get_hf_features_from_features = _get_hf_features_from_features_zero_size_safe
