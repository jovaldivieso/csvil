from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from systems.dynamics import DynamicsProtocol


DEFAULT_SINGLE_ROBOT_SEED = 1
DEFAULT_MULTI_ROBOT_SEED_BASE = 1
DEFAULT_MULTI_ROBOT_SEED_STRIDE = 100
DEFAULT_ACTION_NOISE_SEED = 0
_ACTION_NOISE_STREAM_ID = 1
_INITIAL_STATE_STREAM_ID = 2


def default_seed_argument_for_simulator(
    simulator: DynamicsProtocol,
) -> list[int] | list[list[int]]:
    if simulator.num_robots <= 1:
        return [DEFAULT_SINGLE_ROBOT_SEED]

    return [
        [DEFAULT_MULTI_ROBOT_SEED_BASE + DEFAULT_MULTI_ROBOT_SEED_STRIDE * robot_idx]
        for robot_idx in range(simulator.num_robots)
    ]


def default_action_noise_seed_for_config(config: Mapping[str, Any]) -> int:
    action_noise_seed = config.get("action_noise_seed")
    if action_noise_seed is None:
        raise KeyError("Validated config is missing 'action_noise_seed'.")
    return int(action_noise_seed)


def action_noise_seed_for_rollout(
    base_seed: int,
    *,
    seed_spec: int | list[int] | None = None,
    rollout_index: int | None = None,
) -> int:
    return _derived_rollout_seed(
        base_seed,
        stream_id=_ACTION_NOISE_STREAM_ID,
        seed_spec=seed_spec,
        rollout_index=rollout_index,
    )


def initial_state_seed_for_rollout(
    base_seed: int,
    *,
    seed_spec: int | list[int] | None = None,
    rollout_index: int | None = None,
) -> int:
    return _derived_rollout_seed(
        base_seed,
        stream_id=_INITIAL_STATE_STREAM_ID,
        seed_spec=seed_spec,
        rollout_index=rollout_index,
    )


def _derived_rollout_seed(
    base_seed: int,
    *,
    stream_id: int,
    seed_spec: int | list[int] | None = None,
    rollout_index: int | None = None,
) -> int:
    entropy = [int(base_seed), int(stream_id)]
    if seed_spec is not None:
        if isinstance(seed_spec, int):
            entropy.append(int(seed_spec))
        else:
            entropy.extend(int(seed) for seed in seed_spec)
    elif rollout_index is not None:
        entropy.append(int(rollout_index))

    seed_sequence = np.random.SeedSequence(entropy)
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def action_noise_rng_for_rollout(
    base_seed: int,
    *,
    seed_spec: int | list[int] | None = None,
    rollout_index: int | None = None,
) -> np.random.Generator:
    return np.random.default_rng(
        action_noise_seed_for_rollout(
            base_seed,
            seed_spec=seed_spec,
            rollout_index=rollout_index,
        )
    )