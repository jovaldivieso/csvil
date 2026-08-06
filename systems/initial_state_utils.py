from __future__ import annotations

import ast
from numbers import Real
from typing import Any

import numpy as np

from systems.dynamics import DynamicsProtocol


def parse_initial_states_argument(raw_initial_states: str | None) -> Any | None:
    if raw_initial_states is None:
        return None

    text = raw_initial_states.strip()
    if text == "":
        return None

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            "Unable to parse --initial-states. Use Python-literal syntax like "
            "'[0.5, -0.1, 0.0, 0.0]' (single rollout), "
            "'[[0.5, -0.1, 0.0, 0.0], [0.2, 0.3, 0.0, 0.0]]' (multiple rollouts), or "
            "'[[[x1, y1, ...], [x2, y2, ...]], ...]' (multi-robot per-rollout specs)."
        ) from exc


def _is_numeric_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(x, Real) and not isinstance(x, bool)
        for x in value
    )


def _as_state_vector(values: list[Any], expected_dim: int, context: str) -> np.ndarray:
    if len(values) != expected_dim:
        raise ValueError(
            f"{context} must have length {expected_dim}, got {len(values)}."
        )
    return np.asarray(values, dtype=float)


def normalize_initial_state_specs(
    simulator: DynamicsProtocol,
    initial_states: Any | None,
) -> list[np.ndarray]:
    if initial_states is None:
        return []

    if not isinstance(initial_states, (list, tuple)):
        raise ValueError("--initial-states must evaluate to a list-like object.")

    values = list(initial_states)
    if len(values) == 0:
        return []

    nx = int(simulator.nx)
    num_robots = int(simulator.num_robots)
    robot_dims = [
        int(state_slice.stop - state_slice.start)
        for state_slice in simulator.robot_state_slices
    ]

    # Case 1: one global rollout state, e.g. [x, y, vx, vy]
    if all(isinstance(x, Real) and not isinstance(x, bool) for x in values):
        return [_as_state_vector(values, expected_dim=nx, context="Initial state")]

    if not all(isinstance(item, (list, tuple)) for item in values):
        raise ValueError(
            "--initial-states must be a numeric vector, a list of numeric vectors, "
            "or multi-robot per-rollout nested lists."
        )

    row_values = [list(item) for item in values]

    # Case 2: single multi-robot rollout provided as per-robot states,
    # e.g. [[x1, y1, ...], [x2, y2, ...]]
    if (
        num_robots > 1
        and len(row_values) == num_robots
        and all(_is_numeric_sequence(robot_state) for robot_state in row_values)
        and all(len(robot_state) == robot_dims[idx] for idx, robot_state in enumerate(row_values))
    ):
        parts = [
            np.asarray(robot_state, dtype=float)
            for robot_state in row_values
        ]
        return [np.concatenate(parts)]

    # Case 3: list of global rollout states,
    # e.g. [[...global nx...], [...global nx...]]
    if all(_is_numeric_sequence(row) for row in row_values):
        return [
            _as_state_vector(row, expected_dim=nx, context=f"Initial state #{idx}")
            for idx, row in enumerate(row_values)
        ]

    # Case 4: list of multi-robot rollout states,
    # e.g. [ [[robot1...], [robot2...]], [[robot1...], [robot2...]] ]
    normalized: list[np.ndarray] = []
    for rollout_idx, rollout_state in enumerate(row_values):
        if not isinstance(rollout_state, list) or len(rollout_state) != num_robots:
            raise ValueError(
                "Each multi-robot rollout state must provide one state vector per robot. "
                f"Rollout #{rollout_idx} has {len(rollout_state)} entries, expected {num_robots}."
            )

        robot_parts: list[np.ndarray] = []
        for robot_idx, robot_state in enumerate(rollout_state):
            if not _is_numeric_sequence(robot_state):
                raise ValueError(
                    f"Rollout #{rollout_idx}, robot #{robot_idx} state must be a numeric list."
                )

            expected_dim = robot_dims[robot_idx]
            if len(robot_state) != expected_dim:
                raise ValueError(
                    f"Rollout #{rollout_idx}, robot #{robot_idx} state length must be {expected_dim}, "
                    f"got {len(robot_state)}."
                )
            robot_parts.append(np.asarray(robot_state, dtype=float))

        normalized.append(np.concatenate(robot_parts))

    return normalized