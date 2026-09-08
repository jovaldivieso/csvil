from __future__ import annotations

import ast
from numbers import Real
from typing import Any

import numpy as np

from systems.dynamics import DynamicsProtocol


def _parse_state_spec_argument(raw: str | None, flag_name: str, examples: str) -> Any | None:
    if raw is None:
        return None

    text = raw.strip()
    if text == "":
        return None

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Unable to parse {flag_name}. Use Python-literal syntax like {examples}"
        ) from exc


def parse_initial_states_argument(raw_initial_states: str | None) -> Any | None:
    return _parse_state_spec_argument(
        raw_initial_states,
        "--initial-states",
        "'[0.5, -0.1, 0.0, 0.0]' (single rollout), "
        "'[[0.5, -0.1, 0.0, 0.0], [0.2, 0.3, 0.0, 0.0]]' (multiple rollouts), or "
        "'[[[x1, y1, ...], [x2, y2, ...]], ...]' (multi-robot per-rollout specs).",
    )


def parse_goal_states_argument(raw_goal_states: str | None) -> Any | None:
    return _parse_state_spec_argument(
        raw_goal_states,
        "--goal-states",
        "'[0.5, -0.1, 0.0]' (single rollout), "
        "'[[0.5, -0.1, 0.0], [0.2, 0.3, 0.0]]' (multiple rollouts), or "
        "'[[[x1, y1, ...], [x2, y2, ...]], ...]' (multi-robot per-rollout specs).",
    )


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


def _normalize_state_specs(
    specs: Any | None,
    total_dim: int,
    num_robots: int,
    robot_dims: list[int],
    label: str,
) -> list[np.ndarray]:
    if specs is None:
        return []

    if not isinstance(specs, (list, tuple)):
        raise ValueError(f"{label} specs must evaluate to a list-like object.")

    values = list(specs)
    if len(values) == 0:
        return []

    # Case 1: one global rollout vector, e.g. [x, y, vx, vy]
    if all(isinstance(x, Real) and not isinstance(x, bool) for x in values):
        return [_as_state_vector(values, expected_dim=total_dim, context=f"{label.capitalize()}")]

    if not all(isinstance(item, (list, tuple)) for item in values):
        raise ValueError(
            f"{label} specs must be a numeric vector, a list of numeric vectors, "
            "or multi-robot per-rollout nested lists."
        )

    row_values = [list(item) for item in values]

    # Case 2: single multi-robot rollout provided as per-robot vectors,
    # e.g. [[x1, y1, ...], [x2, y2, ...]]
    if (
        num_robots > 1
        and len(row_values) == num_robots
        and all(_is_numeric_sequence(robot_spec) for robot_spec in row_values)
        and all(len(robot_spec) == robot_dims[idx] for idx, robot_spec in enumerate(row_values))
    ):
        parts = [
            np.asarray(robot_spec, dtype=float)
            for robot_spec in row_values
        ]
        return [np.concatenate(parts)]

    # Case 3: list of global rollout vectors,
    # e.g. [[...global...], [...global...]]
    if all(_is_numeric_sequence(row) for row in row_values):
        return [
            _as_state_vector(row, expected_dim=total_dim, context=f"{label.capitalize()} #{idx}")
            for idx, row in enumerate(row_values)
        ]

    # Case 4: list of multi-robot rollout specs,
    # e.g. [ [[robot1...], [robot2...]], [[robot1...], [robot2...]] ]
    normalized: list[np.ndarray] = []
    for rollout_idx, rollout_spec in enumerate(row_values):
        if not isinstance(rollout_spec, list) or len(rollout_spec) != num_robots:
            raise ValueError(
                f"Each multi-robot rollout {label} must provide one vector per robot. "
                f"Rollout #{rollout_idx} has {len(rollout_spec)} entries, expected {num_robots}."
            )

        robot_parts: list[np.ndarray] = []
        for robot_idx, robot_spec in enumerate(rollout_spec):
            if not _is_numeric_sequence(robot_spec):
                raise ValueError(
                    f"Rollout #{rollout_idx}, robot #{robot_idx} {label} must be a numeric list."
                )

            expected_dim = robot_dims[robot_idx]
            if len(robot_spec) != expected_dim:
                raise ValueError(
                    f"Rollout #{rollout_idx}, robot #{robot_idx} {label} length must be {expected_dim}, "
                    f"got {len(robot_spec)}."
                )
            robot_parts.append(np.asarray(robot_spec, dtype=float))

        normalized.append(np.concatenate(robot_parts))

    return normalized


def normalize_initial_state_specs(
    simulator: DynamicsProtocol,
    initial_states: Any | None,
) -> list[np.ndarray]:
    robot_dims = [
        int(state_slice.stop - state_slice.start)
        for state_slice in simulator.robot_state_slices
    ]
    return _normalize_state_specs(
        initial_states,
        total_dim=int(simulator.nx),
        num_robots=int(simulator.num_robots),
        robot_dims=robot_dims,
        label="initial state",
    )


def normalize_goal_state_specs(
    simulator: DynamicsProtocol,
    goal_states: Any | None,
) -> list[np.ndarray]:
    robot_dims = [int(sim.goal_dim) for sim in simulator.simulators]
    return _normalize_state_specs(
        goal_states,
        total_dim=sum(robot_dims),
        num_robots=int(simulator.num_robots),
        robot_dims=robot_dims,
        label="goal state",
    )
