from __future__ import annotations

import numpy as np


def first_homogeneous_fleet_collision(
    states: list[np.ndarray],
    position_indices: tuple[int, ...],
    d_collision: float,
) -> tuple[int, int, float] | None:
    """Return the first colliding pair and distance, if any."""
    if d_collision <= 0.0 or len(states) < 2:
        return None

    indices = list(position_indices)
    positions = [np.asarray(state)[indices] for state in states]
    threshold_sq = d_collision * d_collision
    for first_idx, first_position in enumerate(positions):
        for second_idx in range(first_idx + 1, len(positions)):
            distance = float(np.linalg.norm(first_position - positions[second_idx]))
            if distance * distance < threshold_sq:
                return first_idx, second_idx, distance
    return None


def check_homogeneous_fleet_collisions(
    states: list[np.ndarray],
    position_indices: tuple[int, ...],
    d_safe: float,
) -> bool:
    return first_homogeneous_fleet_collision(states, position_indices, d_safe) is not None
