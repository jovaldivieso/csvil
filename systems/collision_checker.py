from __future__ import annotations

import numpy as np


def check_homogeneous_fleet_collisions(
    states: list[np.ndarray],
    position_indices: tuple[int, ...],
    d_safe: float,
) -> bool:
    if d_safe <= 0.0 or len(states) < 2:
        return False

    min_dist_sq = d_safe * d_safe
    indices = list(position_indices)
    positions = [np.asarray(state)[indices] for state in states]

    for first_idx, first_position in enumerate(positions):
        for second_position in positions[first_idx + 1:]:
            distance_sq = np.sum((first_position - second_position) ** 2)
            if distance_sq < min_dist_sq:
                return True
    return False
