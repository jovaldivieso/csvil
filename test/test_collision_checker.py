import unittest

import numpy as np

from systems.collision_checker import check_homogeneous_fleet_collisions


class CollisionBoundaryTests(unittest.TestCase):
    def test_distance_below_d_safe_is_collision(self) -> None:
        states = [
            np.array([0.0, 100.0, 0.0]),
            np.array([0.0, -100.0, 0.9]),
        ]

        self.assertTrue(
            check_homogeneous_fleet_collisions(
                states,
                position_indices=(0, 2),
                d_safe=1.0,
            )
        )

    def test_distance_exactly_d_safe_is_not_collision(self) -> None:
        states = [
            np.array([0.0, 100.0, 0.0]),
            np.array([0.0, -100.0, 1.0]),
        ]

        self.assertFalse(
            check_homogeneous_fleet_collisions(
                states,
                position_indices=(0, 2),
                d_safe=1.0,
            )
        )

    def test_distance_above_d_safe_is_not_collision(self) -> None:
        states = [
            np.array([0.0, 100.0, 0.0]),
            np.array([0.0, -100.0, 1.1]),
        ]

        self.assertFalse(
            check_homogeneous_fleet_collisions(
                states,
                position_indices=(0, 2),
                d_safe=1.0,
            )
        )


if __name__ == "__main__":
    unittest.main()