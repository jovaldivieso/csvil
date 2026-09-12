"""Goal-error metrics that respect each state coordinate's geometry.

A plain L2 norm over the whole state vector is wrong twice over on a non-Euclidean
system: a heading that is correct but wrapped by 2*pi contributes 6.28 to the norm,
and position (metres) is summed with heading (radians) and velocity as if they
shared a unit. Both bite in practice -- evaluation rows have reported a *successful*
rollout next to a goal error of 6.28, because `is_done` wraps the angle and the
reported error did not.

The split here is per robot and driven by each simulator's own `position_indices`
and `angular_state_indices`, so no caller hardcodes a state layout.
"""

from __future__ import annotations

import numpy as np

from systems.dynamics import DynamicsProtocol


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray:
    """Map an angular residual into [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def per_robot_goal_errors(
    simulator: DynamicsProtocol,
    state: np.ndarray,
    goal_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-robot (position error, heading error) between a state and a goal state.

    `position_error[i]` is the Euclidean distance between robot i's position
    coordinates and its goal's; `heading_error[i]` is the largest wrapped angular
    residual over that robot's angular coordinates, and 0.0 for a simulator that
    declares none. Both arrays have one entry per robot, so callers choose whether
    the fleet summary is the mean or the worst robot.

    Works for a single-robot simulator as well as a fleet: the former reports one
    entry, since the protocol defaults `num_robots` to 1.
    """
    state_array = np.asarray(state, dtype=float)
    goal_array = np.asarray(goal_state, dtype=float)
    if state_array.shape != goal_array.shape:
        raise ValueError(
            f"State shape {state_array.shape} does not match goal shape {goal_array.shape}."
        )

    robot_simulators = list(getattr(simulator, "simulators", None) or [simulator])
    state_slices = list(
        getattr(simulator, "robot_state_slices", None) or [slice(0, int(simulator.nx))]
    )
    if len(robot_simulators) != len(state_slices):
        raise ValueError(
            f"Simulator exposes {len(robot_simulators)} sub-simulators but "
            f"{len(state_slices)} state slices."
        )

    position_errors: list[float] = []
    heading_errors: list[float] = []
    for robot_simulator, state_slice in zip(robot_simulators, state_slices):
        robot_state = state_array[state_slice]
        robot_goal = goal_array[state_slice]

        position_indices = list(robot_simulator.position_indices)
        position_errors.append(
            float(np.linalg.norm(robot_state[position_indices] - robot_goal[position_indices]))
        )

        angular_indices = list(getattr(robot_simulator, "angular_state_indices", ()))
        heading_errors.append(
            float(np.max(np.abs(wrap_to_pi(robot_state[angular_indices] - robot_goal[angular_indices]))))
            if angular_indices
            else 0.0
        )

    return np.asarray(position_errors), np.asarray(heading_errors)


def fleet_goal_errors(
    simulator: DynamicsProtocol,
    state: np.ndarray,
    goal_state: np.ndarray,
) -> tuple[float, float]:
    """Fleet summary of `per_robot_goal_errors`: the mean over robots of each error."""
    position_errors, heading_errors = per_robot_goal_errors(simulator, state, goal_state)
    return float(np.mean(position_errors)), float(np.mean(heading_errors))
