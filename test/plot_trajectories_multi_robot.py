import os
import sys
import argparse

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, TEST_DIR)

from utils import plot_xy_trajectories
from core.config import load_yaml_config, validate_system_config
from core.factory import DynamicsFactory, PlannerFactory

def rollout_multi_robot_trajectories(simulator, planner, initial_state, num_steps):
    """
    simulates one joint trajectory and stores one path per robot
    """

    state = simulator.reset(initial_state)
    planner.reset()

    trajectory = [state]

    for _ in range(num_steps):
        if simulator.is_done(state):
            break

        observation = simulator.observe(state)
        action = planner(observation)
        state = simulator.step(state, action)

        trajectory.append(state.copy())

    reached_goals = [
        robot.is_done(state[state_slice])
        for robot, state_slice in zip(simulator.simulators,simulator.robot_state_slices)
    ]

    return np.asarray(trajectory), reached_goals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="path to multi-robot db-lacam config",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="maximum number of simulation steps",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="path to generated PDF",
    )
    args = parser.parse_args()

    raw_config = load_yaml_config(args.config)

    initial_state = np.concatenate([robot_config["start"] for robot_config in raw_config["robots"]])

    config = validate_system_config(system_name="multi_robot", raw_config=raw_config)

    simulator = DynamicsFactory.create(system_name="multi_robot", config=config)
    
    planner = PlannerFactory.create(
        planner_name="dblacam",
        simulator=simulator,
        config=config,
    )

    trajectory, reached_goals = rollout_multi_robot_trajectories(simulator, planner, initial_state, args.num_steps)

    output_path = args.output_path or os.path.join(os.path.dirname(__file__), "output", "dblacam_multi_robot.pdf")

    plot_xy_trajectories(
        simulator=simulator,
        trajectories=[trajectory],
        path_to_output=output_path,
        title="db-LaCAM multi-robot trajectory",
        path_labels=["db-LaCAM"],
        show_heading=not simulator.is_euclidean,
    )

    for robot_index, reached_goal in enumerate(reached_goals):
        print(f"robot {robot_index}: goal reached = {reached_goal}")

    print(f"plot saved to {output_path}")

if __name__ == "__main__":
    main()