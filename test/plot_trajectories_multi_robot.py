import os
import sys
import yaml
import argparse

import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from planning.dblacam_planner import DbLacamPlanner
from collect_expert_data import create_multi_robot_simulator
from utils import plot_xy_trajectories

HEADING_SYSTEMS = {
    "unicycle1",
    "unicycle2",
}

def rollout_multi_robot_trajectories(simulator, planner, initial_states, num_steps):
    """
    simulates one joint trajectory and stores one path per robot
    """
    states = simulator.reset(initial_states)
    planner.reset()

    trajectories = [[np.asarray(state, dtype=float).copy()] for state in states]

    for _ in range(num_steps):
        if simulator.is_done(states):
            break

        observations = simulator.observe(states)
        actions = planner(observations)
        states = simulator.step(states, actions)

        for trajectory, state in zip(trajectories, states):
            trajectory.append(np.asarray(state, dtype=float).copy())

    trajectories = [np.asarray(trajectory) for trajectory in trajectories]

    reached_goals = [robot.is_done(state) for robot, state in zip(simulator.robots, states)]

    return trajectories, reached_goals


def main():
    """
    visualizes one db-lacam trajectory per robot
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="path to multi robot experiment config",
    )
    parser.add_argument(
        "--algorithm-config",
        required=True,
        help="path to db-lacam algorithm config",
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

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    with open(args.algorithm_config, "r", encoding="utf-8") as file:
        algorithm_config = yaml.safe_load(file)

    simulator = create_multi_robot_simulator(config)
    planner = DbLacamPlanner(simulator, config, algorithm_config)

    initial_states = [
        np.asarray(
            robot_config["start"],
            dtype=float,
        )
        for robot_config in config["robots"]
    ]

    trajectories, reached_goals = (
        rollout_multi_robot_trajectories(
            simulator=simulator,
            planner=planner,
            initial_states=initial_states,
            num_steps=args.num_steps,
        )
    )

    output_path = args.output_path or os.path.join(
        os.path.dirname(__file__),
        "output",
        f"dblacam_multi_robot.pdf"
    )

    labels = [
        f"robot {robot_index}: {robot_config['system'].replace('_', ' ')}, {str(reached_goal).lower()}"    
        for robot_index, (robot_config, reached_goal) in enumerate(zip(config["robots"], reached_goals))
    ]

    show_headings = [
        robot_config["system"] in HEADING_SYSTEMS
        for robot_config in config["robots"]
    ]

    plot_xy_trajectories(
        simulator=simulator,
        trajectories=trajectories,
        path_to_output=output_path,
        title="db-LaCAM multi-robot trajectories",
        goals=simulator.goal_states,
        show_headings=show_headings,
        labels=labels,
    )

    for robot_index, (trajectory, reached_goal) in enumerate(zip(trajectories, reached_goals)):
        print(f"robot {robot_index}: {len(trajectory) - 1} steps, goal reached = {reached_goal}")

    print(f"plot saved to {output_path}")

if __name__ == "__main__":
    main()
