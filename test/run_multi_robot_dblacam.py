import os
import sys
import yaml
import argparse

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.dblacam_planner import DbLacamPlanner
from systems.multi_robot import MultiRobotSimulator
from systems.single_integrator import SingleIntegrator
from systems.unicycle1 import Unicycle1

SYSTEMS = {
    "single_integrator": SingleIntegrator,
    "unicycle1": Unicycle1,
}

def create_team(config):
    
    robots = []
    initial_states = []

    for robot_config in config["robots"]:
        system_name = robot_config["system"]

        if system_name not in SYSTEMS:
            raise ValueError(f"unsupported system: {system_name}")

        system_config = dict(robot_config.get("config", {}))

        system_config["goal"] = robot_config["goal"]
        system_config["randomize_goal"] = False

        robot = SYSTEMS[system_name](system_config)
        robots.append(robot)
        initial_states.append(np.asarray(robot_config["start"], dtype=float))

    return MultiRobotSimulator(robots), initial_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="test/config/multi_robot_dblacam_config.yaml",
    )
    parser.add_argument(
        "--algorithm-config",
        default="planning/dblacam_algorithm_default.yaml",
    )
    parser.add_argument("--num-steps", type=int, default=200)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with open(args.algorithm_config, "r", encoding="utf-8") as file:
        algorithm_config = yaml.safe_load(file)

    # creates MultiRobotSimulator:
    simulator, initial_states = create_team(config)
    
    # creates DbLacamPlanner:
    planner = DbLacamPlanner(simulator, config, algorithm_config)

    # initializes robot states:
    states = simulator.reset(initial_states)

    trajectories = [[state.copy()] for state in states]

    for _ in range(args.num_steps):
        observations = simulator.observe(states)
        actions = planner(observations)
        states = simulator.step(states, actions)

        for trajectory, state in zip(trajectories, states):
            trajectory.append(state.copy())

        if simulator.is_done(states):
            break

    print(f"finished after {simulator.time} steps")
    print(f"all robots reached their goals: {simulator.is_done(states)}")

    for robot_index, state in enumerate(states):
        print(f"robot {robot_index} final state: {state}")


if __name__ == "__main__":
    main()