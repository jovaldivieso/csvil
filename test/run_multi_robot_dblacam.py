import os
import sys
import yaml
import argparse

import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config
from core.factory import DynamicsFactory, PlannerFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="test/config/multi_robot_dblacam_config.yaml",
    )
    parser.add_argument("--num-steps", type=int, default=200)
    args = parser.parse_args()

    # loads raw config to get specified initial states:
    with open(args.config, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    # loads and validates config for simulator and planner:
    config = load_and_validate_system_config(
        system_name="multi_robot",
        config_path=args.config,
    )

    simulator = DynamicsFactory.create(
        system_name="multi_robot",
        config=config,
    )

    planner = PlannerFactory.create(
        planner_name="dblacam",
        simulator=simulator,
        config=config,
    )

    # creates one concatenated initial state for robot team:
    initial_state = np.concatenate(
        [
            np.asarray(robot_config["start"], dtype=float)
            for robot_config in raw_config["robots"]
        ]
    )

    states = simulator.reset(initial_state)
    planner.reset()

    trajectories = [
        [states[state_slice].copy()]
        for state_slice in simulator.robot_state_slices
    ]

    for _ in range(args.num_steps):
        observations = simulator.observe(states)
        actions = planner(observations)
        states = simulator.step(states, actions)

        for trajectory, state_slice in zip(
            trajectories,
            simulator.robot_state_slices,
        ):
            trajectory.append(states[state_slice].copy())

        if simulator.is_done(states):
            break

    print(f"finished after {simulator.time} steps")
    print(f"all robots reached their goals: {simulator.is_done(states)}")

    for robot_index, state_slice in enumerate(
        simulator.robot_state_slices
    ):
        print(
            f"robot {robot_index} final state: "
            f"{states[state_slice]}"
        )


if __name__ == "__main__":
    main()