import os
import sys
import yaml
import argparse

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from utils import plot_xy_trajectories

# import new systems here, then add them to SIMULATORS and HEADING_SYSTEMS:
from planning.casadi_planner import CasadiPlanner
from planning.dblacam_planner import DbLacamPlanner
from systems.double_integrator import DoubleIntegrator
from systems.single_integrator import SingleIntegrator
from systems.unicycle1 import Unicycle1
from systems.unicycle2 import Unicycle2

SIMULATORS = {
    "single_integrator": SingleIntegrator,
    "double_integrator": DoubleIntegrator,
    "unicycle1": Unicycle1,
    "unicycle2": Unicycle2,
}

HEADING_SYSTEMS = {
    "unicycle1",
    "unicycle2",
}

PLANNERS = {
    "casadi": CasadiPlanner,
    "dblacam": DbLacamPlanner,
}

DBLACAM_SIMULATORS = {
    "single_integrator",
    "unicycle1",
}

def rollout_trajectory(simulator, planner, num_steps):
    """
    simulates one trajectory controlled by a CasADi planner

    args:
        simulator: dynamics simulator used to generate states and observations
        planner: CasADi planner that selects an action from each observation
        num_steps: maximum number of simulation steps.

    returns:
        trajectory: array of visited states
        reached_goal: whether simulator reached its goal condition
    """

    state = simulator.reset_random()
    planner.reset()

    trajectory = [state.copy()]
    reached_goal = False

    for _ in range(num_steps):
        if simulator.is_done(state):
            reached_goal = True
            break

        observation = simulator.observe(state)
        action = planner(observation)

        state = simulator.step(state, action)
        trajectory.append(state.copy())

    if simulator.is_done(state):
        reached_goal = True

    return np.asarray(trajectory), reached_goal


def main():
    """
    plots CasADi controlled trajectories for a selected dynamics system
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        type=str.lower,
        required=True,
        choices=SIMULATORS.keys(),
    )
    parser.add_argument(
        "--planner",
        type=str.lower,
        required=True,
        choices=PLANNERS.keys(),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to system experiment config",
    )
    parser.add_argument(
        "--algorithm-config",
        type=str,
        default=None,
        help="path to db-lacam algorithm yaml config, required with --planner 'dblacam'"
    )
    parser.add_argument(
        "--num-traj",
        type=int,
        default=15,
        help="number of randomized trajectories to plot",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=150,
        help="maximum number of simulation steps per trajectory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="path to generated PDF plot",
    )

    args = parser.parse_args()
    
    if args.planner == "dblacam" and args.algorithm_config is None:
        parser.error( "--algorithm-config is required when using db-lacam")

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    
    simulator = SIMULATORS[args.system](config)
    
    # creates planner:
    if PLANNERS[args.planner] is CasadiPlanner:
        planner = CasadiPlanner(simulator, config)    
    else:
        if args.system not in DBLACAM_SIMULATORS:
            raise ValueError(f"db-lacam does not support {args.system}")
        config.setdefault("db_lacam", {})
        config["db_lacam"]["algorithm_config"] = args.algorithm_config
        planner = DbLacamPlanner(simulator, config)
    output_path = args.output_path or os.path.join(
        os.path.dirname(__file__),
        "output",
        f"{args.planner}_{args.system}_{args.num_steps}_paths.pdf",
    )

    show_heading = args.system in HEADING_SYSTEMS

    print(f"simulating {args.num_traj} randomized trajectories...")
    trajectories = []
    goals_reached = 0
    for _ in range(args.num_traj):
        trajectory, reached_goal = rollout_trajectory(
            simulator=simulator,
            planner=planner,
            num_steps=args.num_steps,
        )
        trajectories.append(trajectory)

        if reached_goal:
            goals_reached += 1

    system_title = args.system.replace("_", " ").title()
    planner_title = "db-LaCAM" if args.planner == "dblacam" else "CasADi"

    plot_xy_trajectories(
        simulator=simulator,
        trajectories=trajectories,
        path_to_output=output_path,
        title=f"{planner_title} optimal control paths for {system_title}",
        show_heading=show_heading,
    )

    print(f"goal reached in {goals_reached}/{args.num_traj} trajectories")
    print(f"plot saved to {output_path}")


if __name__ == "__main__":
    main()
