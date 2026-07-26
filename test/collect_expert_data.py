import os
import sys
import yaml
import argparse

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from data.data_collection import DataCollector

from systems.multi_robot import MultiRobotSimulator
from systems.double_integrator import DoubleIntegrator
from systems.single_integrator import SingleIntegrator
from systems.unicycle1 import Unicycle1
from systems.unicycle2 import Unicycle2

from planning.casadi_planner import CasadiPlanner
from planning.dblacam_planner import DbLacamPlanner


SIMULATORS = {
    "single_integrator": SingleIntegrator,
    "double_integrator": DoubleIntegrator,
    "unicycle1": Unicycle1,
    "unicycle2": Unicycle2,
}

PLANNERS = {
    "casadi": CasadiPlanner,
    "dblacam": DbLacamPlanner,
}

def create_multi_robot_simulator(config):
    """
    creates multi-robot simulator from config
    """

    robots = []
    for robot in config["robots"]:
        system = robot["system"]

        if system not in SIMULATORS:
            raise ValueError(f"unsupported system: {system}")

        robot_config = robot.get("config", {}).copy()

        # adds fixed goal to robot config:
        robot_config["goal"] = robot["goal"]
        robot_config["randomize_goal"] = False

        robot_class = SIMULATORS[system]
        robots.append(robot_class(robot_config))

    environment = config.get("db_lacam", {},).get("environment", {})

    return MultiRobotSimulator(
        robots=robots,
        environment_min=environment.get("min", [-6.0, -6.0]),
        environment_max=environment.get("max", [6.0, 6.0]),
    )


def main():
    """
    generates a lerobot dataset of expert trajectories
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        type=str.lower,
        choices=SIMULATORS.keys(),
        help="name of system class for single robot setup",
    )
    parser.add_argument(
        "--planner",
        required=True,
        type=str.lower,
        choices=PLANNERS.keys(),
        help="name of planner class, 'casadi' or 'dblacam'",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="path to yaml config for experiment",
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
        default=100,
        help="number of expert trajectories to generate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=150,
        help="maximum number of simulation steps per trajectory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="identifier stored in LeRobot dataset metadata",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        help="local directory where dataset is saved",
    )

    args = parser.parse_args()
    
    if args.planner == "dblacam" and args.algorithm_config is None:
        parser.error( "--algorithm-config is required when using db-lacam")

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    # creates multi robot simulator:
    if "robots" in cfg:
        if args.planner != "dblacam":
            parser.error("multi robot setup currently requires db-lacam")

        simulator = create_multi_robot_simulator(cfg)
        system_name = "multi_robot"
        
    # creates single robot simulator:
    else:
        if args.system is None:
            parser.error("--system is required for single-robot setup")

        simulator = SIMULATORS[args.system](cfg)
        system_name = args.system

    repo_id = args.repo_id or f"local/{args.planner}_{system_name}"
    local_dir = args.local_dir or f"data/lerobot_dataset_{args.planner}_{system_name}"

    # creates planner:
    if args.planner == "casadi":
        planner = CasadiPlanner(simulator, cfg)

    else:
        with open(args.algorithm_config, "r", encoding="utf-8") as file:
            algorithm_config = yaml.safe_load(file)
        planner = DbLacamPlanner(simulator, cfg, algorithm_config)
        
    # creates data collector:
    collector = DataCollector(
        simulator=simulator,
        repo_id=repo_id,
        local_dir=local_dir,
    )

    return collector.collect_trajectories(
        motion_planner=planner,
        num_trajectories=args.num_traj,
        num_steps=args.num_steps,
    )

if __name__ == "__main__":
    main()