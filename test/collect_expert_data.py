import os
import sys
import yaml
import argparse

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from data.data_collection import DataCollector

# import new systems here and then add them to SIMULATORS:
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

from planning.casadi_planner import CasadiPlanner
from planning.dblacam_planner import DbLacamPlanner

PLANNERS = {
    "casadi": CasadiPlanner,
    "dblacam": DbLacamPlanner,
}


def main():
    """
    generates a LeRobot dataset of expert trajectorie

    creates a simulator, planner and data collector, 
    then stores generated expert trajectories as a local LeRobot dataset

    args:
        system: dynamics simulator class (e.g. Unicycle2)
        planner: expert planner class (e.g. CasadiPlanner)
        config: config file for simulator and planner
        algorithm-config: db-lacam algorithm config file
        repo_id: identifier stored in LeRobot dataset metadata
        local_dir: local directory where the generated dataset is saved
        num_traj: number of expert trajectories to collect
        num_steps: maximum number of simulation steps per trajectory

    returns:
        result of DataCollector.collect_trajectories()
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        required=True,
        type=str.lower,
        choices=SIMULATORS.keys(),
        help="name of system class, e.g. 'single_integrator', 'unicycle2', ...",
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

    repo_id = args.repo_id or f"local/{args.planner}_{args.system}"
    local_dir = args.local_dir or f"data/lerobot_dataset_{args.planner}_{args.system}"

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)
      
    # creates simulator:          
    simulator = SIMULATORS[args.system](cfg)
    
    # creates planner:
    if PLANNERS[args.planner] is CasadiPlanner:
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
