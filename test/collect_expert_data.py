import os
import sys
import time
import yaml
import shutil
import argparse

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from utils import collect_expert_data

def main():
    """
    generates a local LeRobot dataset of expert trajectories
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        type=str.lower,
        choices=DynamicsFactory.names(),
        help="name of system class, e.g. single_integrator, unicycle2, ...",
    )
    parser.add_argument(
        "--planner",
        type=str.lower,
        choices=PlannerFactory.names(),
        help="name of planner class in lower case, e.g. casadi",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to yaml config file for experiment",
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
        "--action-noise-std",
        type=float,
        default=0.0,
        help="std-dev of Gaussian action noise applied during simulator execution",
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

    timestamp = str(int(time.time()))
    repo_id = args.repo_id or f"local/{args.system}_{args.planner}_expert"
    local_dir = args.local_dir or f"data/lerobot_dataset_{args.system}_{args.planner}_{timestamp}"
        
    config = load_and_validate_system_config(
        system_name=args.system,
        config_path=args.config,
    )
    
    collect_expert_data(
        simulator_name=args.system,
        planner_name=args.planner,
        config=config,
        repo_id=repo_id,
        local_dir=local_dir,
        num_traj=args.num_traj,
        num_steps=args.num_steps,
        action_noise_std=args.action_noise_std,
    )
    
    print(f"dataset saved to {local_dir}")
    
    # copies configuration used for this experiment:
    shutil.copy2(args.config, os.path.join(local_dir, "config.yaml"))

    algorithm_config = config.get("db_lacam", {}).get("algorithm_config")
    if algorithm_config:
        shutil.copy2(algorithm_config, os.path.join(local_dir, "algorithm_config.yaml"))        

    # save command-line params:
    with open(os.path.join(local_dir, "run.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(vars(args), file, sort_keys=False)

    
if __name__ == "__main__":
    main()
