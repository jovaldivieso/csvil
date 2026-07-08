import os
import sys
import yaml
import argparse

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from utils import collect_casadi_expert_data

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


def main():
    """
    generates a local LeRobot dataset of expert trajectories
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        type=str.lower,
        choices=SIMULATORS.keys(),
        help="name of system class, e.g. single_integrator, unicycle2, ...",
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

    repo_id = args.repo_id or f"local/{args.system}_casadi_expert"
    local_dir = args.local_dir or f"data/lerobot_dataset_{args.system}_casadi"

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    collect_casadi_expert_data(
        simulator_class=SIMULATORS[args.system],
        config=cfg,
        repo_id=repo_id,
        local_dir=local_dir,
        num_traj=args.num_traj,
        num_steps=args.num_steps,
    )


if __name__ == "__main__":
    main()
