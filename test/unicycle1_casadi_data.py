import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collection import DataCollector
from planning.single_robot_casadi import SingleRobotCasadiPlanner
from systems.unicycle1 import Unicycle1


def main():
    parser = argparse.ArgumentParser(description="Generate Expert Dataset \
    using CasADi")
    parser.add_argument("--num_traj", type=int, default=100,
                        help="Number of expert trajectories to collect")
    parser.add_argument("--repo_id", type=str,
                        default="local/unicycle1_casadi_expert")
    parser.add_argument("--local_dir", type=str,
                        default="data/lerobot_dataset_unicycle1_casadi"
                        )
    args = parser.parse_args()

    # Pass the argparse goal into the physics and planner configurations
    config = {"dt": 0.05, "max_v": 2.0, "horizon": 20,
              "Q_diag": [10.0, 10.0, 1.0], "R_weight": 0.1}

    simulator = Unicycle1(config)
    planner = SingleRobotCasadiPlanner(simulator, config)

    collector = DataCollector(
        simulator,
        repo_id=args.repo_id,
        local_dir=args.local_dir
    )

    collector.collect_trajectories(planner, num_trajectories=args.num_traj,
                                   num_steps=100)


if __name__ == "__main__":
    main()
