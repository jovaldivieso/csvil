import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collection import DataCollector
from planning.casadi_planner import CasadiPlanner
from systems.unicycle1 import Unicycle1


def main():
    parser = argparse.ArgumentParser(description="Generate Expert Dataset")
    parser.add_argument("--num_traj", type=int, default=100)
    parser.add_argument("--repo_id", type=str,
                        default="local/unicycle1_casadi_expert")
    parser.add_argument("--local_dir", type=str,
                        default="data/lerobot_dataset_unicycle1_casadi")
    # Note: Unicycle has 3 goal states (x, y, theta)
    parser.add_argument("--goal", type=float, nargs=3, default=None,
                        help="Specific goal (x, y, theta). If not set, randomized.")
    args = parser.parse_args()

    config = {"dt": 0.05, "max_v": 2.0, "horizon": 30, "mode": "mpc",
              "Q_diag": [10.0, 10.0, 5.0], "R_weight": 0.1}

    if args.goal is not None:
        config["goal"] = args.goal
        config["randomize_goal"] = False
    else:
        config["randomize_goal"] = True

    simulator = Unicycle1(config)
    planner = CasadiPlanner(simulator, config)

    collector = DataCollector(
        simulator,
        repo_id=args.repo_id,
        local_dir=args.local_dir
    )

    collector.collect_trajectories(planner, num_trajectories=args.num_traj,
                                   num_steps=150)


if __name__ == "__main__":
    main()
