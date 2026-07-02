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
    parser.add_argument("--goal", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Specific goal (x, y, theta). Defaults to [0.0, 0.0, 0.0].")
    args = parser.parse_args()

    config = {"dt": 0.05, "max_v": 2.0, "horizon": 30, "mode": "mpc",
              "Q_diag": [10.0, 10.0, 5.0], "R_weight": 0.1,
              "goal": args.goal, "randomize_goal": False}

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
