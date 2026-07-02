import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collection import DataCollector
from planning.casadi_planner import CasadiPlanner
from systems.single_integrator import SingleIntegrator


def main():
    parser = argparse.ArgumentParser(description="Generate Expert Dataset")
    parser.add_argument("--num_traj", type=int, default=100)
    parser.add_argument("--repo_id", type=str,
                        default="local/single_integrator_casadi_expert")
    parser.add_argument("--local_dir", type=str,
                        default="data/lerobot_dataset_single_integrator_casadi")
    parser.add_argument("--goal", type=float, nargs=2, default=None,
                        help="Specific goal coordinate. If not set, goals are randomized.")
    args = parser.parse_args()

    config = {"dt": 0.05, "max_vel": 2.0, "horizon": 40, "mode": "mpc",
              "Q_diag": [10.0, 10.0], "R_weight": 0.1}

    if args.goal is not None:
        config["goal"] = args.goal
        config["randomize_goal"] = False
    else:
        config["randomize_goal"] = True

    simulator = SingleIntegrator(config)
    planner = CasadiPlanner(simulator, config)

    collector = DataCollector(
        simulator,
        repo_id=args.repo_id,
        local_dir=args.local_dir
    )

    collector.collect_trajectories(planner, num_trajectories=args.num_traj,
                                   num_steps=100)


if __name__ == "__main__":
    main()
