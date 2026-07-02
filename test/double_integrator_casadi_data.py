import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collection import DataCollector
from planning.casadi_planner import CasadiPlanner
from systems.double_integrator import DoubleIntegrator


def main():
    parser = argparse.ArgumentParser(description="Generate Expert Dataset")
    parser.add_argument("--num_traj", type=int, default=200)
    parser.add_argument("--repo_id", type=str,
                        default="local/double_integrator_casadi_expert")
    parser.add_argument(
        "--local_dir", type=str,
        default="data/lerobot_dataset_double_integrator_casadi")
    # Added argument to ensure the goal is explicitly defined
    parser.add_argument(
        "--goal", type=float, nargs=2, default=[0.0, 0.0],
        help="Specific goal coordinate. Defaults to [0.0, 0.0].")
    args = parser.parse_args()

    # Added fixed goal and explicitly turned off randomization
    config = {"dt": 0.05, "max_accel": 2.0, "horizon": 80, "mode": "mpc",
              "Q_diag": [10.0, 10.0, 1.0, 1.0], "R_weight": 0.1,
              "goal": args.goal, "randomize_goal": False}

    simulator = DoubleIntegrator(config)
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
