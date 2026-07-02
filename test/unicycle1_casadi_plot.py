import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.unicycle1 import Unicycle1
from planning.casadi_planner import CasadiPlanner


def main():
    parser = argparse.ArgumentParser(description="Plot CasADi optimal paths")
    parser.add_argument("--num_traj", type=int, default=15)
    parser.add_argument("--goal", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Specific goal (x, y, theta). Defaults to [0.0, 0.0, 0.0].")
    args = parser.parse_args()

    config = {
        "dt": 0.05,
        "max_v": 2.0,
        "goal": args.goal,
        "randomize_goal": False,
        "horizon": 30,
        "mode": "mpc",
        "Q_diag": [10.0, 10.0, 5.0],
    }

    simulator = Unicycle1(config)
    planner = CasadiPlanner(simulator, config)

    num_steps = 200

    plt.figure(figsize=(8, 8))

    gx, gy, gtheta = config["goal"]
    plt.scatter(gx, gy, color="red", marker="*", s=300, label="Goal", zorder=5)
    plt.quiver(gx, gy, np.cos(gtheta), np.sin(gtheta),
               color="red", scale=8, width=0.005, zorder=5)

    print(f"\nSimulating {args.num_traj} randomized trajectories...")

    for _ in range(args.num_traj):
        state = simulator.reset_random()
        planner.reset()

        trajectory_x = []
        trajectory_y = []
        trajectory_theta = []

        for _ in range(num_steps):
            trajectory_x.append(state[0])
            trajectory_y.append(state[1])
            trajectory_theta.append(state[2])

            if simulator.is_done(state):
                break

            obs = simulator.observe(state)
            action = planner(obs)
            state = simulator.step(state, action)

        line, = plt.plot(trajectory_x, trajectory_y, alpha=0.6, linewidth=2)

        plt.scatter(trajectory_x[0], trajectory_y[0], color="black", s=20,
                    zorder=4)
        plt.quiver(trajectory_x[0], trajectory_y[0],
                   np.cos(trajectory_theta[0]), np.sin(trajectory_theta[0]),
                   color=line.get_color(), scale=12, width=0.004, zorder=4)

    plt.title("CasADi Optimal Control Paths (Unicycle 1)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axis("equal")

    output_path = os.path.join(os.path.dirname(__file__),
                               "unicycle1_casadi_paths.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
