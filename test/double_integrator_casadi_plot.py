import sys
import os
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.casadi_planner import CasadiPlanner
from systems.double_integrator import DoubleIntegrator


def main():
    parser = argparse.ArgumentParser(description="Plot CasADi optimal paths")
    parser.add_argument("--num_traj", type=int, default=15)
    # Changed default to [0.0, 0.0] to match the training data
    parser.add_argument(
        "--goal", type=float, nargs=2, default=[0.0, 0.0],
        help="Specific goal coordinate. Defaults to [0.0, 0.0].")
    args = parser.parse_args()

    # randomize_goal is correctly set to False here
    config = {"dt": 0.05, "max_accel": 2.0, "horizon": 40, "mode": "mpc",
              "goal": args.goal, "randomize_goal": False}

    simulator = DoubleIntegrator(config)
    planner = CasadiPlanner(simulator, config)

    num_steps = 150

    plt.figure(figsize=(8, 8))
    plt.scatter(*config["goal"], color="red", marker="*", s=300, label="Goal",
                zorder=5)
    print(f"Simulating {args.num_traj} trajectories for plotting...")

    for i in range(args.num_traj):
        state = simulator.reset_random()
        planner.reset()

        x_history = [state[0]]
        y_history = [state[1]]

        for _ in range(num_steps):
            obs = simulator.observe(state)
            action = planner(obs)
            state = simulator.step(state, action)

            x_history.append(state[0])
            y_history.append(state[1])

            if simulator.is_done(state):
                break

        plt.plot(x_history, y_history, alpha=0.6, linewidth=2)
        plt.scatter(x_history[0], y_history[0], color="black", s=20, zorder=4)

    plt.title("CasADi Optimal Control Paths (Double Integrator)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axis("equal")

    output_path = os.path.join(os.path.dirname(__file__),
                               "double_integrator_casadi_paths.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
