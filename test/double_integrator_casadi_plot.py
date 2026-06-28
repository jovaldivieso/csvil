import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.single_robot_casadi import SingleRobotCasadiPlanner
from systems.double_integrator import DoubleIntegrator


def main():
    config = {"dt": 0.05, "max_accel": 2.0, "horizon": 20, "goal": [1.0, 1.0]}

    simulator = DoubleIntegrator(config)
    planner = SingleRobotCasadiPlanner(simulator, config)

    num_trajectories = 15
    num_steps = 150

    plt.figure(figsize=(8, 8))
    plt.scatter(*config["goal"], color="red", marker="*", s=300, label="Goal",
                zorder=5)
    print(f"Simulating {num_trajectories} trajectories for plotting...")

    for i in range(num_trajectories):
        initial_state = np.random.randn(4) * 2.0
        state = simulator.reset(initial_state)

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
