import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Modify the Python path so it can find custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.single_robot_casadi import SingleRobotCasadiPlanner
from systems.single_integrator import SingleIntegrator


def main():
    config = {"dt": 0.05, "max_speed": 1.0, "horizon": 20, "goal": [1.0, 1.0]}

    simulator = SingleIntegrator(config)
    planner = SingleRobotCasadiPlanner(config)

    num_trajectories = 15
    num_steps = 150

    plt.figure(figsize=(8, 8))
    plt.scatter(*config["goal"], color="red", marker="*", s=300, label="Goal",
                zorder=5)
    print(f"Simulating {num_trajectories} trajectories for plotting...")

    for i in range(num_trajectories):
        initial_state = np.random.randn(4)
        state = simulator.reset(initial_state)

        # Lists to store the X and Y coordinates for this specific run
        x_history = [state[0]]
        y_history = [state[1]]

        for _ in range(num_steps):
            obs = simulator.observe(state)
            action = planner(obs)
            state = simulator.step(state, action)

            x_history.append(state[0])
            y_history.append(state[1])

            # Stop recording if the robot reaches the goal
            if simulator.is_done(state):
                break

        # Plot the trajectory line
        plt.plot(x_history, y_history, alpha=0.6, linewidth=2)
        # Mark the starting position with a small dot
        plt.scatter(x_history[0], y_history[0], color="black", s=20, zorder=4)

    # Formatting the chart
    plt.title("CasADi Optimal Control Paths (Single Integrator)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axis("equal")

    # Optional: Add a legend (handles the duplicate start dots cleanly)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Save as PDF
    output_path = os.path.join(os.path.dirname(__file__),
                               "single_integrator_paths.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")

    print(f"Plot saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
