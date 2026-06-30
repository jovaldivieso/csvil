import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.unicycle1 import Unicycle1
from planning.single_robot_casadi import SingleRobotCasadiPlanner


def main():
    # Updated config for Unicycle1 with a 3D goal and tuned Q_diag
    config = {
        "dt": 0.05,
        "max_v": 2.0,
        "goal": [1.0, 1.0, np.pi / 4],
        "horizon": 30,
        "Q_diag": [10.0, 10.0, 5.0],
    }

    simulator = Unicycle1(config)
    planner = SingleRobotCasadiPlanner(simulator, config)

    num_trajectories = 15
    num_steps = 200

    plt.figure(figsize=(8, 8))

    # Mark goal position and heading
    gx, gy, gtheta = config["goal"]
    plt.scatter(gx, gy, color="red", marker="*", s=300, label="Goal", zorder=5)
    plt.quiver(gx, gy, np.cos(gtheta), np.sin(gtheta),
               color="red", scale=8, width=0.005, zorder=5)

    print(f"\nSimulating {num_trajectories} randomized trajectories...")

    for _ in range(num_trajectories):
        # Generate a random initial position (x, y) around 0 and heading
        initial_pos = np.random.randn(2) * 2.0
        initial_theta = np.random.uniform(-np.pi, np.pi)
        initial_state = np.array([initial_pos[0], initial_pos[1],
                                  initial_theta])

        state = simulator.reset(initial_state)

        trajectory_x = []
        trajectory_y = []
        trajectory_theta = []

        for _ in range(num_steps):
            # Record current state
            trajectory_x.append(state[0])
            trajectory_y.append(state[1])
            trajectory_theta.append(state[2])

            if simulator.is_done(state):
                break

            # Step simulation forward
            obs = simulator.observe(state)
            action = planner(obs)
            state = simulator.step(state, action)

        # Plot the trajectory path
        line, = plt.plot(trajectory_x, trajectory_y, alpha=0.6, linewidth=2)

        # Show start position and initial heading
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
