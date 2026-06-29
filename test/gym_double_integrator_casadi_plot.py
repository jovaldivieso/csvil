import sys
import os
import matplotlib.pyplot as plt
from gymnasium.utils.env_checker import check_env

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.double_integrator import DoubleIntegrator
from systems.gym_wrapper import DynamicsGymWrapper
from planning.single_robot_casadi import SingleRobotCasadiPlanner


def main():
    config = {
        "dt": 0.05,
        "max_accel": 2.0,
        "goal": [1.0, 1.0],
        "horizon": 20
    }

    # Instantiate physics and wrap it
    simulator = DoubleIntegrator(config)
    env = DynamicsGymWrapper(simulator, max_steps=150)

    # Check the environment API (Warnings here are safe and expected)
    check_env(env.unwrapped)

    planner = SingleRobotCasadiPlanner(simulator, config)

    num_trajectories = 15
    plt.figure(figsize=(8, 8))
    plt.scatter(*config["goal"], color="red", marker="*", s=300, label="Goal",
                zorder=5)

    print(f"\nSimulating {num_trajectories} randomized trajectories...")

    for i in range(num_trajectories):
        # Using 'i' as the seed guarantees a different but reproducible start
        # state per trajectory
        obs, info = env.reset(seed=i)

        trajectory_x = []
        trajectory_y = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Record true state for plotting (bypassing the relative
            # observation)
            state = env.unwrapped.sim.state
            trajectory_x.append(state[0])
            trajectory_y.append(state[1])

            # Expert calculates action based on the gym observation
            action = planner(obs)

            # Step the gym environment
            obs, reward, terminated, truncated, info = env.step(action)

        plt.plot(trajectory_x, trajectory_y, alpha=0.6, linewidth=2)
        plt.scatter(trajectory_x[0], trajectory_y[0], color="black", s=20,
                    zorder=4)

    plt.title("Gymnasium Wrapper Test (Double Integrator + CasADi)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axis("equal")

    output_path = os.path.join(os.path.dirname(__file__),
                               "gym_double_integrator_casadi_paths.pdf")
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
