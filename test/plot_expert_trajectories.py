import os
import sys
import argparse
from typing import Any, Mapping

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config, validate_system_config
from core.factory import DynamicsFactory, SE2_SYSTEMS, PlannerFactory
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol


def rollout_trajectory(
    simulator: DynamicsProtocol,
    planner: PlannerProtocol,
    num_steps: int,
) -> tuple[np.ndarray, bool]:
    """
    simulates one trajectory controlled by a planner

    args:
        simulator: dynamics simulator used to generate states and observations
        planner: motion planner that selects an action from each observation
        num_steps: maximum number of simulation steps.

    returns:
        trajectory: array of visited states
        reached_goal: whether simulator reached its goal condition
    """

    state = simulator.reset_random()
    planner.reset()

    trajectory = [state.copy()]
    reached_goal = False

    for _ in range(num_steps):
        if simulator.is_done(state):
            reached_goal = True
            break

        observation = simulator.observe(state)
        action = planner(observation)

        state = simulator.step(state, action)
        trajectory.append(state.copy())

    if simulator.is_done(state):
        reached_goal = True

    return np.asarray(trajectory), reached_goal


def run_plotting(
    system: str,
    planner_name: str,
    config: Mapping[str, Any],
    seeds: list[int],
    num_steps: int,
    output_path: str | None = None,
) -> str:
    validated_config = validate_system_config(system_name=system, raw_config=config)

    simulator = DynamicsFactory.create(system_name=system, config=validated_config)
    planner = PlannerFactory.create(
        planner_name=planner_name,
        simulator=simulator,
        config=validated_config,
    )

    output_path = output_path or os.path.join(
        os.path.dirname(__file__),
        f"{system}_{planner_name}_paths.pdf",
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    goal_x, goal_y = simulator.goal[:2]

    ax.scatter(
        goal_x,
        goal_y,
        marker="*",
        s=300,
        label="Goal",
        zorder=5,
    )

    show_heading = system in SE2_SYSTEMS
    if show_heading:
        goal_theta = simulator.goal[2]

        ax.quiver(
            goal_x,
            goal_y,
            0.25 * np.cos(goal_theta),
            0.25 * np.sin(goal_theta),
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.005,
            zorder=5,
        )

    num_traj = len(seeds)
    print(f"simulating {num_traj} randomized trajectories...")

    goals_reached = 0
    for seed in seeds:
        np.random.seed(seed)
        trajectory, reached_goal = rollout_trajectory(
            simulator=simulator,
            planner=planner,
            num_steps=num_steps,
        )

        if reached_goal:
            goals_reached += 1

        line, = ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            alpha=0.6,
            linewidth=2,
        )

        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            color="black",
            s=20,
            zorder=4,
        )

        if show_heading:
            start_theta = trajectory[0, 2]

            ax.quiver(
                trajectory[0, 0],
                trajectory[0, 1],
                0.25 * np.cos(start_theta),
                0.25 * np.sin(start_theta),
                color=line.get_color(),
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.004,
                zorder=4,
            )

    system_title = system.replace("_", " ").title()

    ax.set_title(
        f"{planner_name.replace('_', ' ').title()} optimal control paths ({system_title})"
    )
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.axis("equal")

    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"goal reached in {goals_reached}/{num_traj} trajectories")
    print(f"plot saved to {output_path}")
    return output_path


def main():
    """
    plots planner-controlled trajectories for a selected dynamics system
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        type=str.lower,
        choices=DynamicsFactory.names(),
        help="name of system class in lower case, e.g. single_integrator, unicycle2, ...",
    )
    parser.add_argument(
        "--planner",
        type=str.lower,
        choices=PlannerFactory.names(),
        help="name of planner class in lower case, e.g. casadi",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to yaml config file for experiment",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 13, 11, 40, 99, 100, 777, 2026, 1],
        help="list of seeds used to sample randomized trajectories",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=150,
        help="maximum number of simulation steps per trajectory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="path to generated PDF plot",
    )

    args = parser.parse_args()
    config = load_and_validate_system_config(system_name=args.system, config_path=args.config)

    run_plotting(
        system=args.system,
        planner_name=args.planner,
        config=config,
        seeds=args.seeds,
        num_steps=args.num_steps,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
