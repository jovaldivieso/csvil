import os
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
import matplotlib.pyplot as plt

from core.factory import DynamicsFactory, PlannerFactory
from data.data_collection import DataCollector

def collect_expert_data(
    simulator_name: str,
    planner_name: str,
    config: Mapping[str, Any],
    repo_id: str,
    local_dir: str,
    num_traj: int,
    num_steps: int,
):
    """
    generates and saves expert trajectories for a dynamics system

    creates a simulator, planner and data collector,
    then stores generated expert trajectories as a local LeRobot dataset

    args:
        simulator_name: dynamics simulator key (e.g. unicycle2)
        planner_name: planner key (e.g. casadi)
        config: configuration dictionary for simulator and planner
        repo_id: identifier stored in LeRobot dataset metadata
        local_dir: local directory where the generated dataset is saved
        num_traj: number of expert trajectories to collect
        num_steps: maximum number of simulation steps per trajectory

    returns:
        result of DataCollector.collect_trajectories()
    """
    
    simulator = DynamicsFactory.create(system_name=simulator_name, config=config)
    planner = PlannerFactory.create(
        planner_name=planner_name,
        simulator=simulator,
        config=config,
    )

    collector = DataCollector(
        simulator=simulator,
        repo_id=repo_id,
        local_dir=local_dir,
    )

    return collector.collect_trajectories(
        motion_planner=planner,
        num_trajectories=num_traj,
        num_steps=num_steps,
    )
 
def plot_xy_trajectories(
    simulator,
    trajectories,
    path_to_output,
    title,
    path_labels=None,
    show_heading=False,
    marker=None,
    trajectory_colors: Sequence[str] | None = None,
):
    """
    plots and saves simulated trajectories

    Args:
        simulator: system instance containing goal state in ``simulator.goal``.
        trajectories: iterable of state trajectories (each trajectories must have
            shape ``(num_steps, state_dim)`` and use x and y as its first two state entries)
        path_labels: a string or list of strings to label the trajectories in the legend
        show_heading: whether to draw heading arrows for initial and goal states 
        (requires orientation as third state entry)
    """
    
    output_dir = os.path.dirname(path_to_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    goal_x, goal_y = simulator.goal[:2]
    ax.scatter(
        goal_x,
        goal_y,
        marker="*",
        s=300,
        label="goal",
        zorder=5,
    )

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

    # Ensure path_labels is a list to handle multiple trajectory labels
    if path_labels is None:
        path_labels = ["trajectory"] + [None] * (len(trajectories) - 1)
    elif isinstance(path_labels, str):
        path_labels = [path_labels] + [None] * (len(trajectories) - 1)

    for trajectory_index, trajectory in enumerate(trajectories):
        line_kwargs = {
            "alpha": 0.6,
            "linewidth": 2,
        }

        if marker is not None:
            line_kwargs["marker"] = marker
            line_kwargs["markersize"] = 3

        # Assign the label if one exists for this index, otherwise None
        current_label = None
        if trajectory_index < len(path_labels):
            current_label = path_labels[trajectory_index]

        if trajectory_colors is not None and trajectory_index < len(trajectory_colors):
            line_kwargs["color"] = trajectory_colors[trajectory_index]

        line, = ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            label=current_label,
            **line_kwargs,
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

    ax.set_title(title)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.axis("equal")
    ax.legend()

    fig.savefig(path_to_output, format="pdf", bbox_inches="tight")
    plt.close(fig)