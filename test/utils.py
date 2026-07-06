import os
import numpy as np
import matplotlib.pyplot as plt

from data.data_collection import DataCollector
from planning.casadi_planner import CasadiPlanner

def collect_casadi_expert_data(
    simulator_class,
    config,
    repo_id,
    local_dir,
    num_traj,
    num_steps,
):
    """
    generates and saves expert trajectories for a dynamics system

    creates a simulator, CasADi planner and data collector, 
    then stores generated expert trajectories as a local LeRobot dataset

    args:
        simulator_class: dynamics simulator class to instantiate (e.g. Unicycle2)
        config: configuration dictionary for simulator and planner
        repo_id: identifier stored in LeRobot dataset metadata
        local_dir: local directory where the generated dataset is saved
        num_traj: number of expert trajectories to collect
        num_steps: maximum number of simulation steps per trajectory

    returns:
        result of DataCollector.collect_trajectories()
    """
    
    simulator = simulator_class(config)
    planner = CasadiPlanner(simulator, config)

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
    path_label="trajectory",
    show_heading=False,
    marker=None,
):
    """
    plots and saves simulated trajectories

    Args:
        simulator: system instance containing goal state in ``simulator.goal``.
        trajectories: iterable of state trajectories (each trajectories must have
            shape ``(num_steps, state_dim)`` and use x and y as its first two state entries)
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

    for trajectory_index, trajectory in enumerate(trajectories):
        line_kwargs = {
            "alpha": 0.6,
            "linewidth": 2,
        }

        if marker is not None:
            line_kwargs["marker"] = marker
            line_kwargs["markersize"] = 3

        line, = ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            label=path_label if trajectory_index == 0 else None,
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