import os
import numpy as np
import matplotlib.pyplot as plt
    
def plot_xy_trajectories(
    simulator,
    trajectories,
    path_to_output,
    title,
    goals=None,
    show_headings=None,
    labels=None,
):
    """
    plots and saves simulated trajectories

    args:
        simulator: single- or multi-robot simulator
        trajectories: iterable containing one trajectory per plotted path
        goals: optional goal state for each trajectory
        show_headings: optional heading flag for each trajectory
        labels: optional label for each trajectory
    """
    output_dir = os.path.dirname(path_to_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    trajectories = [np.asarray(trajectory) for trajectory in trajectories]

    if goals is None:
        goals = [simulator.goal] * len(trajectories)

    if show_headings is None:
        show_headings = [False] * len(trajectories)
        
    if labels is None:
        labels = [None] * len(trajectories)

    fig, ax = plt.subplots(figsize=(8, 8))

    for trajectory, goal, label, show_heading in zip(trajectories, goals, labels, show_headings):
        line_kwargs = {"alpha": 0.6, "linewidth": 2}

        line, = ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            label=label,
            **line_kwargs,
        )

        # marks start:
        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            color=line.get_color(),
            marker="o",
            s=40,
            edgecolor="black",
            zorder=5,
        )

        # marks goal:
        ax.scatter(
            goal[0],
            goal[1],
            color=line.get_color(),
            marker="*",
            s=250,
            edgecolor="black",
            zorder=5,
        )

        if show_heading:
            start_theta = trajectory[0, 2]
            goal_theta = goal[2]

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
                zorder=5,
            )

            ax.quiver(
                goal[0],
                goal[1],
                0.25 * np.cos(goal_theta),
                0.25 * np.sin(goal_theta),
                color=line.get_color(),
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
                zorder=5,
            )

    ax.set_title(title)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.axis("equal")
    ax.legend()

    fig.savefig(path_to_output, format="pdf", bbox_inches="tight")
    plt.close(fig)