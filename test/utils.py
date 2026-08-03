import os
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors as mcolors

from core.factory import DynamicsFactory, PlannerFactory
from data.data_collection import DataCollector


def _blend_robot_family_color(
    family_color: str,
    robot_color: Any,
    robot_idx: int,
    robot_count: int,
) -> tuple[float, float, float]:
    """Blend rollout-family and robot colors to keep both distinctions visible."""
    base_rgb = np.asarray(mcolors.to_rgb(family_color), dtype=float)
    robot_rgb = np.asarray(mcolors.to_rgb(robot_color), dtype=float)

    # Increase robot influence as the robot index grows to create distinct shades.
    if robot_count <= 1:
        robot_weight = 0.25
    else:
        robot_weight = 0.2 + 0.4 * (robot_idx / max(robot_count - 1, 1))

    blended = (1.0 - robot_weight) * base_rgb + robot_weight * robot_rgb
    return tuple(np.clip(blended, 0.0, 1.0))

def collect_expert_data(
    simulator_name: str,
    planner_name: str,
    config: Mapping[str, Any],
    repo_id: str,
    local_dir: str,
    num_traj: int,
    num_steps: int,
    action_noise_std: float = 0.0,
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
        action_noise_std: std-dev of Gaussian action noise applied during execution

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
        action_noise_std=action_noise_std,
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
    trajectory_line_styles: Sequence[str] | None = None,
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

    trajectories = [np.asarray(trajectory) for trajectory in trajectories]

    fig, ax = plt.subplots(figsize=(8, 8))

    robot_state_slices = simulator.robot_state_slices
    robot_simulators = simulator.simulators
    robot_count = len(robot_state_slices)
    robot_color_map = plt.get_cmap("tab10", max(robot_count, 1))

    def has_heading_goal(sub_simulator: Any) -> bool:
        goal = getattr(sub_simulator, "goal", None)
        if goal is None:
            return False
        goal_arr = np.asarray(goal, dtype=float)
        return goal_arr.ndim == 1 and goal_arr.shape[0] >= 3

    for robot_idx, sub_sim in enumerate(robot_simulators):
        robot_color = robot_color_map(robot_idx)
        goal_x, goal_y = sub_sim.goal_state[:2]
        goal_label = "goal" if robot_count == 1 else f"goal r{robot_idx}"
        ax.scatter(
            goal_x,
            goal_y,
            marker="*",
            s=220 if robot_count > 1 else 300,
            color=robot_color,
            label=goal_label,
            zorder=5,
        )

        if show_heading and has_heading_goal(sub_sim):
            goal_theta = np.asarray(sub_sim.goal, dtype=float)[2]
            ax.quiver(
                goal_x,
                goal_y,
                0.25 * np.cos(goal_theta),
                0.25 * np.sin(goal_theta),
                color=robot_color,
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

        if trajectory_line_styles is not None and trajectory_index < len(trajectory_line_styles):
            line_kwargs["linestyle"] = trajectory_line_styles[trajectory_index]

        # Assign the label if one exists for this index, otherwise None
        current_label = None
        if trajectory_index < len(path_labels):
            current_label = path_labels[trajectory_index]

        # For fleet comparisons, per-trajectory colors separate rollout families.
        base_color = None
        if trajectory_colors is not None and trajectory_index < len(trajectory_colors):
            base_color = trajectory_colors[trajectory_index]
        elif robot_count == 1:
            base_color = f"C{trajectory_index % 10}"

        for robot_idx, state_slice in enumerate(robot_state_slices):
            robot_traj = trajectory[:, state_slice]
            if base_color is not None:
                if robot_count > 1:
                    robot_color = _blend_robot_family_color(
                        family_color=base_color,
                        robot_color=robot_color_map(robot_idx),
                        robot_idx=robot_idx,
                        robot_count=robot_count,
                    )
                else:
                    robot_color = base_color
            else:
                robot_color = robot_color_map(robot_idx)

            robot_label = None
            if current_label is not None:
                robot_label = current_label if robot_count == 1 else f"{current_label} r{robot_idx}"
            elif trajectory_index == 0:
                robot_label = "trajectory" if robot_count == 1 else f"robot {robot_idx}"

            robot_line_kwargs = dict(line_kwargs)
            robot_line_kwargs["color"] = robot_color

            if marker is not None:
                marker_options = ["o", "s", "^", "D", "v", "P", "X"]
                robot_line_kwargs["marker"] = marker_options[robot_idx % len(marker_options)]

            line, = ax.plot(
                robot_traj[:, 0],
                robot_traj[:, 1],
                label=robot_label,
                **robot_line_kwargs,
            )

            ax.scatter(
                robot_traj[0, 0],
                robot_traj[0, 1],
                color=line.get_color(),
                s=20,
                zorder=4,
            )

            if show_heading and has_heading_goal(robot_simulators[robot_idx]):
                start_theta = robot_traj[0, 2]
                ax.quiver(
                    robot_traj[0, 0],
                    robot_traj[0, 1],
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


def save_xy_rollout_video(
    simulator,
    trajectories,
    path_to_output,
    title,
    show_heading=False,
    fps: int = 12,
    dpi: int = 180,
    bitrate: int = 5000,
    path_labels=None,
    trajectory_colors: Sequence[str] | None = None,
    trajectory_line_styles: Sequence[str] | None = None,
) -> str | None:
    """Save an MP4 rollout animation with the same geometry as the PDF plot.

    Returns the video path when export succeeds, otherwise None.
    """
    if len(trajectories) == 0:
        return None

    if not animation.writers.is_available("ffmpeg"):
        print("Skipping MP4 export: ffmpeg writer is not available in matplotlib.")
        return None

    output_dir = os.path.dirname(path_to_output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.splitext(path_to_output)[0] + ".mp4"

    fig, ax = plt.subplots(figsize=(8, 8))
    robot_state_slices = simulator.robot_state_slices
    robot_simulators = simulator.simulators
    robot_count = len(robot_state_slices)
    robot_color_map = plt.get_cmap("tab10", max(robot_count, 1))

    def has_heading_goal(sub_simulator: Any) -> bool:
        goal = getattr(sub_simulator, "goal", None)
        if goal is None:
            return False
        goal_arr = np.asarray(goal, dtype=float)
        return goal_arr.ndim == 1 and goal_arr.shape[0] >= 3

    if path_labels is None:
        path_labels = ["trajectory"] + [None] * (len(trajectories) - 1)
    elif isinstance(path_labels, str):
        path_labels = [path_labels] + [None] * (len(trajectories) - 1)

    series: list[dict[str, Any]] = []

    for robot_idx, sub_sim in enumerate(robot_simulators):
        robot_color = robot_color_map(robot_idx)
        goal_x, goal_y = sub_sim.goal_state[:2]
        ax.scatter(goal_x, goal_y, marker="*", s=180 if robot_count > 1 else 220, color=robot_color, zorder=5)

        if show_heading and has_heading_goal(sub_sim):
            goal_theta = np.asarray(sub_sim.goal, dtype=float)[2]
            ax.quiver(
                goal_x,
                goal_y,
                0.25 * np.cos(goal_theta),
                0.25 * np.sin(goal_theta),
                color=robot_color,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
                zorder=5,
            )

    for trajectory_index, trajectory in enumerate(trajectories):
        current_label = None
        if trajectory_index < len(path_labels):
            current_label = path_labels[trajectory_index]

        base_color = None
        if trajectory_colors is not None and trajectory_index < len(trajectory_colors):
            base_color = trajectory_colors[trajectory_index]
        elif robot_count == 1:
            base_color = f"C{trajectory_index % 10}"

        for robot_idx, state_slice in enumerate(robot_state_slices):
            robot_traj = trajectory[:, state_slice]
            if base_color is not None:
                if robot_count > 1:
                    robot_color = _blend_robot_family_color(
                        family_color=base_color,
                        robot_color=robot_color_map(robot_idx),
                        robot_idx=robot_idx,
                        robot_count=robot_count,
                    )
                else:
                    robot_color = base_color
            else:
                robot_color = robot_color_map(robot_idx)

            marker_options = ["o", "s", "^", "D", "v", "P", "X"]
            marker = marker_options[robot_idx % len(marker_options)]

            label = None
            if current_label is not None:
                label = current_label if robot_count == 1 else f"{current_label} r{robot_idx}"
            elif trajectory_index == 0:
                label = "trajectory" if robot_count == 1 else f"robot {robot_idx}"

            series.append(
                {
                    "x": robot_traj[:, 0],
                    "y": robot_traj[:, 1],
                    "color": robot_color,
                    "marker": marker,
                    "label": label,
                    "linestyle": (
                        trajectory_line_styles[trajectory_index]
                        if trajectory_line_styles is not None
                        and trajectory_index < len(trajectory_line_styles)
                        else "-"
                    ),
                }
            )

    line_artists = []
    point_artists = []
    for item in series:
        line, = ax.plot(
            [],
            [],
            color=item["color"],
            linewidth=1.5,
            alpha=0.85,
            marker=item["marker"],
            markersize=2.5,
            linestyle=item.get("linestyle", "-"),
            label=item["label"],
        )
        point, = ax.plot([], [], marker="o", color=item["color"], markersize=3)
        line_artists.append(line)
        point_artists.append(point)

    all_x = np.concatenate([item["x"] for item in series])
    all_y = np.concatenate([item["y"] for item in series])
    x_pad = max(0.5, 0.1 * (np.max(all_x) - np.min(all_x) + 1e-9))
    y_pad = max(0.5, 0.1 * (np.max(all_y) - np.min(all_y) + 1e-9))

    ax.set_xlim(np.min(all_x) - x_pad, np.max(all_x) + x_pad)
    ax.set_ylim(np.min(all_y) - y_pad, np.max(all_y) + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")

    total_frames = max(len(item["x"]) for item in series)

    def _init():
        for line, point in zip(line_artists, point_artists):
            line.set_data([], [])
            point.set_data([], [])
        return [*line_artists, *point_artists]

    def _update(frame_idx: int):
        for item, line, point in zip(series, line_artists, point_artists):
            end = min(frame_idx + 1, len(item["x"]))
            x_data = item["x"][:end]
            y_data = item["y"][:end]
            line.set_data(x_data, y_data)
            point.set_data([x_data[-1]], [y_data[-1]])
        return [*line_artists, *point_artists]

    anim = animation.FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=total_frames,
        interval=max(1, int(1000 / max(1, fps))),
        blit=True,
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        bitrate=bitrate,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(video_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return video_path