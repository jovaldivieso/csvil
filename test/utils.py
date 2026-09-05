import os
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, FancyArrowPatch



def _heading_sample_stride(num_points: int) -> int:
    """Choose a sparse, readable stride for trajectory heading arrows."""
    return max(1, num_points // 20)


def _unicycle_collision_radius(simulator: Any, collision_distance: float | None = None) -> float | None:
    """Return the plotted point-robot radius for unicycle fleet members."""
    if type(simulator).__name__ not in {"Unicycle1", "Unicycle2"}:
        return None
    if collision_distance is None:
        collision_distance = getattr(simulator, "d_collision", None)
    if collision_distance is None:
        collision_distance = getattr(simulator, "config", {}).get("d_collision", 0.0)
    radius = float(collision_distance) / 2.0
    return radius if radius > 0.0 else None

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
    goal_states: Sequence[np.ndarray] | None = None,
):
    """
    plots and saves simulated trajectories

    Args:
        simulator: system instance; only used for fleet geometry (``robot_state_slices``,
            ``simulators``) unless ``goal_states`` is omitted, in which case its current
            (single, shared) goal is used as a fallback.
        trajectories: iterable of state trajectories (each trajectories must have
            shape ``(num_steps, state_dim)`` and use x and y as its first two state entries)
        path_labels: a string or list of strings to label the trajectories in the legend
        show_heading: whether to draw heading arrows for initial and goal states
        (requires orientation as third state entry)
        goal_states: one full ``nx``-dim goal-state vector per trajectory (e.g. each
            rollout's ``simulator.goal_state`` at the time it ran), so trajectories run
            against different goals are each plotted against their own goal instead of
            whatever goal the shared ``simulator`` happens to hold when this is called.
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

    if goal_states is not None:
        if len(goal_states) != len(trajectories):
            raise ValueError(
                f"'goal_states' length ({len(goal_states)}) must match 'trajectories' length ({len(trajectories)})."
            )
        goal_state_array = np.asarray(goal_states, dtype=float)
    else:
        goal_state_array = None

    def has_non_euclidean_state(sub_simulator: Any) -> bool:
        return not bool(getattr(sub_simulator, "is_euclidean", True))

    for robot_idx, sub_sim in enumerate(robot_simulators):
        robot_color = robot_color_map(robot_idx)
        state_slice = robot_state_slices[robot_idx]
        if goal_state_array is not None:
            robot_goal_states = np.unique(goal_state_array[:, state_slice], axis=0)
        else:
            robot_goal_states = np.asarray(sub_sim.goal_state, dtype=float)[None, :]

        goal_label = "goal" if robot_count == 1 else f"goal r{robot_idx}"
        for goal_idx, robot_goal_state in enumerate(robot_goal_states):
            goal_x, goal_y = robot_goal_state[:2]
            ax.scatter(
                goal_x,
                goal_y,
                marker="*",
                s=220 if robot_count > 1 else 300,
                color=robot_color,
                label=goal_label if goal_idx == 0 else None,
                zorder=5,
            )

            if show_heading and has_non_euclidean_state(sub_sim):
                goal_theta = float(robot_goal_state[2])
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

        # Fleet colors identify robot IDs; line styles distinguish rollout families.
        base_color = None
        if trajectory_colors is not None and trajectory_index < len(trajectory_colors):
            base_color = trajectory_colors[trajectory_index]
        elif robot_count == 1:
            base_color = f"C{trajectory_index % 10}"

        for robot_idx, state_slice in enumerate(robot_state_slices):
            robot_traj = trajectory[:, state_slice]
            if base_color is not None:
                robot_color = robot_color_map(robot_idx) if robot_count > 1 else base_color
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

            footprint_radius = _unicycle_collision_radius(
                robot_simulators[robot_idx],
                getattr(simulator, "d_collision", None),
            )
            if footprint_radius is not None:
                # Sparsely sample footprints (matching the heading-arrow stride)
                # and batch them into a single PatchCollection -- a Circle per
                # timestep of every trajectory would add tens of thousands of
                # individual artists to one figure on a long evaluation run.
                stride = _heading_sample_stride(len(robot_traj))
                footprint_idx = np.arange(0, len(robot_traj), stride, dtype=int)
                if footprint_idx[-1] != len(robot_traj) - 1:
                    footprint_idx = np.append(footprint_idx, len(robot_traj) - 1)
                ax.add_collection(
                    PatchCollection(
                        [Circle((x, y), footprint_radius) for x, y in robot_traj[footprint_idx, :2]],
                        facecolor=line.get_color(),
                        edgecolor=line.get_color(),
                        alpha=0.06,
                        linewidth=0.5,
                        zorder=1,
                    )
                )

            ax.scatter(
                robot_traj[0, 0],
                robot_traj[0, 1],
                color=line.get_color(),
                s=20,
                zorder=4,
            )

            if show_heading and has_non_euclidean_state(robot_simulators[robot_idx]):
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

                stride = _heading_sample_stride(len(robot_traj))
                heading_idx = np.arange(0, len(robot_traj), stride, dtype=int)
                if heading_idx[-1] != len(robot_traj) - 1:
                    heading_idx = np.append(heading_idx, len(robot_traj) - 1)

                sampled_states = robot_traj[heading_idx]
                sampled_theta = sampled_states[:, 2]
                ax.quiver(
                    sampled_states[:, 0],
                    sampled_states[:, 1],
                    0.18 * np.cos(sampled_theta),
                    0.18 * np.sin(sampled_theta),
                    color=line.get_color(),
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.003,
                    alpha=0.75,
                    zorder=3,
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
    phase_lengths: Sequence[int] | None = None,
    goal_states: Sequence[np.ndarray] | None = None,
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

    def has_non_euclidean_state(sub_simulator: Any) -> bool:
        return not bool(getattr(sub_simulator, "is_euclidean", True))

    if path_labels is None:
        path_labels = ["trajectory"] + [None] * (len(trajectories) - 1)
    elif isinstance(path_labels, str):
        path_labels = [path_labels] + [None] * (len(trajectories) - 1)

    series: list[dict[str, Any]] = []

    if goal_states is not None:
        if len(goal_states) != len(trajectories):
            raise ValueError(
                f"'goal_states' length ({len(goal_states)}) must match 'trajectories' length ({len(trajectories)})."
            )
        goal_state_array = np.asarray(goal_states, dtype=float)
    else:
        goal_state_array = None

    for robot_idx, sub_sim in enumerate(robot_simulators):
        robot_color = robot_color_map(robot_idx)
        state_slice = robot_state_slices[robot_idx]
        if goal_state_array is not None:
            robot_goal_states = np.unique(goal_state_array[:, state_slice], axis=0)
        else:
            robot_goal_states = np.asarray(sub_sim.goal_state, dtype=float)[None, :]

        for robot_goal_state in robot_goal_states:
            goal_x, goal_y = robot_goal_state[:2]
            ax.scatter(goal_x, goal_y, marker="*", s=180 if robot_count > 1 else 220, color=robot_color, zorder=5)

            if show_heading and has_non_euclidean_state(sub_sim):
                goal_theta = float(robot_goal_state[2])
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
                robot_color = robot_color_map(robot_idx) if robot_count > 1 else base_color
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
                    "theta": (
                        robot_traj[:, 2]
                        if show_heading and has_non_euclidean_state(robot_simulators[robot_idx])
                        else None
                    ),
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
    heading_artists = []
    footprint_artists = []
    heading_length = 0.22
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

        robot_idx = len(footprint_artists) % robot_count
        footprint_radius = _unicycle_collision_radius(
            robot_simulators[robot_idx],
            getattr(simulator, "d_collision", None),
        )
        if footprint_radius is not None:
            footprint = Circle(
                (0.0, 0.0),
                footprint_radius,
                facecolor=item["color"],
                edgecolor=item["color"],
                alpha=0.2,
                linewidth=1.0,
                zorder=2,
            )
            footprint.set_visible(False)
            ax.add_patch(footprint)
            footprint_artists.append(footprint)
        else:
            footprint_artists.append(None)

        theta_series = item.get("theta")
        if theta_series is not None:
            heading = FancyArrowPatch(
                (0.0, 0.0),
                (0.0, 0.0),
                arrowstyle="-|>",
                color=item["color"],
                linewidth=1.8,
                alpha=0.9,
                mutation_scale=12.0,
                zorder=6,
            )
            heading.set_visible(False)
            ax.add_patch(heading)
            heading_artists.append(heading)
        else:
            heading_artists.append(None)

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

    if len(series) % robot_count != 0:
        raise ValueError(
            f"Length of series ({len(series)}) must be an exact multiple of robot count ({robot_count})."
        )
    trajectory_count = len(series) // robot_count
    if phase_lengths is None:
        phase_lengths = [trajectory_count]
    phase_lengths = [int(length) for length in phase_lengths]
    if any(length <= 0 for length in phase_lengths):
        raise ValueError("'phase_lengths' entries must be positive integers.")
    if sum(phase_lengths) != trajectory_count:
        raise ValueError(
            "'phase_lengths' must contain positive group sizes summing to the number of trajectories."
        )

    series_phases: list[int] = []
    for phase_idx, phase_length in enumerate(phase_lengths):
        series_phases.extend([phase_idx] * (phase_length * robot_count))
    phase_frame_counts = []
    series_start = 0
    for phase_length in phase_lengths:
        phase_series = series[series_start : series_start + phase_length * robot_count]
        phase_frame_counts.append(max(len(item["x"]) for item in phase_series))
        series_start += phase_length * robot_count
    phase_frame_offsets = np.cumsum([0, *phase_frame_counts])
    total_frames = int(phase_frame_offsets[-1])

    def _init():
        for line, point, heading in zip(line_artists, point_artists, heading_artists):
            line.set_data([], [])
            point.set_data([], [])
            if heading is not None:
                heading.set_positions((0.0, 0.0), (0.0, 0.0))
                heading.set_visible(False)
        return [*line_artists, *point_artists, *[f for f in footprint_artists if f is not None], *[h for h in heading_artists if h is not None]]

    def _update(frame_idx: int):
        phase_idx = int(np.searchsorted(phase_frame_offsets[1:], frame_idx, side="right"))
        phase_start = int(phase_frame_offsets[phase_idx])
        phase_frame_idx = frame_idx - phase_start

        for item, item_phase, line, point, footprint, heading in zip(
            series, series_phases, line_artists, point_artists, footprint_artists, heading_artists
        ):
            if item_phase > phase_idx:
                line.set_data([], [])
                point.set_data([], [])
                if footprint is not None:
                    footprint.set_visible(False)
                if heading is not None:
                    heading.set_visible(False)
                continue

            if item_phase < phase_idx:
                continue

            end = min(phase_frame_idx + 1, len(item["x"]))
            x_data = item["x"][:end]
            y_data = item["y"][:end]
            line.set_data(x_data, y_data)
            point.set_data([x_data[-1]], [y_data[-1]])
            if footprint is not None:
                footprint.center = (x_data[-1], y_data[-1])
                footprint.set_visible(True)
            theta_series = item.get("theta")
            if heading is not None and theta_series is not None:
                theta = theta_series[end - 1]
                heading.set_positions(
                    (x_data[-1], y_data[-1]),
                    (
                        x_data[-1] + heading_length * np.cos(theta),
                        y_data[-1] + heading_length * np.sin(theta),
                    ),
                )
                heading.set_visible(True)
        return [*line_artists, *point_artists, *[f for f in footprint_artists if f is not None], *[h for h in heading_artists if h is not None]]

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