"""Plot evaluation scenarios -- starts, goals, robot size, sensing -- without running anything.

A quick visual check of the scenario configs before any training or evaluation uses
them. Random-goal configs are sampled exactly as test/evaluate_scaling.py samples its
episodes (evaluation_seed_specs from seed 50000, then sample_initial_state), so each
panel is one real evaluation episode. Configs with a per-robot 'start' (the circle
scenario) are drawn from those fixed starts and goals, as with --use-config-start.

Each panel shows the workspace box (the region starts and goals are drawn from; robots
may leave it), every robot as a d_collision disk at its start with a heading arrow, a
line to its goal, and robot 0's visibility circle. Titles give the density, the mean
number of neighbours visible at t=0 and the step budget: 1.5 x the longest possible
straight-line travel at max speed (workspace diagonal, or the longest start-goal
distance for fixed starts).

Usage:
    python test/plot_scenarios.py                         # the default review set
    python test/plot_scenarios.py --configs a.yaml b.yaml --episodes 6
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_and_validate_system_config  # noqa: E402
from core.factory import DynamicsFactory  # noqa: E402
from learning.dagger import evaluation_seed_specs, sample_initial_state  # noqa: E402

STUDY = "test/config/study"
DEFAULT_CONFIGS = (
    [f"{STUDY}/unicycle2_fleet_{n:02d}.yaml" for n in (2, 4, 8, 32)]
    + [f"{STUDY}/density/unicycle2_n08_d{f}.yaml" for f in ("0.25", "0.5", "1", "2", "3")]
    + [f"{STUDY}/circle/unicycle2_circle_{n:02d}.yaml" for n in (2, 8, 32)]
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/plots/scenarios"
SEED_START = 50000
STEP_BUDGET_FACTOR = 1.5


def config_starts_and_goals(raw_config: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Fixed per-robot starts and goals, or None for a random-goal config."""
    starts, goals = [], []
    for entry in raw_config["robots"]:
        start = entry.get("start", entry["config"].get("start"))
        if start is None:
            return None
        starts.append(np.asarray(start, dtype=float))
        goals.append(np.asarray(entry["config"]["goal"], dtype=float))
    return np.stack(starts), np.stack(goals)


def step_budget(config: dict, max_travel: float) -> int:
    max_speed = float(config["robots"][0]["config"]["max_linear_vel"])
    return math.ceil(STEP_BUDGET_FACTOR * max_travel / (max_speed * float(config["dt"])))


def visible_neighbours(simulator, state: np.ndarray) -> float:
    observation = simulator.observe(state, validate=False)
    return float(np.mean([
        simulator.decentralized_policy_observation(observation, robot)["observation.neighbor_mask"].sum()
        for robot in range(simulator.num_robots)
    ]))


def draw_episode(ax, starts: np.ndarray, goals: np.ndarray, config: dict, half_width: float | None) -> None:
    num_robots = len(starts)
    colors = plt.get_cmap("tab10" if num_robots <= 10 else "hsv")(
        np.linspace(0, 1, num_robots, endpoint=num_robots <= 10)
    )
    robot_radius = float(config["d_collision"]) / 2.0
    visibility = float(config["inter_robot_visibility_radius"])
    if half_width is not None:
        ax.add_patch(plt.Rectangle((-half_width, -half_width), 2 * half_width, 2 * half_width,
                                   fill=False, linestyle="--", color="0.5", linewidth=1))
    for robot, (start, goal, color) in enumerate(zip(starts, goals, colors)):
        ax.plot([start[0], goal[0]], [start[1], goal[1]], color=color, linewidth=0.8, alpha=0.6)
        ax.add_patch(plt.Circle(start[:2], robot_radius, color=color, alpha=0.8))
        ax.arrow(start[0], start[1], robot_radius * 1.6 * math.cos(start[2]),
                 robot_radius * 1.6 * math.sin(start[2]), width=0.03 * robot_radius * 2,
                 color="k", length_includes_head=True)
        ax.plot(goal[0], goal[1], marker="x", color=color, markersize=6, markeredgewidth=2)
    ax.add_patch(plt.Circle(starts[0][:2], visibility, fill=False, color=colors[0], linestyle=":"))
    points = np.vstack([starts[:, :2], goals[:, :2]])
    reach = max(np.abs(points).max() + robot_radius, half_width or 0.0) * 1.08
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def plot_config(config_path: str, episodes: int, output_dir: Path) -> Path:
    config = load_and_validate_system_config("multi_robot", config_path)
    simulator = DynamicsFactory.create(system_name="multi_robot", config=config)
    num_robots = int(simulator.num_robots)
    fixed = config_starts_and_goals(config)

    if fixed is not None:
        starts, goals = fixed
        panels = [(starts, goals, visible_neighbours(simulator, starts.reshape(-1)))]
        half_width = None
        max_travel = float(np.max(np.linalg.norm(starts[:, :2] - goals[:, :2], axis=1)))
        summary = f"fixed starts, longest travel {max_travel:.1f} m"
    else:
        half_width = float(config["robots"][0]["config"]["workspace_bounds"][1])
        panels = []
        for seed_spec in evaluation_seed_specs(simulator, episodes, SEED_START):
            state = sample_initial_state(simulator, seed_spec)
            starts = state.reshape(num_robots, -1)
            goals = np.asarray(simulator.goal_state).reshape(num_robots, -1)
            panels.append((starts, goals, visible_neighbours(simulator, state)))
        max_travel = 2.0 * math.sqrt(2.0) * half_width
        density = num_robots / (2.0 * half_width) ** 2
        summary = f"workspace ±{half_width:g} m, {density:.3f} robots/m²"

    columns = min(len(panels), 3)
    rows = math.ceil(len(panels) / columns)
    # At least two panels wide, so a single-panel figure still has room for the title.
    width = 4.2 * max(columns, 2)
    fig, axes = plt.subplots(rows, columns, figsize=(width, 4.2 * rows + 0.6), squeeze=False)
    for ax, (starts, goals, visible) in zip(axes.flat, panels):
        draw_episode(ax, starts, goals, config, half_width)
        ax.set_title(f"{visible:.2f} visible neighbours at t=0", fontsize=8)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        f"{Path(config_path).name}: N={num_robots}, {summary}\n"
        f"d_collision {config['d_collision']} m (disks), visibility {config['inter_robot_visibility_radius']} m "
        f"(dotted, robot 0), step budget {step_budget(config, max_travel)}",
        fontsize=9,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(config_path).stem}.png"
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--episodes", type=int, default=6, help="panels per random-goal config")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for config_path in args.configs:
        print(f"wrote {os.path.relpath(plot_config(config_path, args.episodes, args.output_dir), PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
