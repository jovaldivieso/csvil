"""Generate corner-swap ("two stacks trade places") evaluation configs for unicycle2 fleets.

Half the fleet starts as a vertical stack hanging off the top-right corner of the
workspace, the other half as the same stack rising from the bottom-left corner, and
the two stacks trade places: every robot's goal is the antipode of its own start,
which at even N is exactly the start of robot i + N/2.
Because the two corners are antipodal and the stacks are mirror images, every goal is
the negation of its own start, so every path runs through the origin -- the same
head-on property as the antipodal ring, but with the robots packed at ``d_safe``
inside each group, so a policy has to hold a column together while it crosses.

Both endpoints are written into the config: ``start`` per robot (the schema already
supports it, see ``_validate_multi_robot_start`` in core/config.py) and a fixed
``goal`` with ``randomize_goal: false``. Evaluate them with

    python test/evaluate_scaling.py --use-config-start ...

Box: the workspace is the *same* ``+-2.5*d_safe*sqrt(N/2)`` box as the random fleet of
that size (``goal_half_width`` in generate_fleet_configs.py), so the swap differs from
the training scenario of the same N in layout only, not in extent. The corners are the
literal corners of that box, so the crossing distance is its diagonal, 2*sqrt(2)*W,
and it grows with N -- pass ``--steps`` accordingly; the generator prints the budget.

Stacking is a single column at ``--spacing-per-d-safe`` * d_safe. At the default 1.0
the neighbours within a stack start exactly d_safe apart: feasible (d_collision is
smaller) but with no margin, which is the intent -- the formation is the difficulty.
A column of N/2 robots is taller than the box once N >= 24, so for N = 32 the stacks
run down the side walls and past the midline; the generator reports the span.

Usage:
    python test/config/generate_swap_configs.py
    python test/config/generate_swap_configs.py --spacing-per-d-safe 1.5 --out-subdir loose
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import validate_system_config  # noqa: E402

# Loaded by path: test/config is not a package, and the name "test" would shadow
# the standard library's own module.
_spec = importlib.util.spec_from_file_location(
    "_generate_fleet_configs", Path(__file__).resolve().parent / "generate_fleet_configs.py"
)
_fleet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fleet)
Q_BLOCK, _FlowListDumper, format_q_diag = _fleet.Q_BLOCK, _fleet._FlowListDumper, _fleet.format_q_diag
first_robot_template = _fleet.first_robot_template
goal_half_width = _fleet.goal_half_width
TASK_TOLERANCES = _fleet.TASK_TOLERANCES

# The 2-robot fleet config rather than test/config/2_multi_unicycle2_casadi_config.yaml:
# the swaps must match the scenarios the policies are actually trained and scored on,
# and the raw template has since been retuned (4 m/s instead of 1) for other work.
# Everything taken from here is fleet-size independent -- dt, d_safe, d_collision,
# visibility, horizon, cost weights and the per-robot dynamics; the swap writes its own
# starts, goals, workspace_bounds and Q_diag.
TEMPLATE_PATH = PROJECT_ROOT / "test/config/study/unicycle2_fleet_02.yaml"
OUTPUT_DIR = PROJECT_ROOT / "test/config/study/swap"
FLEET_SIZES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32)

# Vertical gap inside a stack, in units of d_safe. 1.0 puts the neighbours exactly on
# the safety radius; below 1.0 the starting formation violates it (and below
# d_collision/d_safe it starts in collision, which is a hard failure here).
DEFAULT_SPACING_PER_D_SAFE = 1.0


def robot_endpoints(num_robots: int, half_width: float, spacing: float, robot_idx: int):
    """Start state and goal for one robot: the goal is the point opposite the start.

    Robots 0..N//2-1 hang downwards from the top-right corner, robots N//2..N-1 rise
    from the bottom-left corner, each stack in fleet order from its own corner. The
    two corners are antipodal, so at even N the goal -start is the start of robot
    i +- N/2 and the two equal stacks trade places exactly.

    At odd N the bottom-left stack carries the extra robot, and that robot's goal is
    still -start: a point one gap beyond the top of the shorter column, which no robot
    is vacating. The odd sizes are therefore a slightly easier problem than their
    neighbours -- read the sweep as a curve, not as odd against even.
    """
    stack_size = num_robots // 2
    if robot_idx < stack_size:
        x, y = half_width, half_width - spacing * robot_idx
    else:
        x, y = -half_width, -half_width + spacing * (robot_idx - stack_size)
    # Face straight at the goal, which is the antipode (-x, -y).
    heading = math.atan2(-y - y, -x - x)
    start = [round(x, 6), round(y, 6), round(heading, 6), 0.0, 0.0]
    goal = [round(-x, 6), round(-y, 6), round(heading, 6)]
    return start, goal


def build_config(template: dict, num_robots: int, spacing: float) -> tuple[dict, float]:
    robot_template = first_robot_template(template)
    half_width = goal_half_width(num_robots, template["d_safe"])

    base_robot_config = {
        key: value
        for key, value in robot_template["config"].items()
        if key not in {"goal", "start"}
    }

    robots = []
    for robot_idx in range(num_robots):
        start, goal = robot_endpoints(num_robots, half_width, spacing, robot_idx)
        robot_config = copy.deepcopy(base_robot_config)
        robot_config["randomize_goal"] = False
        robot_config.update(TASK_TOLERANCES)
        robot_config["workspace_bounds"] = [-half_width, half_width]
        robot_config["goal"] = goal
        robot_config["start"] = start
        robots.append({"system": robot_template["system"], "config": robot_config})

    config = {
        key: value
        for key, value in template.items()
        if key not in {"robots", "Q_diag"}
    }
    config["Q_diag"] = Q_BLOCK * num_robots
    config["robots"] = robots
    return config, half_width


def render_config(config: dict, num_robots: int) -> str:
    scalars = {key: value for key, value in config.items() if key not in {"Q_diag", "robots"}}
    return (
        yaml.safe_dump(scalars, sort_keys=False)
        + format_q_diag(num_robots)
        + yaml.dump({"robots": config["robots"]}, Dumper=_FlowListDumper, sort_keys=False)
    )


def check_scenario(config: dict, num_robots: int, half_width: float, d_safe: float) -> dict:
    """Verify the swap is actually feasible and report what the run will need."""
    import numpy as np

    from core.factory import DynamicsFactory

    simulator = DynamicsFactory.create(system_name="multi_robot", config=config)
    # validate_system_config lifts 'start' to the robot-entry top level, so read
    # whichever spelling this config carries.
    def entry_start(entry):
        start = entry.get("start")
        if start is None:
            start = entry["config"]["start"]
        return np.asarray(start, dtype=float)

    start_state = np.concatenate([entry_start(entry) for entry in config["robots"]])

    if simulator.is_collision(start_state):
        details = simulator.collision_details(start_state)
        raise SystemExit(
            f"{num_robots}-robot swap starts in collision: robots {details['robot_i']} and "
            f"{details['robot_j']} are {details['distance']:.3f} apart, inside "
            f"d_collision={config['d_collision']}; raise --spacing-per-d-safe."
        )

    positions = np.stack([entry_start(entry)[:2] for entry in config["robots"]])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)

    # At even N every goal must coincide with some other robot's start -- the two equal
    # stacks trade places exactly. At odd N the longer stack's extra robot aims one gap
    # beyond the top of the shorter column, which no robot is vacating, so the check
    # would always fail and is not the contract there.
    if num_robots % 2 == 0:
        starts = {(round(p[0], 4), round(p[1], 4)) for p in positions}
        for robot_idx, entry in enumerate(config["robots"]):
            goal_xy = (round(entry["config"]["goal"][0], 4), round(entry["config"]["goal"][1], 4))
            if goal_xy not in starts:
                raise SystemExit(f"robots[{robot_idx}] goal {goal_xy} is not another robot's start.")

    observation = simulator.observe(simulator.reset(start_state), validate=False)
    visible = float(np.mean([
        simulator.decentralized_policy_observation(observation, i)["observation.neighbor_mask"].sum()
        for i in range(num_robots)
    ]))

    dt = float(config["dt"])
    max_linear_vel = float(config["robots"][0]["config"]["max_linear_vel"])
    # The corner robots travel the full diagonal; that is the longest trip in the fleet.
    longest_trip = 2.0 * math.hypot(half_width, half_width)
    return {
        "min_pair_distance": float(distances.min()),
        "visible_neighbours": visible,
        "longest_trip": longest_trip,
        "straight_line_steps": math.ceil(longest_trip / (max_linear_vel * dt)),
        "stack_bottom": float(positions[:num_robots // 2, 1].min()),
        "stack_span": float(np.ptp(positions[:num_robots // 2, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spacing-per-d-safe", type=float, default=DEFAULT_SPACING_PER_D_SAFE,
                        help="stack gap in units of d_safe "
                             f"(default {DEFAULT_SPACING_PER_D_SAFE}, i.e. exactly d_safe)")
    parser.add_argument("--out-subdir", default=None,
                        help="write into test/config/study/swap/<subdir> instead of swap/")
    args = parser.parse_args()

    template = yaml.safe_load(TEMPLATE_PATH.read_text())
    d_safe = float(template["d_safe"])
    spacing = round(args.spacing_per_d_safe * d_safe, 6)
    output_dir = OUTPUT_DIR / args.out_subdir if args.out_subdir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"template d_safe={d_safe}, visibility={template['inter_robot_visibility_radius']}, "
          f"stack spacing {spacing} ({args.spacing_per_d_safe}x d_safe)")
    worst_steps = 0
    for num_robots in FLEET_SIZES:
        config, half_width = build_config(template, num_robots, spacing)
        validated = validate_system_config(system_name="multi_robot", raw_config=config)
        stats = check_scenario(validated, num_robots, half_width, d_safe)
        worst_steps = max(worst_steps, stats["straight_line_steps"])

        # A column taller than the box runs along the side wall and past the midline.
        # Legal (workspace_bounds only bounds *sampling*) but worth saying out loud.
        if stats["stack_span"] > 2.0 * half_width:
            print(f"  note: {num_robots} robots stack {stats['stack_span']:.3f} tall in a "
                  f"{2.0 * half_width:.3f} box, so each stack reaches "
                  f"{stats['stack_span'] - half_width:.3f} past the midline")

        output_path = output_dir / f"unicycle2_swap_{num_robots:02d}.yaml"
        header = (
            f"# {num_robots}x unicycle2 corner swap for the encoder-scaling study.\n"
            f"# Generated by test/config/generate_swap_configs.py -- do not edit by hand.\n"
            f"# Robots 0..{num_robots // 2 - 1} stack downwards from the top-right corner "
            f"(+{half_width}, +{half_width}) at\n"
            f"# {spacing} apart ({args.spacing_per_d_safe}x d_safe); robots "
            f"{num_robots // 2}..{num_robots - 1} stack the same way from the\n"
            f"# bottom-left corner. Robot i's goal is its own antipode -start, "
            + (f"i.e. the start of\n# robot i+{num_robots // 2} (mod {num_robots}); "
               if num_robots % 2 == 0
               else "which at this odd N is\n# a point no robot is vacating; ")
            + f"all {num_robots} paths cross the origin.\n"
            f"# Deterministic: fixed 'start' and 'goal', randomize_goal false.\n"
            f"# Closest starting pair {stats['min_pair_distance']:.3f} (d_safe={d_safe}); "
            f"{stats['visible_neighbours']:.2f} visible neighbours at t=0;\n"
            f"# the corner robots cross {stats['longest_trip']:.3f} m, so the run needs "
            f">= {stats['straight_line_steps']} steps\n"
            f"# even in a straight line.\n"
        )
        output_path.write_text(header + render_config(config, num_robots))
        print(
            f"wrote {output_path.relative_to(PROJECT_ROOT)} "
            f"(box +-{half_width}, closest pair {stats['min_pair_distance']:.2f}, "
            f"{stats['visible_neighbours']:.2f} visible, >={stats['straight_line_steps']} steps)"
        )

    print(f"\nRun with --steps at least {math.ceil(worst_steps * 1.6)} so the largest swap "
          f"is not cut off before it can finish.")


if __name__ == "__main__":
    main()
