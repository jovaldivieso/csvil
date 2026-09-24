"""Generate antipodal-circle ("position swap") evaluation configs for unicycle2 fleets.

Robot i starts at angle 2*pi*i/N on a circle and its goal is the antipode, which for
an even fleet is exactly the start of robot i + N/2. Every robot must therefore cross
the centre, and every path conflicts with the one coming the other way -- the hardest
standard layout for a decentralized collision-avoidance policy, and fully
deterministic, unlike the randomized-goal configs used for training.

Both endpoints are written into the config: ``start`` per robot (the schema already
supports it, see ``_validate_multi_robot_start`` in core/config.py) and a fixed
``goal`` with ``randomize_goal: false``. Evaluate them with

    python test/evaluate_scaling.py --use-config-start ...

Radius: the ring is sized so that **every fleet size rings at the same robot density**,
R = sqrt(N / density) / 2, i.e. the circle inscribed in the square a fleet of that
density would occupy. The alternative -- one radius for every fleet -- makes the ring
density grow with N (4x from 2 to 8 robots at radius 3), so a "more robots" curve would
really be a "more crowding" curve, which is exactly what the other evaluation axes are
built to avoid. Adjacent robots sit 2*R*sin(pi/N) apart; that spacing shrinks slowly
with N, so the generator checks it against d_safe and says how much margin is left.

--density defaults to the training density of the study's runs (0.1667 robots/m^2,
the +-1.73/2.45/3.0/3.46 boxes of learning/config/study/data_*_n<NN>), so the ring is
in-distribution in density and differs from training only in the layout.

Usage:
    python test/config/generate_circle_configs.py
    python test/config/generate_circle_configs.py --density 0.0556 --out-subdir sparse
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
TASK_TOLERANCES = _fleet.TASK_TOLERANCES

# The 2-robot fleet config rather than test/config/2_multi_unicycle2_casadi_config.yaml:
# the rings must match the scenarios the policies are actually trained and scored on,
# and the raw template has since been retuned (4 m/s instead of 1) for other work.
# Everything taken from here is fleet-size independent -- dt, d_safe, d_collision,
# visibility, horizon, cost weights and the per-robot dynamics; the ring writes its own
# starts, goals and Q_diag.
TEMPLATE_PATH = PROJECT_ROOT / "test/config/study/unicycle2_fleet_02.yaml"
OUTPUT_DIR = PROJECT_ROOT / "test/config/study/circle"
FLEET_SIZES = (2, 4, 6, 8, 16, 32)

# Neighbour spacing on the ring, in units of d_safe. Below 1.0 the starting formation
# is in collision; 1.5 is the margin this generator prefers and warns below.
RING_SPACING_PER_D_SAFE = 1.5
# Robots per m^2 the ring is sized for. The default is what the study's runs train at
# (learning/config/study/generate_data_pilot_configs.py, --density-factor 3).
DEFAULT_DENSITY = 0.1667


def ring_radius(num_robots: int, density: float) -> float:
    """Radius that puts `num_robots` on a ring at `density` robots per m^2.

    The circle inscribed in the square that many robots would occupy at that density,
    so the ring's crowding matches the randomized scenarios of the same density.
    """
    return round(math.sqrt(num_robots / density) / 2.0, 4)


def ring_spacing(num_robots: int, radius: float) -> float:
    """Distance between neighbouring robots on the ring."""
    return 2.0 * radius * math.sin(math.pi / num_robots)


def robot_endpoints(num_robots: int, radius: float, robot_idx: int):
    """Start state and goal for one robot: the goal is the antipodal start."""
    angle = 2.0 * math.pi * robot_idx / num_robots
    x, y = radius * math.cos(angle), radius * math.sin(angle)
    # Face straight across the circle: the goal lies at (-x, -y), so the heading
    # is the start angle turned by pi.
    heading = math.atan2(-y - y, -x - x)
    start = [round(x, 6), round(y, 6), round(heading, 6), 0.0, 0.0]
    goal = [round(-x, 6), round(-y, 6), round(heading, 6)]
    return start, goal


def build_config(template: dict, num_robots: int, density: float) -> tuple[dict, float]:
    if num_robots % 2 != 0:
        raise SystemExit(
            f"Fleet size {num_robots} is odd; the antipode of a start is then not "
            "another robot's start, which is the property this scenario is for."
        )
    robot_template = first_robot_template(template)
    radius = ring_radius(num_robots, density)

    base_robot_config = {
        key: value
        for key, value in robot_template["config"].items()
        if key not in {"goal", "start"}
    }

    robots = []
    for robot_idx in range(num_robots):
        start, goal = robot_endpoints(num_robots, radius, robot_idx)
        robot_config = copy.deepcopy(base_robot_config)
        robot_config["randomize_goal"] = False
        robot_config.update(TASK_TOLERANCES)
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
    return config, radius


def render_config(config: dict, num_robots: int) -> str:
    scalars = {key: value for key, value in config.items() if key not in {"Q_diag", "robots"}}
    return (
        yaml.safe_dump(scalars, sort_keys=False)
        + format_q_diag(num_robots)
        + yaml.dump({"robots": config["robots"]}, Dumper=_FlowListDumper, sort_keys=False)
    )


def check_scenario(config: dict, num_robots: int, radius: float, d_safe: float) -> dict:
    """Verify the ring is actually feasible and report what the run will need."""
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
        raise SystemExit(
            f"{num_robots}-robot ring starts in collision at radius {radius} "
            f"(spacing {ring_spacing(num_robots, radius):.3f} < d_safe {d_safe}); "
            "lower --density."
        )

    positions = np.stack([entry_start(entry)[:2] for entry in config["robots"]])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)

    # Every goal must coincide with some other robot's start -- the property asked for.
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
    straight_line_steps = math.ceil(2.0 * radius / (max_linear_vel * dt))
    return {
        "min_pair_distance": float(distances.min()),
        "visible_neighbours": visible,
        "straight_line_steps": straight_line_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY,
                        help=f"robots per m^2 the rings are sized for (default {DEFAULT_DENSITY})")
    parser.add_argument("--out-subdir", default=None,
                        help="write into test/config/study/circle/<subdir> instead of circle/")
    args = parser.parse_args()

    template = yaml.safe_load(TEMPLATE_PATH.read_text())
    d_safe = float(template["d_safe"])
    output_dir = OUTPUT_DIR / args.out_subdir if args.out_subdir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"template d_safe={d_safe}, visibility={template['inter_robot_visibility_radius']}, "
          f"ring density {args.density} robots/m^2")
    worst_steps = 0
    for num_robots in FLEET_SIZES:
        config, radius = build_config(template, num_robots, args.density)
        validated = validate_system_config(system_name="multi_robot", raw_config=config)
        stats = check_scenario(validated, num_robots, radius, d_safe)
        worst_steps = max(worst_steps, stats["straight_line_steps"])

        spacing = ring_spacing(num_robots, radius)
        # Above d_safe the ring is feasible, but a thin margin means the robots start
        # nearly touching, so the run is decided in the first few steps.
        if spacing < RING_SPACING_PER_D_SAFE * d_safe:
            print(f"  note: {num_robots} robots sit {spacing:.3f} apart on the ring, "
                  f"{spacing / d_safe:.2f}x d_safe (preferred >= {RING_SPACING_PER_D_SAFE})")

        output_path = output_dir / f"unicycle2_circle_{num_robots:02d}.yaml"
        header = (
            f"# {num_robots}x unicycle2 antipodal-circle swap for the encoder-scaling study.\n"
            f"# Generated by test/config/generate_circle_configs.py -- do not edit by hand.\n"
            f"# Robot i starts at angle 2*pi*i/{num_robots} on a circle of radius {radius},\n"
            f"# sized for {args.density} robots/m^2 so every fleet size rings at the same density.\n"
            f"# and targets the antipode, which is the start of robot i+{num_robots // 2}.\n"
            f"# Deterministic: fixed 'start' and 'goal', randomize_goal false.\n"
            f"# Closest starting pair {stats['min_pair_distance']:.3f} (d_safe={d_safe}); "
            f"{stats['visible_neighbours']:.2f} visible neighbours at t=0;\n"
            f"# needs >= {stats['straight_line_steps']} steps even in a straight line.\n"
        )
        output_path.write_text(header + render_config(config, num_robots))
        print(
            f"wrote {output_path.relative_to(PROJECT_ROOT)} "
            f"(radius {radius}, closest pair {stats['min_pair_distance']:.2f}, "
            f"{stats['visible_neighbours']:.2f} visible, >={stats['straight_line_steps']} steps)"
        )

    print(f"\nRun with --steps at least {math.ceil(worst_steps * 1.6)} so the largest ring "
          f"is not cut off before it can finish.")


if __name__ == "__main__":
    main()
