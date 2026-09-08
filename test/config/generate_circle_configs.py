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

Radius: adjacent robots on the circle sit 2*R*sin(pi/N) apart, so R has to grow with
the fleet or the *starting* formation already violates d_safe. R is the larger of a
floor (so small fleets still travel a meaningful distance) and the spacing
requirement, which keeps the ring feasible out to 32 robots.

Usage:
    python test/config/generate_circle_configs.py
"""

from __future__ import annotations

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

TEMPLATE_PATH = PROJECT_ROOT / "test/config/multi_unicycle2_casadi_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "test/config/study/circle"
FLEET_SIZES = (2, 4, 6, 8, 16, 32)

# Neighbour spacing on the ring, in units of d_safe. 1.5 leaves the starting
# formation clearly feasible without spreading the ring so wide that robots never
# meet before reaching the centre.
RING_SPACING_PER_D_SAFE = 1.5
# Smallest ring radius, so a 2-robot swap is not a trivially short hop.
MIN_RADIUS = 3.0


def ring_radius(num_robots: int, d_safe: float) -> float:
    """Radius where neighbouring starts clear d_safe with margin."""
    required = RING_SPACING_PER_D_SAFE * d_safe / (2.0 * math.sin(math.pi / num_robots))
    return round(max(MIN_RADIUS, required), 4)


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


def build_config(template: dict, num_robots: int) -> tuple[dict, float]:
    if num_robots % 2 != 0:
        raise SystemExit(
            f"Fleet size {num_robots} is odd; the antipode of a start is then not "
            "another robot's start, which is the property this scenario is for."
        )
    robot_template = template["robots"][0]
    d_safe = float(template["d_safe"])
    radius = ring_radius(num_robots, d_safe)

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
            f"{num_robots}-robot ring starts in collision at radius {radius}; "
            f"raise RING_SPACING_PER_D_SAFE above {RING_SPACING_PER_D_SAFE}."
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
    max_speed = float(config["robots"][0]["config"]["max_speed"])
    straight_line_steps = math.ceil(2.0 * radius / (max_speed * dt))
    return {
        "min_pair_distance": float(distances.min()),
        "visible_neighbours": visible,
        "straight_line_steps": straight_line_steps,
    }


def main() -> None:
    template = yaml.safe_load(TEMPLATE_PATH.read_text())
    d_safe = float(template["d_safe"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"template d_safe={d_safe}, visibility={template['inter_robot_visibility_radius']}")
    worst_steps = 0
    for num_robots in FLEET_SIZES:
        config, radius = build_config(template, num_robots)
        validated = validate_system_config(system_name="multi_robot", raw_config=config)
        stats = check_scenario(validated, num_robots, radius, d_safe)
        worst_steps = max(worst_steps, stats["straight_line_steps"])

        output_path = OUTPUT_DIR / f"unicycle2_circle_{num_robots:02d}.yaml"
        header = (
            f"# {num_robots}x unicycle2 antipodal-circle swap for the encoder-scaling study.\n"
            f"# Generated by test/config/generate_circle_configs.py -- do not edit by hand.\n"
            f"# Robot i starts at angle 2*pi*i/{num_robots} on a circle of radius {radius}\n"
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
