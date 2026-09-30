"""Generate the head-on crash ladder: one scenario, five initial speeds.

Two robots start on a collision course -- antipodal, facing each other, each one's goal
being the other's start -- and the *only* thing that changes from config to config is how
fast they are already moving when the episode begins. Everything else (separation,
headings, goals, box, robot) is held fixed, so a policy's success rate across the five is
a clean difficulty curve rather than five unrelated scenarios.

The point is to find where each policy breaks, in order: the deterministic MLP is
expected to go first, then flow, and the ladder then runs on far enough to say something
about where SafeFlow's projection stops being able to save it.

Geometry: the training ring, not a new layout
---------------------------------------------
Starts are at +-``RADIUS`` on the x-axis with headings 0 and pi -- which is exactly the
``nominal`` ring layout that half of every DAgger round already trains on, and exactly
the circle-evaluation layout. That is deliberate: RADIUS equals the policy configs'
workspace half-width, so the goal vectors stay inside the range the policies saw in
training, and the only out-of-distribution quantity in the whole ladder is the initial
velocity. Push the robots further apart to buy room and the goal vector grows past
anything training produced -- the policies would then degrade for a reason that has
nothing to do with the crash.

Exact head-on symmetry is kept rather than broken with a lateral offset. It is solvable:
the observations are ego-centric, so two mirrored robots that both turn the same way *in
their own frame* move to opposite sides of the world and separate. What it does remove is
any hint about which way to go, which is the part that should hurt.

Why the ladder has a ceiling, and where it is
---------------------------------------------
At separation D the two robots close at 2*v0. Braking alone buys 2 * v0^2/(2a) of closure
before they stop, so pure braking stops being enough once v0^2/a > D - d_collision; past
that they *must* turn, and the turn radius at speed is v0/max_angular_vel. Both limits
tighten with v0, which is what makes this a ladder and not a set.

There is therefore a speed past which the *expert* cannot solve it either, and a rung
beyond that measures the task, not the policy. print_feasibility() reports the expert's
own success per rung so the ceiling is visible; keep the top rung at or inside it.

Usage:
    python test/config/generate_crash_configs.py
    python test/config/generate_crash_configs.py --speeds 0 0.4 0.8 1.2 1.5
    python test/config/generate_crash_configs.py --calibrate      # fine sweep + expert
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

_spec = importlib.util.spec_from_file_location(
    "_generate_fleet_configs", Path(__file__).resolve().parent / "generate_fleet_configs.py"
)
_fleet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fleet)

NUM_ROBOTS = 2

# The 0.1 m robot is the default, but the ladder is not specific to it -- the same
# scenario applies to any 2-robot scenario config, e.g. the 1 m study robot the
# data_mid / data_mid_best checkpoints were trained on. Each entry is
# (scenario template, ring radius, output directory).
#
# The radius is the *policy* training box half-width, not the scenario config's own box,
# because that is what keeps the goal vectors inside the range training produced. It
# cannot be read off the template: the pilots override the box in the policy config. Both
# presets put the pair 3.46 x d_collision apart, so the two ladders are dimensionally the
# same scenario and their figures can sit side by side.
ROBOTS = {
    "small": {
        "template": PROJECT_ROOT / "test/config/study/unicycle2_fleet_02_small.yaml",
        "radius": 0.1732,
        "out_dir": PROJECT_ROOT / "test/config/study/crash",
        # 0.1 m robot, max_linear_vel 1.5.
        "speeds": (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
    },
    "1m": {
        "template": PROJECT_ROOT / "test/config/study/unicycle2_fleet_02.yaml",
        "radius": 1.732,
        "out_dir": PROJECT_ROOT / "test/config/study/crash_1m",
        # 1 m robot, max_linear_vel 1.0, so the ladder tops out lower. Run --calibrate
        # before trusting the top rungs: the expert's own ceiling has not been measured
        # for this robot the way it was for the small one.
        "speeds": (0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    },
}
DEFAULT_ROBOT = "small"

# Bound by apply_robot() from the chosen preset, so the rest of the module can go on
# reading them as plain constants.
TEMPLATE_PATH = ROBOTS[DEFAULT_ROBOT]["template"]
OUTPUT_DIR = ROBOTS[DEFAULT_ROBOT]["out_dir"]
RADIUS = ROBOTS[DEFAULT_ROBOT]["radius"]
DEFAULT_SPEEDS = ROBOTS[DEFAULT_ROBOT]["speeds"]

# Why the small ladder's rungs are where they are, since the shape carried over to the
# 1 m preset: the first version spaced them over the robot's whole speed range on the
# assumption the interesting region was in the middle. Measured, every policy but
# SafeFlow was already failing by 0.4 m/s, so the rungs are now stated in absolute m/s
# and spaced to resolve the onset -- coarse where every policy is comfortable, 0.1 steps
# through the band where failures appear. The top rung is set by the *expert's* ceiling
# (--calibrate), not by the robot's limit, so that the ladder measures the policy rather
# than the task.


def apply_robot(name: str) -> None:
    """Point the module's constants at one robot preset."""
    global TEMPLATE_PATH, OUTPUT_DIR, RADIUS, DEFAULT_SPEEDS
    preset = ROBOTS[name]
    TEMPLATE_PATH = preset["template"]
    OUTPUT_DIR = preset["out_dir"]
    RADIUS = preset["radius"]
    DEFAULT_SPEEDS = preset["speeds"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--robot", choices=sorted(ROBOTS), default=DEFAULT_ROBOT,
        help=f"which scenario template and ring radius to build the ladder on "
             f"(default {DEFAULT_ROBOT})",
    )
    parser.add_argument(
        "--speeds", type=float, nargs="+", default=None,
        help="absolute initial speeds in m/s (default: the chosen robot's preset)",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="roll the CasADi expert down a fine speed sweep and report where it stops "
             "solving, then exit without writing",
    )
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    return parser.parse_args()


def build_config(template: dict, initial_speed: float) -> dict:
    """Head-on pair at +-RADIUS, both already moving at ``initial_speed``."""
    robot_template = template["robots"][0]["config"]
    base = {
        key: value
        for key, value in robot_template.items()
        if key not in {"goal", "start", "randomize_goal", "workspace_bounds"}
    }

    robots = []
    for robot_idx in range(NUM_ROBOTS):
        sign = -1.0 if robot_idx == 0 else 1.0
        x = sign * RADIUS
        # Facing straight at the other robot, which is also straight at the goal.
        heading = 0.0 if robot_idx == 0 else math.pi
        robot_config = copy.deepcopy(base)
        robot_config["randomize_goal"] = False
        robot_config["workspace_bounds"] = [-RADIUS, RADIUS]
        # state = [x, y, theta, v, omega]; the ladder varies v and nothing else.
        robot_config["start"] = [round(x, 6), 0.0, round(heading, 6), round(initial_speed, 6), 0.0]
        robot_config["goal"] = [round(-x, 6), 0.0, round(heading, 6)]
        robots.append({"system": "unicycle2", "config": robot_config})

    config = {
        key: value for key, value in template.items() if key not in {"robots", "Q_diag"}
    }
    config["Q_diag"] = list(template["Q_diag"][:5]) * NUM_ROBOTS
    config["robots"] = robots
    return config


def format_q_diag(block: list[float], num_robots: int) -> str:
    """One line per robot, from the block actually in the config.

    Deliberately not generate_fleet_configs.format_q_diag: that one renders its own
    module-level Q_BLOCK (the 1 m robot's weights) and ignores whatever config it is
    called for, so reusing it silently wrote the wrong cost weights into every rung
    while the in-memory validation saw the right ones.
    """
    lines = []
    for robot_idx in range(num_robots):
        values = ", ".join(str(value) for value in block)
        prefix = "Q_diag: [" if robot_idx == 0 else " " * 9
        suffix = "]" if robot_idx == num_robots - 1 else ","
        lines.append(f"{prefix}{values}{suffix}")
    return "\n".join(lines) + "\n"


def render(config: dict, initial_speed: float, template: dict) -> str:
    robot = template["robots"][0]["config"]
    v_max = float(robot["max_linear_vel"])
    a_v = float(robot["max_linear_accel"])
    w_max = float(robot["max_angular_vel"])
    d_collision = float(template["d_collision"])
    separation = 2.0 * RADIUS
    brake_closure = initial_speed ** 2 / a_v          # both robots braking, combined
    turn_radius = initial_speed / w_max if initial_speed > 0 else 0.0
    gap_if_only_braking = separation - brake_closure

    header = (
        f"# Head-on crash course, 2x unicycle2, initial speed {initial_speed:.3f} m/s "
        f"({initial_speed / v_max:.0%} of max).\n"
        f"# Generated by test/config/generate_crash_configs.py -- do not edit by hand.\n"
        f"#\n"
        f"# Rung of a ladder in which ONLY the initial speed changes: both robots start\n"
        f"# {separation:.4f} m apart ({separation / d_collision:.2f} x d_collision) on the x-axis,\n"
        f"# each facing and already driving straight at the other, each one's goal being\n"
        f"# the other's start. Same layout as the training ring and the circle scenario.\n"
        f"#\n"
        f"# If both robots only brake they still close {brake_closure:.4f} m, leaving\n"
        f"# {gap_if_only_braking:.4f} m of the {separation:.4f} m gap -- "
        f"{'ENOUGH' if gap_if_only_braking > d_collision else 'NOT enough'} against "
        f"d_collision {d_collision}.\n"
        f"# Turn radius at this speed is {turn_radius:.4f} m "
        f"({turn_radius / d_collision:.2f} x d_collision).\n"
        f"#\n"
        f"# Evaluate with --use-config-start. The step budget must cover the final\n"
        f"# in-place rotation, not just the drive: use STEP_BUDGET_FACTOR=6 (~278 steps),\n"
        f"# since the distance-derived default of 3 stops short at ~139.\n"
    )
    scalars = {k: v for k, v in config.items() if k not in {"Q_diag", "robots"}}
    return (
        header
        + yaml.safe_dump(scalars, sort_keys=False, default_flow_style=None)
        + format_q_diag(list(config["Q_diag"][:5]), NUM_ROBOTS)
        + yaml.dump({"robots": config["robots"]}, Dumper=_fleet._FlowListDumper, sort_keys=False)
    )


def run_expert(raw: dict, validated: dict, steps: int = 300) -> tuple[bool, float, int | None]:
    """(solved, closest approach / d_collision, steps used) for the centralized expert.

    The start state comes from the *raw* config, the way test/evaluate_scaling.py's
    --use-config-start reads it: validation folds 'start' into the simulator rather than
    leaving it on the robot entries, so reading it back off the validated dict fails.
    """
    import numpy as np

    from core.factory import DynamicsFactory, PlannerFactory
    from planning.casadi_planner import PlannerSolveError

    sys.path.insert(0, str(PROJECT_ROOT / "test"))
    from evaluate_scaling import config_start_state

    simulator = DynamicsFactory.create(system_name="multi_robot", config=validated)
    planner = PlannerFactory.create("casadi", simulator=simulator, config=validated)
    d_collision = float(validated["d_collision"])
    state = np.asarray(config_start_state(raw), dtype=float)
    planner.reset()
    closest, done_at = math.inf, None
    for step in range(steps):
        try:
            action = planner(simulator.observe(state, validate=False))
        except PlannerSolveError:
            return False, closest / d_collision, None
        state = simulator.step(state, action)
        positions = [
            np.asarray(state)[s.start:s.start + 2] for s in simulator.robot_state_slices
        ]
        closest = min(closest, float(np.linalg.norm(positions[0] - positions[1])))
        if simulator.is_done(state):
            done_at = step + 1
            break
    solved = done_at is not None and closest >= d_collision
    return solved, closest / d_collision, done_at


def print_feasibility(template: dict, speeds) -> None:
    """Where the expert itself stops solving -- the ceiling the ladder must respect."""
    print(f"{'v0 (m/s)':>9} {'% of max':>9} {'expert':>8} {'closest/d_coll':>15} {'steps':>7}")
    for speed in speeds:
        raw = build_config(template, speed)
        validated = validate_system_config(system_name="multi_robot", raw_config=raw)
        solved, closest, used = run_expert(raw, validated)
        v_max = float(template["robots"][0]["config"]["max_linear_vel"])
        verdict = "solves" if solved else ("COLLIDES" if closest < 1.0 else "timeout")
        print(f"{speed:9.3f} {speed / v_max:8.0%} {verdict:>8} {closest:15.2f} "
              f"{used if used else '-':>7}", flush=True)


def main() -> None:
    args = parse_args()
    apply_robot(args.robot)
    template = yaml.safe_load(TEMPLATE_PATH.read_text())
    v_max = float(template["robots"][0]["config"]["max_linear_vel"])
    print(f"robot preset '{args.robot}': {TEMPLATE_PATH.name}, ring radius {RADIUS} m "
          f"-> separation {2 * RADIUS:.4f} m "
          f"({2 * RADIUS / float(template['d_collision']):.2f} x d_collision)")

    if args.calibrate:
        sweep = [round(v_max * i / 12.0, 4) for i in range(13)]
        print(f"expert feasibility sweep, separation {2 * RADIUS:.4f} m, "
              f"a_v={template['robots'][0]['config']['max_linear_accel']}, "
              f"w_max={template['robots'][0]['config']['max_angular_vel']}\n")
        print_feasibility(template, sweep)
        return

    speeds = args.speeds if args.speeds else list(DEFAULT_SPEEDS)
    print(f"crash ladder: {len(speeds)} rungs, v0 = {speeds} m/s (max {v_max})")
    print_feasibility(template, speeds)

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, float]] = []
    for stale in OUTPUT_DIR.glob("*.yaml"):
        stale.unlink()
    for speed in speeds:
        config = build_config(template, speed)
        validate_system_config(system_name="multi_robot", raw_config=config)
        # Speed in mm/s, zero padded, so the glob sorts into ladder order.
        name = f"unicycle2_crash_v{int(round(speed * 1000)):04d}.yaml"
        (OUTPUT_DIR / name).write_text(render(config, speed, template))
        written.append((OUTPUT_DIR / name, speed))
        print(f"wrote {(OUTPUT_DIR / name).relative_to(PROJECT_ROOT)}")

    # Re-read every file and roll the expert out on it. The in-memory dict validating is
    # not the same claim as the YAML on disk being right -- a rendering bug put the 1 m
    # robot's Q_diag into these files while the in-memory check was passing.
    from core.config import load_and_validate_system_config
    print("\nfrom disk:")
    for path, speed in written:
        raw = yaml.safe_load(path.read_text())
        validated = load_and_validate_system_config("multi_robot", path)
        solved, closest, used = run_expert(raw, validated)
        q_ok = list(raw["Q_diag"][:5]) == list(template["Q_diag"][:5])
        print(f"  {path.name}  v0={speed:<6} expert "
              f"{'solves' if solved else 'FAILS':>6}  closest={closest:.2f}  "
              f"steps={used}  Q_diag {'matches template' if q_ok else 'MISMATCH'}")


if __name__ == "__main__":
    main()
