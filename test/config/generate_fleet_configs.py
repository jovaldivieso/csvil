"""Generate homogeneous unicycle2 fleet configs for the encoder-scaling study.

Every fleet size shares one per-robot task definition (dynamics limits, tolerances,
start-offset distribution, sensing radius) so the only quantity that varies is the
number of robots. The goal box grows as sqrt(N/2) to hold goal density -- and
therefore the expected number of visible neighbours -- bounded; with a fixed box the
fleet-level rejection sampler in ``MultiRobotSystem`` cannot place 32 goals at all.

The box is sized in units of ``d_safe``, because that sampler redraws *all* goals
until every pair clears ``d_safe`` and gives up after a fixed number of attempts.
Raising ``d_safe`` without widening the box pushes the fleet past the packing
density where random rejection sampling still terminates, and training then dies
at the first episode. Every generated config is sampled here before it is written
so that failure surfaces now rather than hours into a run.

Usage:
    python test/config/generate_fleet_configs.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import validate_system_config  # noqa: E402

TEMPLATE_PATH = PROJECT_ROOT / "test/config/multi_unicycle2_casadi_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "test/config/study"
FLEET_SIZES = (2, 4, 6, 8, 16, 32)

# Per-robot state-cost block for unicycle2: [x, y, theta, v, omega].
Q_BLOCK = [100.0, 100.0, 10.0, 100.0, 1.0]
REFERENCE_FLEET_SIZE = 2

# Goal-box half-width at the reference fleet size, in units of d_safe. 2.5 is the
# smallest value that samples reliably for every fleet size up to 32 robots;
# lower it and the large fleets stop being placeable, raise it and the robots
# spread out until they rarely see one another.
GOAL_BOX_HALF_WIDTHS_PER_D_SAFE = 2.5

# Convergence tolerances for the generated study scenarios, shared by the random
# fleets and the circle rings. These belong to the *task*, not to a policy:
# success_rate is only comparable across policies if every one of them is judged by
# the same criterion, and the scenario config is the file they all share. So the
# study policy configs carry no `tolerance_overrides` -- there is exactly one source,
# and it is here.
#
# unicycle2.is_done() requires all four simultaneously: within pos_tol of the goal
# position, within theta_tol of the goal heading (wrapped), and nearly stopped
# (|v| < vel_tol and |omega| < omega_tol). Absent from a config, each falls back to
# the simulator default of 0.05, which is a 22x stricter heading criterion -- that
# silent fallback is why these keys are injected explicitly rather than inherited
# from the canonical template.
TASK_TOLERANCES = {
    "pos_tol": 0.1,
    "theta_tol": 1.1,
    "vel_tol": 0.05,
    "omega_tol": 0.05,
}

# Per-robot initial-state sampling bounds, used by the simulator's random reset.
INITIAL_STATE_SAMPLING = {
    "initial_position_min_goal_distance": 0.05,
    "initial_position_radius_bounds": [0.05, 3.0],
}

# Seeds drawn per config to prove the fleet is actually placeable.
SAMPLING_CHECK_EPISODES = 20


def first_robot_template(template: dict) -> dict:
    """The per-robot entry of a template, in either 'robots' form.

    core/config.py accepts both the homogeneous-fleet shorthand
    ({num_robots, system, config}) and the long list form, and the canonical
    template has since switched to the shorthand. Indexing ["robots"][0] therefore
    raises KeyError on it, which is what silently froze these generated configs.
    """
    robots = template["robots"]
    if isinstance(robots, list):
        return robots[0]
    return {"system": robots["system"], "config": robots["config"]}


def goal_half_width(num_robots: int, d_safe: float) -> float:
    scale = GOAL_BOX_HALF_WIDTHS_PER_D_SAFE * float(d_safe)
    return round(scale * (num_robots / REFERENCE_FLEET_SIZE) ** 0.5, 3)


def assert_fleet_is_placeable(config: dict, num_robots: int) -> float:
    """Draw seeded episodes so an unplaceable fleet fails here, not mid-training.

    Returns the mean number of visible neighbours, which is what the neighbour
    encoders actually get to encode -- near zero means the fleet is so spread out
    that the study would compare encoders on empty inputs.
    """
    import numpy as np

    from core.factory import DynamicsFactory

    simulator = DynamicsFactory.create(system_name="multi_robot", config=config)
    visible_counts = []
    for episode in range(SAMPLING_CHECK_EPISODES):
        rng = np.random.default_rng(3000 + episode)
        try:
            simulator.randomize_goal_for_reset(rng)
            state = simulator.random_initial_state(rng)
        except RuntimeError as exc:
            raise SystemExit(
                f"{num_robots}-robot fleet is not placeable: {exc}\n"
                f"d_safe={config['d_safe']} needs a wider goal box; raise "
                f"GOAL_BOX_HALF_WIDTHS_PER_D_SAFE above "
                f"{GOAL_BOX_HALF_WIDTHS_PER_D_SAFE}."
            ) from exc
        observation = simulator.observe(state, validate=False)
        visible_counts.append(
            np.mean([
                simulator.decentralized_policy_observation(observation, robot_id)["observation.neighbor_mask"].sum()
                for robot_id in range(num_robots)
            ])
        )
    return float(np.mean(visible_counts))


def build_config(template: dict, num_robots: int) -> dict:
    robot_template = first_robot_template(template)
    half_width = goal_half_width(num_robots, template["d_safe"])

    robot_config = {
        key: value
        for key, value in robot_template["config"].items()
        if key != "goal"  # goals are randomized per episode
    }
    robot_config["randomize_goal"] = True
    robot_config.update(INITIAL_STATE_SAMPLING)
    robot_config.update(TASK_TOLERANCES)
    robot_config["goal_position_bounds"] = [-half_width, half_width]

    config = {
        key: value
        for key, value in template.items()
        if key not in {"robots", "Q_diag"}
    }
    config["Q_diag"] = Q_BLOCK * num_robots
    # deepcopy per robot: a shallow copy would share the bounds *lists* between
    # entries, and yaml.safe_dump renders repeated objects as &anchor/*alias
    # references instead of writing the values out at each robot.
    config["robots"] = [
        {"system": robot_template["system"], "config": copy.deepcopy(robot_config)}
        for _ in range(num_robots)
    ]
    return config


class _FlowListDumper(yaml.SafeDumper):
    """Renders lists of plain scalars inline, so bound pairs stay on one line.

    Only scalar lists: the robot list holds mappings, and forcing that inline
    collapses the whole fleet into one unreadable wrapped blob.
    """


def _represent_list(dumper: yaml.Dumper, data: list) -> yaml.Node:
    inline = all(isinstance(item, (int, float, str, bool)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


_FlowListDumper.add_representer(list, _represent_list)


def format_q_diag(num_robots: int) -> str:
    """Emit Q_diag as one line per robot, matching the hand-written configs.

    The default block style would put all 5*N entries on their own lines (160 of
    them at N=32) and hide the per-robot block structure.
    """
    lines = []
    for robot_idx in range(num_robots):
        values = ", ".join(str(value) for value in Q_BLOCK)
        prefix = "Q_diag: [" if robot_idx == 0 else " " * 9
        suffix = "]" if robot_idx == num_robots - 1 else ","
        lines.append(f"{prefix}{values}{suffix}")
    return "\n".join(lines) + "\n"


def render_config(config: dict, num_robots: int) -> str:
    scalars = {key: value for key, value in config.items() if key not in {"Q_diag", "robots"}}
    return (
        yaml.safe_dump(scalars, sort_keys=False)
        + format_q_diag(num_robots)
        + yaml.dump({"robots": config["robots"]}, Dumper=_FlowListDumper, sort_keys=False)
    )


def main() -> None:
    template = yaml.safe_load(TEMPLATE_PATH.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    d_safe = template["d_safe"]
    print(f"template d_safe={d_safe}, visibility={template['inter_robot_visibility_radius']}")

    for num_robots in FLEET_SIZES:
        config = build_config(template, num_robots)
        validated = validate_system_config(system_name="multi_robot", raw_config=config)
        mean_visible = assert_fleet_is_placeable(validated, num_robots)
        half_width = goal_half_width(num_robots, d_safe)
        output_path = OUTPUT_DIR / f"unicycle2_fleet_{num_robots:02d}.yaml"
        header = (
            f"# {num_robots}x unicycle2 fleet for the encoder-scaling study.\n"
            f"# Generated by test/config/generate_fleet_configs.py -- do not edit by hand.\n"
            f"# Per-robot task is identical across fleet sizes; only 'robots', 'Q_diag' and\n"
            f"# 'goal_position_bounds' (+-{half_width}) vary with the fleet size.\n"
            f"# Sized for d_safe={d_safe}; mean visible neighbours ~{mean_visible:.2f}.\n"
        )
        output_path.write_text(header + render_config(config, num_robots))
        print(
            f"wrote {output_path.relative_to(PROJECT_ROOT)} "
            f"({num_robots} robots, goal box +-{half_width}, "
            f"{mean_visible:.2f} visible neighbours)"
        )


if __name__ == "__main__":
    main()
