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

The box is written as ``workspace_bounds``, which bounds *both* the random start
positions and the random goals (``unicycle2.random_initial_state`` and
``randomize_goal_for_reset``). The older goal-relative start sampling
(``initial_position_*``) and ``goal_position_bounds`` no longer exist in the schema.

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

# Retuned since the study configs were built; check_template_matches_existing guards against it.
TEMPLATE_PATH = PROJECT_ROOT / "test/config/2_multi_unicycle2_casadi_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "test/config/study"
FLEET_SIZES = (2, 4, 6, 8, 16, 32)

Q_BLOCK = [100.0, 100.0, 10.0, 100.0, 1.0]
REFERENCE_FLEET_SIZE = 2

GOAL_BOX_HALF_WIDTHS_PER_D_SAFE = 2.5

TASK_TOLERANCES = {
    "pos_tol": 0.1,
    "theta_tol": 1.1,
    "vel_tol": 0.05,
    "omega_tol": 0.05,
}

SAMPLING_CHECK_EPISODES = 20


def first_robot_template(template: dict) -> dict:
    """The per-robot entry of a template, in either 'robots' form."""
    robots = template["robots"]
    if isinstance(robots, list):
        return robots[0]
    return {"system": robots["system"], "config": robots["config"]}


def goal_half_width(num_robots: int, d_safe: float) -> float:
    scale = GOAL_BOX_HALF_WIDTHS_PER_D_SAFE * float(d_safe)
    return round(scale * (num_robots / REFERENCE_FLEET_SIZE) ** 0.5, 3)


def assert_fleet_is_placeable(config: dict, num_robots: int) -> float:
    """Draw seeded episodes so an unplaceable fleet fails here, not mid-training."""
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


def build_config(template: dict, num_robots: int, half_width: float | None = None) -> dict:
    """A random-goal fleet config; ``half_width`` defaults to the study's sqrt(N) box."""
    robot_template = first_robot_template(template)
    if half_width is None:
        half_width = goal_half_width(num_robots, template["d_safe"])

    robot_config = {
        key: value
        for key, value in robot_template["config"].items()
        if key != "goal"
    }
    robot_config["randomize_goal"] = True
    robot_config.update(TASK_TOLERANCES)
    robot_config["workspace_bounds"] = [-half_width, half_width]

    config = {
        key: value
        for key, value in template.items()
        if key not in {"robots", "Q_diag"}
    }
    config["Q_diag"] = Q_BLOCK * num_robots
    # deepcopy: shared lists would be dumped as YAML &anchor/*alias references.
    config["robots"] = [
        {"system": robot_template["system"], "config": copy.deepcopy(robot_config)}
        for _ in range(num_robots)
    ]
    return config


class _FlowListDumper(yaml.SafeDumper):
    """Renders lists of plain scalars inline, so bound pairs stay on one line."""


def _represent_list(dumper: yaml.Dumper, data: list) -> yaml.Node:
    inline = all(isinstance(item, (int, float, str, bool)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


_FlowListDumper.add_representer(list, _represent_list)


def format_q_diag(num_robots: int) -> str:
    """Emit Q_diag as one line per robot, matching the hand-written configs."""
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


def check_template_matches_existing(template: dict) -> None:
    """Refuse to regenerate when the template would change the robots themselves."""
    existing_path = OUTPUT_DIR / "unicycle2_fleet_02.yaml"
    if not existing_path.exists():
        return
    existing = yaml.safe_load(existing_path.read_text())["robots"][0]["config"]
    candidate = first_robot_template(template)["config"]
    changed = {
        key: (existing.get(key), candidate.get(key))
        for key in ("dt", "max_linear_vel", "max_angular_vel", "max_linear_accel", "max_angular_accel")
        if existing.get(key) != candidate.get(key)
    }
    if changed:
        details = ", ".join(f"{k}: {old} -> {new}" for k, (old, new) in changed.items())
        raise SystemExit(
            f"{TEMPLATE_PATH.name} would change the robot dynamics of every study scenario "
            f"({details}).\nExisting checkpoints were trained under the current values; "
            "delete test/config/study/unicycle2_fleet_*.yaml to confirm the change."
        )


def main() -> None:
    template = yaml.safe_load(TEMPLATE_PATH.read_text())
    check_template_matches_existing(template)
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
            f"# {num_robots}x unicycle2 expert config, +-{half_width} m box, "
            f"~{mean_visible:.2f} visible neighbours.\n"
            f"# Generated by test/config/generate_fleet_configs.py -- do not edit by hand.\n"
        )
        output_path.write_text(header + render_config(config, num_robots))
        print(
            f"wrote {output_path.relative_to(PROJECT_ROOT)} "
            f"({num_robots} robots, goal box +-{half_width}, "
            f"{mean_visible:.2f} visible neighbours)"
        )


if __name__ == "__main__":
    main()
