"""Generate per-fleet-size policy configs for the encoder study (study 2).

The study trains one policy per (encoder, policy head, training fleet size) cell.
The template files next to this script -- ``<encoder>_<head>_config.yaml`` -- hold
the ``model`` section, the only thing that is supposed to differ between cells.
This script copies each template once per fleet size and appends the ``training``
section, which is where the fleet size matters:

``initial_states`` / ``goal_states`` pin the opening episodes of every DAgger
round to antipodal-ring layouts -- the training-time counterpart of the ``circle``
evaluation scenario. Those coordinates are per-robot, so a config only fits the
fleet size it was generated for; hence one directory per size:
    learning/config/study/n<NN>/<encoder>_<head>.yaml

Train one directory against the expert config of the same fleet size with
``python learning/train_grid.py <experiment> learning/config/study/n<NN> <expert config>``.

The ``data_*_n<NN>/`` directories are hand-adapted examples of this output: a denser
training box (``workspace_bounds``), episodes sized to a fixed data budget, and a
different ring share.

Episodes beyond the provided list fall back to the expert config's randomized
sampling (see ``collect_dagger_rollouts``), so every round is part ring, part
random. Trajectories per round and the ring share (``RING_FRACTION``) are the
same for every fleet size.

Layouts vary along the axes the policy can actually perceive. Observations are
ego-centric (``global_vector_to_ego``), so rotating a whole ring is close to a
no-op and is deliberately not used as a variation; radius, start heading,
symmetry breaking and goal assignment are.

Usage:
    python learning/config/study/generate_study_policy_configs.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_and_validate_system_config  # noqa: E402
from core.factory import DynamicsFactory  # noqa: E402
from systems.initial_state_utils import (  # noqa: E402
    normalize_goal_state_specs,
    normalize_initial_state_specs,
)

CONFIG_DIR = Path(__file__).resolve().parent
FLEET_CONFIG_TEMPLATE = "test/config/study/unicycle2_fleet_%02d.yaml"
ENCODERS = ("deepset", "transformer", "gnn")

HEADS = {
    "mlp": (1, "single step"),
    "flow": (10, "action chunk"),
}
FLEET_SIZES = (2, 4, 6, 8)

TRAJECTORIES_PER_ROUND = 200
DAGGER_ITERATIONS = 5

REFERENCE_STEPS_PER_TRAJECTORY = 250
REFERENCE_HALF_WIDTH = 3.0


def steps_per_trajectory(half_width: float) -> int:
    """Episode budget for a fleet whose workspace is +-half_width, rounded to 50."""
    scaled = REFERENCE_STEPS_PER_TRAJECTORY * half_width / REFERENCE_HALF_WIDTH
    return int(round(scaled / 50.0) * 50)


def training_schedule(half_width: float) -> dict[str, object]:
    """The DAgger schedule shared by every generated config."""
    rounds = DAGGER_ITERATIONS
    return {
        "dagger_iterations": rounds,
        "trajectories_per_iteration": [TRAJECTORIES_PER_ROUND] * rounds,
        "steps_per_trajectory": steps_per_trajectory(half_width),
        "target_epochs_per_round": [40] * rounds,
        "action_noise_std": 0.03,
        "expert_mix_beta_start": 0.5,
        "expert_mix_beta_decay_rate": 0.25,
        "expert_mix_decay_after_eval_success": 0.5,
        "eval_episodes": 20,
    }

RING_FRACTION = 1.0 / 3.0

NOMINAL_RADIUS = 3.0
RADIUS_RANGE = (2.0, 4.0)
HEADING_OFFSET_RANGE = (0.3, 1.4)
ELLIPSE_SQUASH_RANGE = (0.5, 0.85)

MIN_SEPARATION_PER_D_SAFE = 1.25
MAX_JITTER_ATTEMPTS = 200
MIN_TRAVEL_DISTANCE = 1.0


JITTER_SEED = 20240909
GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def spread(wave: int, low: float, high: float, offset: float) -> float:
    """A low-discrepancy point in [low, high) for wave `wave`."""
    return low + (high - low) * math.fmod(offset + wave * GOLDEN_RATIO_CONJUGATE, 1.0)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def min_pairwise_distance(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return math.inf
    return min(
        math.dist(points[i], points[j])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    )


def fit_to_box(points: list[tuple[float, float]], max_radius: float) -> list[tuple[float, float]]:
    """Shrink a layout about the origin until it fits inside the fleet's goal box."""
    reach = max(math.hypot(x, y) for x, y in points)
    if reach <= max_radius:
        return points
    scale = max_radius / reach
    return [(x * scale, y * scale) for x, y in points]


def ellipse_points(
    angles: list[float],
    radius: float,
    squash: float,
    minimum: float,
) -> list[tuple[float, float]]:
    """Squash a ring vertically, relaxing toward a circle until the starts clear `minimum`."""
    while squash < 1.0:
        points = [(radius * math.cos(a), squash * radius * math.sin(a)) for a in angles]
        if min_pairwise_distance(points) >= minimum:
            return points
        squash += 0.01
    raise SystemExit(
        f"No ellipse of {len(angles)} robots at radius {radius} clears {minimum:.2f}; "
        "raise the radius range or drop the variant for this fleet size."
    )


def jittered_ring(
    angles: list[float],
    radius: float,
    max_radius: float,
    minimum: float,
    rng: random.Random,
) -> list[tuple[float, float]]:
    """Perturb a ring, redrawing until the starting formation clears `minimum`."""
    spacing = 2.0 * math.pi / len(angles)
    angular = min(0.4 * spacing, 0.5)
    for _ in range(MAX_JITTER_ATTEMPTS):
        points = fit_to_box(
            [
                (
                    radius * (1.0 + rng.uniform(-0.25, 0.25)) * math.cos(angle + rng.uniform(-angular, angular)),
                    radius * (1.0 + rng.uniform(-0.25, 0.25)) * math.sin(angle + rng.uniform(-angular, angular)),
                )
                for angle in angles
            ],
            max_radius,
        )
        if min_pairwise_distance(points) >= minimum:
            return points
    raise SystemExit(
        f"No jittered ring of {len(angles)} robots at radius {radius} cleared "
        f"{minimum:.2f} in {MAX_JITTER_ATTEMPTS} draws; loosen the jitter or raise the radius."
    )


def layout_kinds(num_robots: int) -> list[str]:
    if num_robots == 2:
        return ["nominal", "heading", "jitter", "heading_neg", "heading_tangent"]
    return ["nominal", "heading", "jitter", "heading_neg", "ellipse", "heading_tangent", "skew"]


def variant_sequence(num_robots: int, count: int) -> list[tuple[str, int]]:
    kinds = layout_kinds(num_robots)
    sequence: list[tuple[str, int]] = []
    wave = 0
    while len(sequence) < count:
        for kind in kinds:
            sequence.append((kind, wave))
            if len(sequence) == count:
                break
        wave += 1
    return sequence


def build_layout(
    kind: str,
    num_robots: int,
    wave: int,
    max_radius: float,
    d_safe: float,
    rng: random.Random,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[float]]:
    """Return (start positions, goal positions, start headings) for one rollout."""
    radius = (
        min(NOMINAL_RADIUS, max_radius)
        if wave == 0
        else round(spread(wave, RADIUS_RANGE[0], max_radius, offset=0.0), 3)
    )
    angles = [2.0 * math.pi * idx / num_robots for idx in range(num_robots)]

    minimum = MIN_SEPARATION_PER_D_SAFE * d_safe
    if kind == "jitter":
        # Exact symmetry can deadlock a decentralized policy; jitter breaks it.
        points = jittered_ring(angles, radius, max_radius, minimum, rng)
    elif kind == "ellipse":
        points = ellipse_points(
            angles, radius, spread(wave, *ELLIPSE_SQUASH_RANGE, offset=0.11), minimum
        )
    else:
        points = [(radius * math.cos(angle), radius * math.sin(angle)) for angle in angles]

    if kind == "skew":
        shift = num_robots // 2 + 1
        goals = [points[(idx + shift) % num_robots] for idx in range(num_robots)]
    else:
        goals = [(-x, -y) for x, y in points]

    closest = min_pairwise_distance(points)
    if closest < minimum:
        raise SystemExit(
            f"Layout '{kind}' at radius {radius} puts {num_robots} robots {closest:.3f} apart, "
            f"under the {minimum:.2f} floor."
        )

    if kind == "heading":
        heading_offset = spread(wave, *HEADING_OFFSET_RANGE, offset=0.37)
    elif kind == "heading_neg":
        heading_offset = -spread(wave, *HEADING_OFFSET_RANGE, offset=0.37)
    elif kind == "heading_tangent":
        heading_offset = math.pi / 2.0 if wave % 2 == 0 else -math.pi / 2.0
    else:
        heading_offset = 0.0

    headings = []
    for start, goal in zip(points, goals):
        if math.dist(start, goal) < MIN_TRAVEL_DISTANCE:
            raise SystemExit(f"Layout '{kind}' produced a robot that barely has to move.")
        travel = math.atan2(goal[1] - start[1], goal[0] - start[0])
        headings.append(wrap_angle(travel + heading_offset))
    return points, goals, headings


def build_rollouts(num_robots: int, count: int, max_radius: float, d_safe: float):
    rng = random.Random(JITTER_SEED + num_robots)
    initial_states: list[list[list[float]]] = []
    goal_states: list[list[list[float]]] = []
    for kind, wave in variant_sequence(num_robots, count):
        points, goals, headings = build_layout(kind, num_robots, wave, max_radius, d_safe, rng)
        initial_states.append(
            [[x, y, heading, 0.0, 0.0] for (x, y), heading in zip(points, headings)]
        )
        goal_states.append(
            [
                [gx, gy, math.atan2(gy - sy, gx - sx)]
                for (sx, sy), (gx, gy) in zip(points, goals)
            ]
        )
    return initial_states, goal_states


def validate(num_robots: int, initial_states, goal_states, d_safe: float) -> float:
    """Push the layouts through the same normalizers train_dagger.py uses."""
    fleet_path = PROJECT_ROOT / (FLEET_CONFIG_TEMPLATE % num_robots)
    validated = load_and_validate_system_config("multi_robot", fleet_path)
    simulator = DynamicsFactory.create(system_name="multi_robot", config=validated)
    states = normalize_initial_state_specs(simulator, initial_states)
    goals = normalize_goal_state_specs(simulator, goal_states)
    if len(states) != len(initial_states) or len(goals) != len(goal_states):
        raise SystemExit("Normalizer collapsed the rollout list; check the nesting.")
    closest = math.inf
    for state in states:
        positions = [
            (float(state[state_slice.start]), float(state[state_slice.start + 1]))
            for state_slice in simulator.robot_state_slices
        ]
        closest = min(closest, min_pairwise_distance(positions))
    if closest < d_safe:
        raise SystemExit(f"Fleet {num_robots}: starting pair at {closest:.3f} < d_safe {d_safe}.")
    return closest


def format_number(value: float) -> str:
    # `+ 0.0` turns -0.0 into 0.0.
    text = f"{round(float(value), 4) + 0.0:.4f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def format_rollouts(key: str, rollouts: list[list[list[float]]]) -> str:
    lines = [f"  {key}:"]
    for rollout in rollouts:
        robots = ", ".join(
            "[" + ", ".join(format_number(value) for value in robot) + "]" for robot in rollout
        )
        lines.append(f"    - [{robots}]")
    return "\n".join(lines) + "\n"


TEMPLATE_ONLY_MARKER = "[template-only]"


def template_body(text: str, prediction_horizon: int, horizon_note: str) -> str:
    """A template's content as it should appear in a generated config."""
    lines = []
    horizon_written = False
    for line in text.splitlines():
        if line.startswith("training:"):
            break
        if TEMPLATE_ONLY_MARKER in line:
            continue
        if line.strip().startswith("prediction_horizon:"):
            lines.append(f"  prediction_horizon: {prediction_horizon}        # {horizon_note}")
            horizon_written = True
            continue
        lines.append(line)
    if not horizon_written:
        raise ValueError("Template has no 'prediction_horizon' line to rewrite.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    count = round(TRAJECTORIES_PER_ROUND * RING_FRACTION)
    for num_robots in FLEET_SIZES:
        fleet_path = PROJECT_ROOT / (FLEET_CONFIG_TEMPLATE % num_robots)
        fleet_config = yaml.safe_load(fleet_path.read_text())
        d_safe = float(fleet_config["d_safe"])
        goal_box = float(fleet_config["robots"][0]["config"]["workspace_bounds"][1])
        max_radius = min(RADIUS_RANGE[1], goal_box)

        initial_states, goal_states = build_rollouts(
            num_robots, count, max_radius, d_safe
        )
        validate(num_robots, initial_states, goal_states, d_safe)

        out_dir = CONFIG_DIR / f"n{num_robots:02d}"
        out_dir.mkdir(exist_ok=True)
        # Stale YAMLs would otherwise be trained along with the new ones.
        for stale in out_dir.glob("*.yaml"):
            stale.unlink()

        for encoder in ENCODERS:
            for head, (prediction_horizon, horizon_note) in HEADS.items():
                template_path = CONFIG_DIR / f"{encoder}_{head}_config.yaml"
                out_path = out_dir / f"{encoder}_{head}.yaml"
                text = (
                    f"# N={num_robots} {encoder} {head}, {count} ring rollouts per round.\n"
                    f"# Generated by learning/config/study/generate_study_policy_configs.py "
                    f"-- do not edit by hand.\n"
                    + template_body(template_path.read_text(), prediction_horizon, horizon_note)
                    + "\ntraining:\n"
                    + format_schedule(training_schedule(goal_box))
                    + format_rollouts("initial_states", initial_states)
                    + format_rollouts("goal_states", goal_states)
                )
                out_path.write_text(text)
                print(f"wrote {out_path.relative_to(PROJECT_ROOT)}  ({count} ring rollouts)")


def format_training_value(value: object) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {item}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def format_schedule(schedule: dict[str, object]) -> str:
    return "".join(
        f"  {key}: {format_training_value(value)}\n" for key, value in schedule.items()
    )


if __name__ == "__main__":
    main()
