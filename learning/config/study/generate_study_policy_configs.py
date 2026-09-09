"""Generate per-fleet-size policy configs for the encoder-scaling study.

The study trains one policy per (encoder, training fleet size) cell. The three
template files next to this script -- ``deepset_mlp_config.yaml``,
``gnn_mlp_config.yaml`` and ``transformer_mlp_config.yaml`` -- hold the ``model``
section, the only thing that is supposed to differ between encoders. This script
copies each template once per fleet size and appends the ``training`` section,
which is where the fleet size matters:

``initial_states`` / ``goal_states`` pin the opening episodes of every DAgger
round to antipodal-ring layouts -- the training-time counterpart of the ``circle``
evaluation scenario. Those coordinates are per-robot, so a config only fits the
fleet size it was generated for; hence one file per size.

Episodes beyond the provided list fall back to the expert config's randomized
sampling (see ``collect_dagger_rollouts``), so every round is part ring, part
random. The ring share is held constant across fleet sizes (``RING_FRACTION``),
the same way ``run_study.sh`` holds frames-per-round constant -- otherwise the
fleet sizes would no longer be comparable, which is the point of the study.

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
import textwrap
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
FLEET_SIZES = (2, 4, 6, 8)

# Mirrors TRAJECTORIES in run_study.sh: episodes collected per DAgger round.
TRAJECTORIES_PER_ROUND = {2: 150, 4: 100, 6: 75, 8: 50}
# Share of each round spent on the ring layouts. A third leaves the majority of
# the data on the randomized-goal distribution the policies are evaluated on
# first, while still exposing every round to the hard crossing case.
RING_FRACTION = 1.0 / 3.0

# 3.0 is exactly what test/config/generate_circle_configs.py emits for fleets of
# 2-8 robots, so the first rollout of every generated file reproduces the circle
# evaluation layout the policies are scored on.
NOMINAL_RADIUS = 3.0
# Remaining radii are drawn from this range, whose upper end is lowered per fleet
# to the goal box (``goal_position_bounds``) so the ring stays inside the region
# the randomized rollouts sample. Smaller rings crowd the centre; larger ones make
# the crossing longer and the approach faster.
RADIUS_RANGE = (2.0, 4.0)
# Start-heading offsets from the direction of travel: the robot has to turn before
# it can cross, which the pure ring never asks for.
HEADING_OFFSET_RANGE = (0.3, 1.4)
# Vertical squash factors turning the ring into an ellipse, which staggers arrival
# at the centre instead of having every robot reach it at the same instant.
ELLIPSE_SQUASH_RANGE = (0.5, 0.85)

# Minimum pairwise start separation, in units of d_safe, so a perturbed ring can
# never be born in collision. A jittered draw that violates it is redrawn rather
# than scaled up, which would push the ring outside the fleet's goal box.
MIN_SEPARATION_PER_D_SAFE = 1.25
MAX_JITTER_ATTEMPTS = 200
# Minimum distance a robot has to travel, so no rollout starts on its own goal.
MIN_TRAVEL_DISTANCE = 1.0

# Shared with the deepset template's previous training block, and applied to all
# three encoders so the only difference between study cells stays the encoder.
TOLERANCE_OVERRIDES = {
    "pos_tol": 0.1,
    "theta_tol": 0.78,
    "vel_tol": 0.1,
    "omega_tol": 0.1,
}

JITTER_SEED = 20240909
GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def spread(wave: int, low: float, high: float, offset: float) -> float:
    """A low-discrepancy point in [low, high) for wave `wave`.

    Cycling a short table of radii or heading offsets instead would repeat a
    layout as soon as the wave count exceeded the table, and the fleets that
    collect the most ring episodes are exactly the ones that would repeat most.
    Stepping by the golden ratio keeps every wave's value distinct and spread.
    """
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
    """Shrink a layout about the origin until it fits inside the fleet's goal box.

    Explicit goals bypass the box the randomized rollouts sample from, so nothing
    would stop a ring from being written outside it -- but a ring the policy never
    meets under randomized goals is a different task, not a harder one.
    """
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
    """Squash a ring vertically, relaxing toward a circle until the starts clear `minimum`.

    A squash close to the drawn value is what staggers arrival at the centre, but
    squashing pulls the points nearest the x-axis together, and how much room there
    is depends on the fleet size. Relaxing is preferred to widening the ring, which
    would leave the fleet's goal box.
    """
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
    # Ordered so that the first K entries of the sequence cover every kind before
    # a wave repeats: rollout 0 is the plain circle-eval layout, and a fleet whose
    # rounds are short still sees turning, asymmetry and skewed goals.
    if num_robots == 2:
        # 'ellipse' and 'skew' both degenerate at 2 robots: the ring lies on the
        # x-axis, so squashing y changes nothing, and the goal one seat past the
        # antipode is the robot's own start.
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
    # Spread over the fleet's own range rather than clamping a wider one: clamping
    # collapses every wave past the cap onto the same radius, and the small fleets
    # (whose goal box is the tightest and whose rounds are the longest) would then
    # collect the same layout again and again.
    radius = (
        min(NOMINAL_RADIUS, max_radius)
        if wave == 0
        else round(spread(wave, RADIUS_RANGE[0], max_radius, offset=0.0), 3)
    )
    angles = [2.0 * math.pi * idx / num_robots for idx in range(num_robots)]

    minimum = MIN_SEPARATION_PER_D_SAFE * d_safe
    if kind == "jitter":
        # Break the ring's exact symmetry. Identical relative geometry for every
        # robot is the one case where a decentralized policy can deadlock, every
        # robot picking the same avoidance direction at the same moment.
        points = jittered_ring(angles, radius, max_radius, minimum, rng)
    elif kind == "ellipse":
        points = ellipse_points(
            angles, radius, spread(wave, *ELLIPSE_SQUASH_RANGE, offset=0.11), minimum
        )
    else:
        points = [(radius * math.cos(angle), radius * math.sin(angle)) for angle in angles]

    if kind == "skew":
        # Goal one seat past the antipode: every robot still crosses the centre,
        # but the conflicts are no longer symmetric head-on pairs.
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
    kind_counts: dict[str, int] = {}
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
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    return initial_states, goal_states, kind_counts


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
    # `+ 0.0` turns -0.0 into 0.0; the trailing-zero trim keeps one decimal so the
    # values stay unambiguously floats.
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


# Comment lines carrying this marker document the template itself and would be
# false in a generated file, so they are dropped on the way through.
TEMPLATE_ONLY_MARKER = "[template-only]"


def template_body(text: str) -> str:
    """A template's content as it should appear in a generated config.

    Drops the template-only commentary and any 'training' block: this script owns
    that section, so one left in a template would silently be ignored.
    """
    lines = []
    for line in text.splitlines():
        if line.startswith("training:"):
            break
        if TEMPLATE_ONLY_MARKER in line:
            continue
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    for num_robots in FLEET_SIZES:
        fleet_path = PROJECT_ROOT / (FLEET_CONFIG_TEMPLATE % num_robots)
        fleet_config = yaml.safe_load(fleet_path.read_text())
        d_safe = float(fleet_config["d_safe"])
        goal_box = float(fleet_config["robots"][0]["config"]["goal_position_bounds"][1])
        max_radius = min(RADIUS_RANGE[1], goal_box)

        count = round(TRAJECTORIES_PER_ROUND[num_robots] * RING_FRACTION)
        initial_states, goal_states, kind_counts = build_rollouts(
            num_robots, count, max_radius, d_safe
        )
        closest = validate(num_robots, initial_states, goal_states, d_safe)
        layout_summary = textwrap.fill(
            "Layouts: " + ", ".join(f"{kind} x{n}" for kind, n in sorted(kind_counts.items())) + ".",
            width=86,
            initial_indent="# ",
            subsequent_indent="# ",
        )

        for encoder in ENCODERS:
            template_path = CONFIG_DIR / f"{encoder}_mlp_config.yaml"
            out_path = CONFIG_DIR / f"{encoder}_mlp_n{num_robots:02d}_config.yaml"
            text = (
                f"# {encoder} policy for the {num_robots}-robot cell of the encoder-scaling study.\n"
                f"# Generated by learning/config/study/generate_study_policy_configs.py from\n"
                f"# {template_path.name} -- do not edit by hand; edit the template or the script.\n"
                f"#\n"
                f"# {count} of the {TRAJECTORIES_PER_ROUND[num_robots]} episodes per DAgger round "
                f"({RING_FRACTION:.0%}) start from the ring layouts\n"
                f"# below; the rest fall back to the expert config's randomized goals.\n"
                f"# Radii {NOMINAL_RADIUS} (rollout 0, the circle-eval layout) then "
                f"{RADIUS_RANGE[0]}-{max_radius}; closest\n"
                f"# starting pair {closest:.3f} (d_safe={d_safe}).\n"
                + layout_summary + "\n"
                + template_body(template_path.read_text())
                + "\ntraining:\n"
                + "  # unicycle\n"
                + "  tolerance_overrides:\n"
                + "".join(f"    {key}: {value}\n" for key, value in TOLERANCE_OVERRIDES.items())
                + "\n"
                + "  # Antipodal-ring rollouts, the training-time counterpart of the 'circle'\n"
                + f"  # evaluation scenario. Per-robot, so this file only fits {num_robots} robots.\n"
                + format_rollouts("initial_states", initial_states)
                + format_rollouts("goal_states", goal_states)
            )
            out_path.write_text(text)
            print(f"wrote {out_path.relative_to(PROJECT_ROOT)}  ({count} ring rollouts)")


if __name__ == "__main__":
    main()
