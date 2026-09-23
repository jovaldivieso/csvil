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
``./train.sh <experiment> learning/config/study/n<NN> <expert config>``.

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

# Each config carries its full DAgger schedule in its 'training' section. train.sh
# passes only the run identity (--experiment-name/--system/--expert-config/
# --policy-config/--seed/--checkpoint-dir): train_dagger.py lets command-line flags
# override that section, so the schedule has exactly one source -- the constants below.
#
# Policy heads and their prediction horizons, each head at its natural horizon. The
# horizon lives here rather than in the templates so a template cannot silently
# disagree with the file it is generated into.
HEADS = {
    "mlp": (1, "single-step regression"),
    "flow": (10, "action chunk; the generative head's intended setting"),
}
FLEET_SIZES = (2, 4, 6, 8)

# Episodes collected per DAgger round, the same for every fleet size. Also sizes each
# config's ring-layout list (RING_FRACTION of the round).
TRAJECTORIES_PER_ROUND = 200
DAGGER_ITERATIONS = 5

# Episode length, from learning/config/multi_unicycle2_casadi_flow_config.yaml: 250 steps
# for the +-3 workspace of a 2-robot fleet. Larger fleets get a proportionally wider
# workspace (see test/config/generate_fleet_configs.py), so their robots have further to
# drive; scaling the budget with the workspace keeps the time to reach a goal comparable
# across fleet sizes. With a fixed 200 the expert itself finishes almost no 8-robot
# episode, the in-training eval then never clears the beta-decay gate, and large fleets
# would see far fewer arrivals than small ones.
REFERENCE_STEPS_PER_TRAJECTORY = 250
REFERENCE_HALF_WIDTH = 3.0


def steps_per_trajectory(half_width: float) -> int:
    """Episode budget for a fleet whose workspace is +-half_width, rounded to 50."""
    scaled = REFERENCE_STEPS_PER_TRAJECTORY * half_width / REFERENCE_HALF_WIDTH
    return int(round(scaled / 50.0) * 50)


def training_schedule(half_width: float) -> dict[str, object]:
    """The DAgger schedule shared by every generated config.

    Rounds, trajectories per round and epochs follow
    learning/config/multi_unicycle2_casadi_flow_config.yaml. There is deliberately no
    max_train_steps: every round trains for target_epochs_per_round epochs over the
    whole aggregated dataset, so a fleet that collects more frames also trains longer.
    """
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

# Share of each round spent on the ring layouts. A third leaves the majority of
# the data on the randomized-goal distribution the policies are evaluated on
# first, while still exposing every round to the hard crossing case.
RING_FRACTION = 1.0 / 3.0

# 3.0 is exactly what test/config/generate_circle_configs.py emits for fleets of
# 2-8 robots, so the first rollout of every generated file reproduces the circle
# evaluation layout the policies are scored on.
NOMINAL_RADIUS = 3.0
# Remaining radii are drawn from this range, whose upper end is lowered per fleet
# to the workspace box (``workspace_bounds``) so the ring stays inside the region
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

# Convergence tolerances deliberately live in the *scenario* config
# (TASK_TOLERANCES in test/config/generate_fleet_configs.py), not here. They define
# what counts as success, so they must be identical for every policy in a
# comparison and identical between training and evaluation. A `tolerance_overrides`
# block in a policy config is per-policy and is applied by train_dagger.py but not
# by test/evaluate_scaling.py, which is exactly how the earlier runs ended up
# trained at theta_tol 0.78 and scored at 1.1.

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


def layout_kinds(num_robots: int, antipodal_only: bool = False) -> list[str]:
    # Ordered so that the first K entries of the sequence cover every kind before
    # a wave repeats: rollout 0 is the plain circle-eval layout, and a fleet whose
    # rounds are short still sees turning, asymmetry and skewed goals.
    #
    # 'skew' is the one kind that moves the goal off the antipode (one seat further
    # round), so its robots no longer meet head-on through the centre. With
    # antipodal_only every robot crosses to the opposite side, which is the conflict
    # the ring layouts exist for -- at 4 robots the skewed goal is a quarter turn
    # away and barely forces an interaction at all.
    if num_robots == 2:
        # 'ellipse' and 'skew' both degenerate at 2 robots: the ring lies on the
        # x-axis, so squashing y changes nothing, and the goal one seat past the
        # antipode is the robot's own start.
        return ["nominal", "heading", "jitter", "heading_neg", "heading_tangent"]
    kinds = ["nominal", "heading", "jitter", "heading_neg", "ellipse", "heading_tangent"]
    return kinds if antipodal_only else kinds + ["skew"]


def variant_sequence(num_robots: int, count: int, antipodal_only: bool = False) -> list[tuple[str, int]]:
    kinds = layout_kinds(num_robots, antipodal_only)
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


def build_rollouts(
    num_robots: int,
    count: int,
    max_radius: float,
    d_safe: float,
    antipodal_only: bool = False,
):
    rng = random.Random(JITTER_SEED + num_robots)
    initial_states: list[list[list[float]]] = []
    goal_states: list[list[list[float]]] = []
    kind_counts: dict[str, int] = {}
    for kind, wave in variant_sequence(num_robots, count, antipodal_only):
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


def template_body(text: str, prediction_horizon: int, horizon_note: str) -> str:
    """A template's content as it should appear in a generated config.

    Drops the template-only commentary and any 'training' block: this script owns
    that section, so one left in a template would silently be ignored. Rewrites
    'prediction_horizon' to the variant's value, since the variant label -- not the
    template -- is what defines the horizon.
    """
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

        # train.sh trains every YAML in the directory, so a file left over from a
        # removed template or head would silently keep being trained.
        out_dir = CONFIG_DIR / f"n{num_robots:02d}"
        out_dir.mkdir(exist_ok=True)
        for stale in out_dir.glob("*.yaml"):
            stale.unlink()

        for encoder in ENCODERS:
            for head, (prediction_horizon, horizon_note) in HEADS.items():
                template_path = CONFIG_DIR / f"{encoder}_{head}_config.yaml"
                out_path = out_dir / f"{encoder}_{head}.yaml"
                text = (
                    f"# {encoder} {head} policy for the {num_robots}-robot cell of the study.\n"
                    f"# Generated by learning/config/study/generate_study_policy_configs.py from\n"
                    f"# {template_path.name} -- do not edit by hand; edit the template or the script.\n"
                    f"#\n"
                    f"# {count} of the {TRAJECTORIES_PER_ROUND} episodes per DAgger round "
                    f"({RING_FRACTION:.0%}) start from the ring layouts\n"
                    f"# below; the rest fall back to the expert config's randomized goals.\n"
                    f"# {steps_per_trajectory(goal_box)} steps per episode, scaled from "
                    f"{REFERENCE_STEPS_PER_TRAJECTORY} at +-{REFERENCE_HALF_WIDTH} to this "
                    f"fleet's +-{goal_box} workspace.\n"
                    f"# Radii {NOMINAL_RADIUS} (rollout 0, the circle-eval layout) then "
                    f"{RADIUS_RANGE[0]}-{max_radius}; closest\n"
                    f"# starting pair {closest:.3f} (d_safe={d_safe}).\n"
                    + layout_summary + "\n"
                    + template_body(template_path.read_text(), prediction_horizon, horizon_note)
                    + "\ntraining:\n"
                    + format_schedule(training_schedule(goal_box))
                    + "  # No tolerance_overrides: convergence tolerances belong to the scenario\n"
                    + "  # config, so training and evaluation share one definition of success.\n"
                    + "  # Antipodal-ring rollouts, the training-time counterpart of the 'circle'\n"
                    + f"  # evaluation scenario. Per-robot, so this file only fits {num_robots} robots.\n"
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


SCHEDULE_NOTE = (
    "  # DAgger schedule. train.sh passes only the run identity, so these values are\n"
    "  # the single source; a flag given to train_dagger.py by hand overrides them.\n"
)


def format_schedule(schedule: dict[str, object]) -> str:
    return SCHEDULE_NOTE + "".join(
        f"  {key}: {format_training_value(value)}\n" for key, value in schedule.items()
    )


if __name__ == "__main__":
    main()
