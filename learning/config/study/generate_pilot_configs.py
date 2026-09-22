"""Generate the N=4 flow pilot configs: which training settings give a usable 4-robot policy?

The 4-robot flow runs so far were weak (in-training eval success 0.13-0.33). The leading
suspicion is the training *density*: a 4-robot fleet in the study's +-4.243 box meets
neighbours rarely, so most demonstrations show driving to a goal rather than resolving a
conflict. Each variant below changes one thing against the baseline, so the comparison
says which setting is responsible.

Density is set through ``training.workspace_bounds``, which train_dagger.py applies to
data collection and to the in-training evaluation (dagger_trainer._apply_runtime_config
_overrides). The expert config is the unmodified 4-robot one, so the *task* is identical
in every variant -- only the region starts and goals are drawn from changes. The ring
layouts are capped to the same box, otherwise a variant would train on rings its random
episodes never reach.

Because the in-training eval also moves with the box, its success rates are NOT
comparable across variants. Compare the variants with test/evaluate_scaling.py on the
standard scenarios instead.

The schedule is the study's (5 rounds x 200 episodes, 40 epochs), with 300 steps per
episode, so a variant that wins here can be carried over unchanged.

Usage:
    python learning/config/study/generate_pilot_configs.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import textwrap
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "_generate_study_policy_configs",
    Path(__file__).resolve().parent / "generate_study_policy_configs.py",
)
_study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_study)

CONFIG_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CONFIG_DIR / "pilot_n04"
ENCODERS = ("deepset", "transformer", "gnn")
HEAD = "flow"
NUM_ROBOTS = 4

# The study's schedule, with more episodes per round and a slightly shorter episode
# budget. Every grid cell is an encoder x density pair, so the pilot already answers
# whether the density that helps one encoder helps the others too.
PILOT_ROUNDS = 5
PILOT_TRAJECTORIES = 300
PILOT_STEPS = 300

# Density factors, dividing the study's box half width. 1x (the study setting) is left
# out: the runs that motivated this pilot were trained at it. 3x is the densest the goal
# sampler can place (see test/config/generate_density_configs.py).
VARIANTS = {
    "d2": (2.0, "twice the training density"),
    "d3": (3.0, "three times the training density"),
}
# Sum pooling (study 1's setting) instead of the deepset template's mean. Only the
# deepset template has the key; the GNN sums by construction and the transformer
# normalizes through attention, so for those this is a no-op.
MODEL_OVERRIDES = {"pool_type": "sum"}


def apply_model_overrides(body: str, overrides: dict[str, str]) -> str:
    """Rewrite model keys (e.g. pool_type) in a template body, where present."""
    lines = []
    for line in body.splitlines():
        stripped = line.lstrip()
        key = stripped.split(":", 1)[0]
        if key in overrides and ":" in stripped:
            indent = line[: len(line) - len(stripped)]
            lines.append(f"{indent}{key}: {overrides[key]}")
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def pilot_schedule(half_width: float) -> dict[str, object]:
    schedule = {
        "dagger_iterations": PILOT_ROUNDS,
        "trajectories_per_iteration": [PILOT_TRAJECTORIES] * PILOT_ROUNDS,
        "steps_per_trajectory": PILOT_STEPS,
        "target_epochs_per_round": [40] * PILOT_ROUNDS,
        "action_noise_std": 0.03,
        "expert_mix_beta_start": 0.5,
        "expert_mix_beta_decay_rate": 0.25,
        "expert_mix_decay_after_eval_success": 0.5,
        "eval_episodes": 20,
        "workspace_bounds": [-half_width, half_width],
    }
    return schedule


def main() -> None:
    fleet_path = PROJECT_ROOT / (_study.FLEET_CONFIG_TEMPLATE % NUM_ROBOTS)
    fleet_config = yaml.safe_load(fleet_path.read_text())
    d_safe = float(fleet_config["d_safe"])
    study_half_width = float(fleet_config["robots"][0]["config"]["workspace_bounds"][1])
    OUTPUT_DIR.mkdir(exist_ok=True)
    for stale in OUTPUT_DIR.glob("*.yaml"):
        stale.unlink()

    count = round(PILOT_TRAJECTORIES * _study.RING_FRACTION)
    horizon, horizon_note = _study.HEADS[HEAD]
    for label, (factor, note) in VARIANTS.items():
        half_width = round(study_half_width / math.sqrt(factor), 3)
        density = NUM_ROBOTS / (2.0 * half_width) ** 2
        # Rings must stay inside this variant's box: a layout the randomized episodes
        # can never produce would be a different task, not a harder one.
        max_radius = min(_study.RADIUS_RANGE[1], half_width)
        initial_states, goal_states, kind_counts = _study.build_rollouts(
            NUM_ROBOTS, count, max_radius, d_safe
        )
        closest = _study.validate(NUM_ROBOTS, initial_states, goal_states, d_safe)
        layout_summary = textwrap.fill(
            "Layouts: " + ", ".join(f"{kind} x{n}" for kind, n in sorted(kind_counts.items())) + ".",
            width=86, initial_indent="  # ", subsequent_indent="  # ",
        )
        for encoder in ENCODERS:
            template_path = CONFIG_DIR / f"{encoder}_{HEAD}_config.yaml"
            body = apply_model_overrides(
                _study.template_body(template_path.read_text(), horizon, horizon_note),
                MODEL_OVERRIDES,
            )
            out_path = OUTPUT_DIR / f"{encoder}_{label}.yaml"
            text = (
                f"# N=4 {encoder} flow pilot '{label}': {note}.\n"
                f"# Generated by learning/config/study/generate_pilot_configs.py -- do not edit by hand.\n"
                f"#\n"
                f"# Trained against test/config/study/unicycle2_fleet_04.yaml, but with starts and\n"
                f"# goals drawn from +-{half_width} instead of the config's +-{study_half_width}:\n"
                f"# {density:.4f} robots/m^2, {factor:g}x the study's training density.\n"
                f"# Pilot schedule: {PILOT_ROUNDS} rounds x {PILOT_TRAJECTORIES} episodes x "
                f"{PILOT_STEPS} steps, 40 epochs.\n"
                + body
                + "\ntraining:\n"
                + _study.format_schedule(pilot_schedule(half_width))
                + f"  # {count} of the {PILOT_TRAJECTORIES} episodes per round start from ring\n"
                f"  # layouts of radius {_study.RADIUS_RANGE[0]}-{max_radius}; closest starting pair\n"
                f"  # {closest:.3f} (d_safe={d_safe}).\n"
                + layout_summary + "\n"
                + _study.format_rollouts("initial_states", initial_states)
                + _study.format_rollouts("goal_states", goal_states)
            )
            out_path.write_text(text)
            print(
                f"wrote {out_path.relative_to(PROJECT_ROOT)} "
                f"(+-{half_width}, {factor:g}x density, {encoder}, horizon {horizon})"
            )


if __name__ == "__main__":
    main()
