"""Generate time-boxed data pilots: how much data does a usable policy need?

The study schedule (5 rounds x 200 episodes) costs 27 h at N=4 and 124 h at N=8 per run,
almost all of it in the CasADi expert. This generator writes the same task with less data,
sized to a wall-clock budget, so the question "is less data enough?" can be answered in
hours instead of days.

Two things make a run cheap, and they are separate:

* **Density.** Episodes are shorter in a smaller workspace. The pilot places the fleet at
  ``--density-factor`` times the study's training density through
  ``training.workspace_bounds`` (train_dagger.py applies it to collection and in-loop
  evaluation), and scales ``steps_per_trajectory`` with the box the same way
  generate_study_policy_configs.py does. This changes the *task*: a policy trained here
  meets more neighbours than one trained at the study density, so compare pilots with
  pilots, and use test/evaluate_scaling.py for anything else.
* **Data.** Episodes per round follow from the budget: the expert dominates the cost, and
  its per-step solve time is known per fleet size (measured with test/evaluate_policy.py).

Gradient steps are deliberately held fixed (``max_train_steps``) instead of following the
data. With 40 epochs over the aggregated set, a tenth of the data would mean a tenth of
the optimizer steps, and a weak policy could not be told apart from an undertrained one --
which is the whole question here.

Usage:
    python learning/config/study/generate_data_pilot_configs.py \
        --name data_small --hours 3 --fleets 4 6 --encoders deepset
"""

from __future__ import annotations

import argparse
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

# Wall-clock model. The CasADi expert is queried at every step to label the state, so it
# sets the cost; the policy's own forward pass and the supervised training are minor.
# Seconds per step at the study's training density, measured with test/evaluate_policy.py
# (centralized solve for the whole fleet).
EXPERT_SECONDS_PER_STEP = {2: 0.157, 4: 0.463, 6: 0.835, 8: 1.483}
# Episodes end as soon as every robot is at its goal, so a round does not spend the full
# budget. From the expert rollouts: it reaches the goals in roughly 60% of the budget.
FINISH_FRACTION = 0.6
# A denser workspace means more neighbour pairs close enough to matter, and the MPC grows
# a little slower per solve. Measured only at 1x, so this is a margin, not a measurement.
DENSITY_SOLVE_PENALTY = 1.25

# Gradient steps per round, from the study-2 grid. Fixed here so that only the amount of
# data varies between pilot and study; target_epochs is set high so the cap is what binds.
MAX_TRAIN_STEPS = 40000
TARGET_EPOCHS = 400


def episode_seconds(num_robots: int, steps: int) -> float:
    return steps * EXPERT_SECONDS_PER_STEP[num_robots] * FINISH_FRACTION * DENSITY_SOLVE_PENALTY


def trajectories_for_budget(num_robots: int, steps: int, hours: float, rounds: int) -> int:
    """Episodes per round that fit the budget, rounded down to a multiple of 5."""
    affordable = hours * 3600.0 / (rounds * episode_seconds(num_robots, steps))
    return max(5, int(affordable // 5) * 5)


def pilot_schedule(half_width: float, steps: int, rounds: int, trajectories: int) -> dict[str, object]:
    return {
        "dagger_iterations": rounds,
        "trajectories_per_iteration": [trajectories] * rounds,
        "steps_per_trajectory": steps,
        "target_epochs_per_round": [TARGET_EPOCHS] * rounds,
        "max_train_steps": MAX_TRAIN_STEPS,
        "action_noise_std": 0.03,
        "expert_mix_beta_start": 0.5,
        "expert_mix_beta_decay_rate": 0.25,
        "expert_mix_decay_after_eval_success": 0.5,
        "eval_episodes": 20,
        "workspace_bounds": [-half_width, half_width],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="directory prefix, e.g. data_small")
    parser.add_argument("--hours", type=float, required=True, help="wall-clock budget per run")
    parser.add_argument("--rounds", type=int, default=3, help="DAgger rounds (default 3)")
    parser.add_argument("--fleets", type=int, nargs="+", default=[4, 6],
                        choices=sorted(EXPERT_SECONDS_PER_STEP))
    parser.add_argument("--encoders", nargs="+", default=["deepset"], choices=list(_study.ENCODERS))
    parser.add_argument("--head", default="mlp", choices=sorted(_study.HEADS))
    parser.add_argument("--density-factor", type=float, default=3.0,
                        help="times the study's training density (3 = 0.167 robots/m^2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizon, horizon_note = _study.HEADS[args.head]

    for num_robots in args.fleets:
        fleet_path = PROJECT_ROOT / (_study.FLEET_CONFIG_TEMPLATE % num_robots)
        fleet_config = yaml.safe_load(fleet_path.read_text())
        d_safe = float(fleet_config["d_safe"])
        study_half_width = float(fleet_config["robots"][0]["config"]["workspace_bounds"][1])

        half_width = round(study_half_width / math.sqrt(args.density_factor), 3)
        density = num_robots / (2.0 * half_width) ** 2
        steps = _study.steps_per_trajectory(half_width)
        trajectories = trajectories_for_budget(num_robots, steps, args.hours, args.rounds)
        estimate = args.rounds * trajectories * episode_seconds(num_robots, steps) / 3600.0

        # Rings must stay inside this box: a layout the randomized episodes can never
        # produce would be a different task, not a harder one.
        max_radius = min(_study.RADIUS_RANGE[1], half_width)
        count = round(trajectories * _study.RING_FRACTION)
        initial_states, goal_states, kind_counts = _study.build_rollouts(
            num_robots, count, max_radius, d_safe
        )
        closest = _study.validate(num_robots, initial_states, goal_states, d_safe)
        layout_summary = textwrap.fill(
            "Layouts: " + ", ".join(f"{kind} x{n}" for kind, n in sorted(kind_counts.items())) + ".",
            width=86, initial_indent="  # ", subsequent_indent="  # ",
        )

        # train.sh trains every YAML in the directory, so a file left over from another
        # encoder or head would silently be trained too.
        out_dir = CONFIG_DIR / f"{args.name}_n{num_robots:02d}"
        out_dir.mkdir(exist_ok=True)
        for stale in out_dir.glob("*.yaml"):
            stale.unlink()

        for encoder in args.encoders:
            template_path = CONFIG_DIR / f"{encoder}_{args.head}_config.yaml"
            out_path = out_dir / f"{encoder}_{args.head}.yaml"
            text = (
                f"# N={num_robots} {encoder} {args.head} data pilot '{args.name}': "
                f"{args.rounds} rounds x {trajectories} episodes.\n"
                f"# Generated by learning/config/study/generate_data_pilot_configs.py "
                f"-- do not edit by hand.\n"
                f"#\n"
                f"# Trained against test/config/study/unicycle2_fleet_{num_robots:02d}.yaml, but with\n"
                f"# starts and goals drawn from +-{half_width} instead of the config's "
                f"+-{study_half_width}:\n"
                f"# {density:.4f} robots/m^2, {args.density_factor:g}x the study's training density.\n"
                f"# {steps} steps per episode, scaled from {_study.REFERENCE_STEPS_PER_TRAJECTORY} "
                f"at +-{_study.REFERENCE_HALF_WIDTH} to this box.\n"
                f"# Episodes per round sized for ~{args.hours:g} h per run "
                f"(estimate {estimate:.1f} h); gradient steps are\n"
                f"# capped at {MAX_TRAIN_STEPS} per round, so only the amount of data varies.\n"
                + _study.template_body(template_path.read_text(), horizon, horizon_note)
                + "\ntraining:\n"
                + _study.format_schedule(
                    pilot_schedule(half_width, steps, args.rounds, trajectories))
                + f"  # {count} of the {trajectories} episodes per round start from ring\n"
                f"  # layouts of radius {_study.RADIUS_RANGE[0]}-{max_radius}; closest starting pair\n"
                f"  # {closest:.3f} (d_safe={d_safe}).\n"
                + layout_summary + "\n"
                + _study.format_rollouts("initial_states", initial_states)
                + _study.format_rollouts("goal_states", goal_states)
            )
            out_path.write_text(text)
            print(
                f"wrote {out_path.relative_to(PROJECT_ROOT)} "
                f"(+-{half_width}, {density:.3f} robots/m^2, {steps} steps, "
                f"{args.rounds}x{trajectories} episodes, ~{estimate:.1f} h)"
            )


if __name__ == "__main__":
    main()
