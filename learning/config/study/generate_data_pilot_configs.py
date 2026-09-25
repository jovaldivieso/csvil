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

Every fleet size gets the same rounds, episodes and epochs, so the policies differ only in
the fleet they were trained on. Note that this equalizes *episodes*, not frames: one
episode yields one LeRobot episode per robot, so an 8-robot round holds four times the
frames of a 2-robot one, and at fixed epochs it also takes four times the optimizer steps.
``--max-train-steps`` caps those steps instead (utils.resolve_round_steps takes the
minimum), which equalizes optimization but then makes the epochs differ between fleet
sizes -- pick whichever of the two should be comparable.

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
# budget. Measured with test/plot_expert_trajectories.py at this density: the expert needs
# 65-78% of the budget. Later rounds run on the policy, which needs more steps than the
# expert and more often reaches the limit, so treat an estimate as a lower bound.
FINISH_FRACTION = 0.75
# A denser workspace means more neighbour pairs close enough to matter, and the MPC grows
# a little slower per solve. Measured only at 1x, so this is a margin, not a measurement.
DENSITY_SOLVE_PENALTY = 1.25

# Epochs over the aggregated dataset per round, as in the study schedule.
TARGET_EPOCHS = 40


def episode_seconds(num_robots: int, steps: int) -> float:
    return steps * EXPERT_SECONDS_PER_STEP[num_robots] * FINISH_FRACTION * DENSITY_SOLVE_PENALTY


def trajectories_for_budget(num_robots: int, steps: int, hours: float, rounds: int) -> int:
    """Episodes per round that fit the budget, rounded down to a multiple of 5."""
    affordable = hours * 3600.0 / (rounds * episode_seconds(num_robots, steps))
    return max(5, int(affordable // 5) * 5)


def trajectories_for_frames(num_robots: int, steps: int, frames: int, minimum: int) -> int:
    """Episodes per round holding the dataset size roughly equal across fleet sizes.

    One episode yields one dataset episode per robot, so it is worth `steps * num_robots`
    frames: a large fleet reaches the same dataset with far fewer episodes, and far less
    expert time. The floor is there because those frames are not equally informative --
    the per-robot trajectories of one episode all come from the same scene, so cutting a
    fleet to a handful of episodes buys equal frames at the price of scenario variety.
    """
    return max(minimum, round(frames / (steps * num_robots) / 5) * 5)


def pilot_schedule(
    half_width: float,
    steps: int,
    rounds: int,
    trajectories: int,
    epochs: float,
    max_train_steps: int | None,
    eval_episodes: int,
) -> dict[str, object]:
    schedule = {
        "dagger_iterations": rounds,
        "trajectories_per_iteration": [trajectories] * rounds,
        "steps_per_trajectory": steps,
        "target_epochs_per_round": [epochs] * rounds,
        "action_noise_std": 0.03,
        "expert_mix_beta_start": 0.5,
        "expert_mix_beta_decay_rate": 0.25,
        "expert_mix_decay_after_eval_success": 0.5,
        # Backtrack recovery: when an episode gets stuck, replay it 75% expert-driven
        # rather than purely expert, escalating by 0.25 per failed attempt. At the
        # default 1.0 the gradual path is skipped (see rollouts.py), so the recovered
        # frames would all be pure-expert ones.
        "expert_mix_beta_recovery": 0.75,
        "expert_mix_beta_recovery_increment": 0.25,
        "eval_episodes": eval_episodes,
        "workspace_bounds": [-half_width, half_width],
    }
    if max_train_steps is not None:
        schedule["max_train_steps"] = max_train_steps
    return schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="directory prefix, e.g. data_small")
    parser.add_argument("--hours", type=float, default=None,
                        help="wall-clock budget per run; sizes --trajectories when that is unset")
    parser.add_argument("--rounds", type=int, default=3, help="DAgger rounds (default 3)")
    parser.add_argument(
        "--trajectories", type=int, default=None,
        help="episodes per round for every fleet size; without it, sized from --hours. "
             "A fixed count gives every policy the same amount of data, which is what makes "
             "the fleet sizes comparable -- the runs then differ in length.",
    )
    parser.add_argument("--fleets", type=int, nargs="+", default=[4, 6],
                        choices=sorted(EXPERT_SECONDS_PER_STEP))
    parser.add_argument("--encoders", nargs="+", default=["deepset"], choices=list(_study.ENCODERS))
    parser.add_argument("--head", default="mlp", choices=sorted(_study.HEADS))
    parser.add_argument(
        "--frames-per-round", type=int, default=None,
        help="size the episodes so every fleet size collects about this many frames "
             "per round (dataset-equal instead of episode-equal)",
    )
    parser.add_argument(
        "--ring-fraction", type=float, default=None,
        help=f"share of each round that starts from an antipodal ring layout "
             f"(default {_study.RING_FRACTION:.2f}, the study's). The rest falls back to "
             f"the expert config's randomized goals",
    )
    parser.add_argument(
        "--eval-episodes", default="20",
        help="episodes of the in-training evaluation per round, or 'match' for one "
             "round's worth. That evaluation gates the expert-share decay "
             "(expert_mix_decay_after_eval_success), so at 20 episodes the gate sits "
             "inside its own noise: +-0.11 around a threshold of 0.5. It also draws its "
             "first min(rings, episodes) episodes from the ring layouts, so the count "
             "sets the mix too, and 'match' is the count that reproduces the training "
             "mix exactly",
    )
    parser.add_argument("--min-episodes", type=int, default=50,
                        help="floor for --frames-per-round (default 50), so a large fleet "
                             "keeps some scenario variety")
    parser.add_argument("--epochs", type=float, default=TARGET_EPOCHS,
                        help=f"epochs over the aggregated set per round (default {TARGET_EPOCHS})")
    parser.add_argument("--max-train-steps", type=int, default=None,
                        help="cap the optimizer steps per round; equalizes optimization across "
                             "fleet sizes at the cost of unequal epochs")
    parser.add_argument("--density-factor", type=float, default=3.0,
                        help="times the study's training density (3 = 0.167 robots/m^2)")
    args = parser.parse_args()
    chosen = [args.hours, args.trajectories, args.frames_per_round]
    if sum(value is not None for value in chosen) != 1:
        parser.error(
            "pass exactly one of --hours (budget), --trajectories (fixed episodes) or "
            "--frames-per-round (fixed dataset size)."
        )
    return args


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
        if args.trajectories is not None:
            trajectories = args.trajectories
        elif args.frames_per_round is not None:
            trajectories = trajectories_for_frames(
                num_robots, steps, args.frames_per_round, args.min_episodes)
        else:
            trajectories = trajectories_for_budget(num_robots, steps, args.hours, args.rounds)
        frames = trajectories * steps * num_robots
        estimate = args.rounds * trajectories * episode_seconds(num_robots, steps) / 3600.0

        # Rings must stay inside this box: a layout the randomized episodes can never
        # produce would be a different task, not a harder one.
        max_radius = min(_study.RADIUS_RANGE[1], half_width)
        ring_fraction = args.ring_fraction if args.ring_fraction is not None else _study.RING_FRACTION
        count = round(trajectories * ring_fraction)
        # The evaluation takes its first min(rings, episodes) episodes from the ring list
        # and samples the rest, so one round's worth of episodes reproduces the training
        # mix: rings/trajectories = ring_fraction either way.
        eval_episodes = trajectories if args.eval_episodes == "match" else int(args.eval_episodes)
        # Antipodal goals only: a ring is here for the head-on conflict through the
        # centre, and the 'skew' layout's goal one seat further round does not force one.
        initial_states, goal_states, kind_counts = _study.build_rollouts(
            num_robots, count, max_radius, d_safe, antipodal_only=True
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
                f"# {args.rounds} rounds x {trajectories} episodes x {args.epochs:g} epochs; "
                f"{frames} frames per round\n"
                f"# ({trajectories} episodes x {steps} steps x {num_robots} robots, one dataset "
                f"episode per robot). Estimate {estimate:.1f} h per run.\n"
                + _study.template_body(template_path.read_text(), horizon, horizon_note)
                + "\ntraining:\n"
                + _study.format_schedule(pilot_schedule(
                    half_width, steps, args.rounds, trajectories,
                    args.epochs, args.max_train_steps, eval_episodes))
                + f"  # {count} of the {trajectories} episodes per round ({ring_fraction:.0%}) "
                f"start from ring\n"
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
                f"{args.rounds}x{trajectories} episodes, {frames} frames/round, "
                f"{ring_fraction:.0%} rings, ~{estimate:.1f} h)"
            )


if __name__ == "__main__":
    main()
