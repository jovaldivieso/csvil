"""Run the head-on crash ladder for several policies and write one CSV.

Split out of test/evaluate_scaling.py, which is built around a question this scenario
does not ask. That script sweeps fleet size and density and reports a train-size x
eval-size matrix; the crash ladder fixes the fleet at 2, fixes the encoder, fixes the
layout, and sweeps exactly one quantity -- the speed the robots already carry at t=0.
Running it through the scaling harness meant carrying columns that are constant by
construction and, worse, remembering two env vars that silently produce wrong answers if
forgotten. Both are now derived here instead:

**Step budget.** ``evaluate_scaling.step_budget`` sizes an episode from the start-goal
distance over max speed, which covers the drive across the workspace but not the in-place
rotation every unicycle2 episode ends with -- and that rotation is 113 of the ~200 steps
this task needs. Getting it wrong times out every rung and reads as a policy failure. Here
the budget is computed from the robot's own dynamics, the same traverse-plus-settle model
test/config/generate_small_robot_configs.py sizes training episodes with, so there is no
``STEP_BUDGET_FACTOR=6`` to forget.

**Action noise.** The ladder starts from fixed configured states, so a deterministic MLP
produces one identical episode and the repeats collapse. 0.03 -- what training used -- is
the default here rather than something the caller has to remember.

One invocation covers every policy: pass ``--models outputs/<experiment>/models`` and it
evaluates each checkpoint under it against every rung, writing a single tidy CSV with the
initial speed as a first-class column.

Usage:
    python test/evaluate_crash.py --models outputs/small_big/models
    python test/evaluate_crash.py --models outputs/small_big/models --episodes 20
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test"))

from core.config import load_and_validate_system_config  # noqa: E402
from core.factory import DynamicsFactory  # noqa: E402
from evaluate_scaling import (  # noqa: E402
    build_policy, config_start_state, evaluate_fleet, read_checkpoint,
)
from systems.seed_utils import default_action_noise_seed_for_config  # noqa: E402

DEFAULT_CONFIG_GLOB = "test/config/study/crash/unicycle2_crash_v*.yaml"
# What training used. A fixed start plus a deterministic policy is one episode repeated,
# so without this the MLP's success rate is pass/fail rather than a rate.
DEFAULT_ACTION_NOISE = 0.03
# Margin over the modelled traverse+settle time, and the rounding grid -- the same values
# generate_small_robot_configs.py sizes training episodes with, so an episode that fits in
# training also fits here.
BUDGET_MARGIN = 1.18
BUDGET_GRID = 50

CSV_FIELDS = (
    "run", "policy_type", "encoder_type", "train_seed",
    "config", "v0", "eval_fleet_size", "episodes", "steps", "action_noise_std",
    "success_rate", "collision_rate", "timeout_rate",
    "robot_success_rate", "robot_collision_rate",
    "mean_min_pair_distance", "min_min_pair_distance", "p05_min_pair_distance",
    "mean_steps", "mean_goal_position_error", "mean_goal_heading_error",
    "mean_action_ms", "wall_time_s",
)


def initial_speed(raw_config: dict) -> float:
    robot = raw_config["robots"][0]
    start = robot.get("start") or robot["config"]["start"]
    return float(start[3])  # unicycle2 state = [x, y, theta, v, omega]


def crash_step_budget(raw_config: dict) -> int:
    """Steps per episode, from the robot's dynamics rather than from distance alone.

    An episode is a drive plus a turn: unicycle2 reaches a goal *pose*, and a unicycle can
    only change heading by rotating, so every episode ends with an in-place rotation that
    is_done also requires to have stopped. The rotation is the part a distance-based budget
    misses, and on the 0.1 m robot it is the larger half -- ~113 steps against ~84 for the
    drive. Modelled from rest to rest, which over-counts slightly for a rung that starts
    already moving; that is the safe direction to be wrong in.
    """
    robot = raw_config["robots"][0]["config"]
    dt = float(robot["dt"])
    v_max = float(robot["max_linear_vel"])
    a_v = float(robot["max_linear_accel"])
    w_max = float(robot["max_angular_vel"])
    a_w = float(robot["max_angular_accel"])

    distance = max(
        float(np.linalg.norm(
            np.asarray((entry.get("start") or entry["config"]["start"])[:2], dtype=float)
            - np.asarray((entry.get("goal") or entry["config"]["goal"])[:2], dtype=float)
        ))
        for entry in raw_config["robots"]
    )
    accel_distance = v_max * v_max / (2.0 * a_v)
    if 2.0 * accel_distance >= distance:
        traverse = 2.0 * math.sqrt(a_v * distance) / a_v
    else:
        traverse = 2.0 * (v_max / a_v) + (distance - 2.0 * accel_distance) / v_max
    settle = math.pi / w_max + w_max / a_w
    steps = (traverse + settle) * BUDGET_MARGIN / dt
    return int(math.ceil(steps / BUDGET_GRID) * BUDGET_GRID)


def discover_checkpoints(models_dir: Path) -> list[tuple[str, Path]]:
    """(run name, checkpoint) for every run under a models directory.

    train_dagger.py names the file <policy_type>_dagger_checkpoint.pt and the run
    directory does not always carry the type, so take whichever one is present -- the
    same rule eval.sh uses.
    """
    runs = []
    for run_dir in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        found = sorted(run_dir.glob("*_dagger_checkpoint.pt"))
        if found:
            runs.append((run_dir.name, found[0]))
        else:
            print(f"[skip] {run_dir.name}: no checkpoint")
    if not runs:
        raise SystemExit(f"no checkpoints under {models_dir}")
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--models", type=Path,
                        help="models directory, e.g. outputs/small_big/models -- every run under it")
    source.add_argument("--checkpoints", type=Path, nargs="+", help="explicit checkpoint paths")
    parser.add_argument("--configs", type=Path, nargs="+", default=None,
                        help=f"crash configs (default: {DEFAULT_CONFIG_GLOB})")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--action-noise-std", type=float, default=DEFAULT_ACTION_NOISE)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--steps", type=int, default=None,
                        help="override the derived per-config budget")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="default: <models>/../eval/crash.csv")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.models:
        runs = discover_checkpoints(args.models)
        default_csv = args.models.parent / "eval" / "crash.csv"
    else:
        runs = [(path.parent.name, path) for path in args.checkpoints]
        default_csv = Path("outputs/crash_eval.csv")
    output_csv = args.output_csv or default_csv

    configs = args.configs or sorted((PROJECT_ROOT).glob(DEFAULT_CONFIG_GLOB))
    if not configs:
        raise SystemExit(f"no configs matched {DEFAULT_CONFIG_GLOB}")

    # Loaded once: every rung shares the robot, so the budget and the speed are read per
    # config but the fleet geometry is not re-derived per policy.
    ladder = []
    for config_path in configs:
        raw = yaml.safe_load(Path(config_path).read_text())
        ladder.append((Path(config_path), raw, initial_speed(raw)))
    ladder.sort(key=lambda entry: entry[2])
    print(f"crash ladder: {len(ladder)} rungs, v0 = "
          f"{[round(v, 3) for _, _, v in ladder]} m/s")
    print(f"{len(runs)} runs: {', '.join(name for name, _ in runs)}\n")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Truncate rather than append: evaluate_scaling.py appends, which is what let a smoke
    # run's rows survive into a real run's CSV and be plotted as the result.
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for run_name, checkpoint_path in runs:
            checkpoint = read_checkpoint(str(checkpoint_path), device)
            policy_type = str(checkpoint.get("policy_type", "mlp")).lower()
            train_seed = run_name.rsplit("_s", 1)[-1] if "_s" in run_name else ""
            policy = None if policy_type == "safeflow" else build_policy(checkpoint, device)
            print(f"=== {run_name}  ({policy_type}, "
                  f"{checkpoint.get('encoder_type')} encoder)")

            for config_path, raw, v0 in ladder:
                validated = load_and_validate_system_config("multi_robot", config_path)
                simulator = DynamicsFactory.create(system_name="multi_robot", config=validated)
                if policy_type == "safeflow":
                    # One CasADi Opti per robot, sized to this fleet, so it is rebuilt per
                    # config; the scenario doubles as the projector's planner config.
                    policy = build_policy(
                        checkpoint, device, simulator=simulator, planner_config=validated
                    )
                start = config_start_state(validated)
                if simulator.is_collision(start):
                    raise SystemExit(f"{config_path}: configured start is already in collision.")
                steps = args.steps or crash_step_budget(raw)

                began = time.perf_counter()
                metrics = evaluate_fleet(
                    simulator=simulator, policy=policy, device=device,
                    episodes=args.episodes, steps=steps, seed_start=args.seed_start,
                    action_noise_std=args.action_noise_std,
                    action_noise_seed=default_action_noise_seed_for_config(validated),
                    fixed_initial_state=start,
                    observation_horizon=int(checkpoint.get("observation_horizon", 1)),
                )
                elapsed = time.perf_counter() - began

                row: dict[str, Any] = {
                    "run": run_name, "policy_type": policy_type,
                    "encoder_type": checkpoint.get("encoder_type", ""),
                    "train_seed": train_seed,
                    "config": str(Path(config_path).relative_to(PROJECT_ROOT))
                    if Path(config_path).is_absolute() else str(config_path),
                    "v0": round(v0, 4), "eval_fleet_size": simulator.num_robots,
                    "episodes": args.episodes, "steps": steps,
                    "action_noise_std": args.action_noise_std,
                    "wall_time_s": round(elapsed, 1),
                }
                row.update({
                    field: metrics[field] for field in CSV_FIELDS
                    if field in metrics
                })
                writer.writerow(row)
                handle.flush()
                print(f"  v0={v0:<5.2f} success={metrics['success_rate']:.3f} "
                      f"collision={metrics['collision_rate']:.3f} "
                      f"timeout={metrics['timeout_rate']:.3f}  "
                      f"clearance worst={metrics['min_min_pair_distance']:.4f} "
                      f"mean={metrics['mean_min_pair_distance']:.4f}  "
                      f"{steps} steps  {metrics['mean_action_ms']:.1f} ms/step  ({elapsed:.0f}s)")
            print()

    print(f"wrote {output_csv}")
    print(f"plot it: python3 test/plot_crash_results.py --results {output_csv} --with-failures")


if __name__ == "__main__":
    main()
