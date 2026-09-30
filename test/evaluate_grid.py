"""Evaluate every checkpoint of an experiment on one scenario, in parallel.

Wraps ``test/evaluate_scaling.py``: this script only iterates the (run, config) grid,
manages parallelism, and merges per-run CSVs into one. Training and evaluation are
independent -- this needs only the checkpoints and the configs, so it can run later,
elsewhere, or again with other settings.

Usage:
    python test/evaluate_grid.py <experiment> [scenario|all]
        [--episodes 50] [--max-parallel 8] [--runner {docker,local}]
        [--seed-start 50000] [--step-budget-factor 3] [--action-noise-std 0.0]
        [--configs GLOB]

Reads  outputs/<experiment>/models/<run>/<policy_type>_dagger_checkpoint.pt
Writes outputs/<experiment>/eval/<scenario>/<run>.csv, merged into
       outputs/<experiment>/eval/<scenario>.csv (one row per checkpoint x config)
       outputs/<experiment>/eval/<scenario>/logs/<run>.log

Scenarios (each varies exactly one quantity):
    arena    N=2..32 in ONE fixed workspace: the task is identical -- same box,
             ~7 m to drive -- and only the number of robots sharing it changes.
    fleet    N=2..32 at the training density. Not in 'all': the box grows as sqrt(N),
             so it drags the path length along and isolates no better than 'arena'.
    density  0.25x..1.25x the training density at fixed N: spacing shrinks, ceiling
             does not.
    circle   antipodal swap from the configs' fixed starts, ringed at the training
             density for every N -- the stress case for head-on conflicts.
    crash    two-robot head-on ladder from fixed starts at rising initial speeds --
             fixed step budget (drive + in-place settle rotation) and action noise
             matching training (0.03), so the deterministic MLP does not collapse to
             one identical episode per rung.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import os
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# scenario -> (config glob, extra evaluate_scaling.py flags)
SCENARIOS: dict[str, tuple[str, list[str]]] = {
    "arena":   ("test/config/study/arena/*.yaml",   []),
    "fleet":   ("test/config/study/fleet/*.yaml",   []),
    "density": ("test/config/study/density/*.yaml", []),
    "circle":  ("test/config/study/circle/*.yaml",  ["--use-config-start"]),
    "crash":   ("test/config/study/crash/*.yaml",
                ["--use-config-start", "--steps", "250", "--action-noise-std", "0.03"]),
}
DEFAULT_ALL = ("arena", "density", "circle")


def docker_prefix() -> list[str]:
    return [
        "docker", "compose", "run", "--rm", "-T",
        "-u", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp", "-e", "USER=csvil",
        "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1",
        "-e", "OPENBLAS_NUM_THREADS=1",
        "csvil",
    ]


def local_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    return env


def find_checkpoint(run_dir: Path) -> Path | None:
    """The run's <policy_type>_dagger_checkpoint.pt, whichever policy type it is."""
    found = sorted(run_dir.glob("*_dagger_checkpoint.pt"))
    return found[0] if found else None


def merge_csvs(merged: Path, per_run: list[Path]) -> None:
    """Concatenate per-run CSVs, keeping one header."""
    merged.write_text("")
    first = True
    with merged.open("w") as out:
        for csv in per_run:
            if not csv.is_file():
                continue
            lines = csv.read_text().splitlines(keepends=True)
            if not lines:
                continue
            if first:
                out.writelines(lines)
                first = False
            else:
                out.writelines(lines[1:])
    rows = max(sum(1 for _ in merged.open()) - 1, 0)
    print(f"merged {rows} result rows into {merged}")


def eval_one(
    scenario: str, run: str, checkpoint: Path, configs: list[Path], extra: list[str],
    args: argparse.Namespace, out_dir: Path,
) -> tuple[str, bool]:
    per_run_csv = out_dir / f"{run}.csv"
    log_path = out_dir / "logs" / f"{run}.log"
    per_run_csv.unlink(missing_ok=True)  # evaluate_scaling.py appends; a rerun would stack rows.

    train_seed = ""
    match = re.search(r"_s(\d+)$", run)
    if match:
        train_seed = match.group(1)

    inner_cmd = [
        "python", "test/evaluate_scaling.py",
        "--checkpoint", str(checkpoint.relative_to(REPO_ROOT)),
        "--configs", *(str(c.relative_to(REPO_ROOT)) for c in configs),
        "--episodes", str(args.episodes),
        "--seed-start", str(args.seed_start),
        "--action-noise-std", str(args.action_noise_std),
        "--train-seed", train_seed,
        "--output-csv", str(per_run_csv.relative_to(REPO_ROOT)),
        *extra,
    ]
    if "--steps" not in extra:
        inner_cmd += ["--step-budget-factor", str(args.step_budget_factor)]

    if args.runner == "docker":
        cmd = docker_prefix() + inner_cmd
        env = None
    else:
        cmd = inner_cmd
        env = local_env()

    with log_path.open("w") as log:
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=REPO_ROOT, env=env)
    stamp = time.strftime("%H:%M:%S")
    if completed.returncode == 0:
        print(f"[{stamp}] {scenario} {run} OK", flush=True)
        return run, True
    tail = log_path.read_text().splitlines()[-5:]
    tail_lines = "\n".join(f"    {run}| {line}" for line in tail)
    print(f"[{stamp}] {scenario} {run} FAILED (exit {completed.returncode}) -- last log lines:\n{tail_lines}",
          flush=True)
    return run, False


def run_scenario(scenario: str, args: argparse.Namespace, model_dir: Path) -> None:
    glob_str = args.configs or SCENARIOS[scenario][0]
    extra = SCENARIOS[scenario][1]
    configs = sorted(Path(p) for p in glob.glob(str(REPO_ROOT / glob_str)))
    if not configs:
        raise SystemExit(f"no configs for scenario '{scenario}' ({glob_str}); "
                         f"run the generators in test/config/")

    runs = sorted(p for p in model_dir.iterdir() if p.is_dir())
    if not runs:
        raise SystemExit(f"no run directories under {model_dir}")

    out_dir = REPO_ROOT / "outputs" / args.experiment / "eval" / scenario
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    print(f"scenario={scenario}: {len(runs)} checkpoints x {len(configs)} configs, "
          f"{args.episodes} episodes, {args.max_parallel} at a time")

    per_run_csvs: list[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = []
        for run_dir in runs:
            checkpoint = find_checkpoint(run_dir)
            if checkpoint is None:
                print(f"[skip] {scenario} {run_dir.name}: no checkpoint in {run_dir}/")
                continue
            per_run_csvs.append(out_dir / f"{run_dir.name}.csv")
            futures.append(pool.submit(
                eval_one, scenario, run_dir.name, checkpoint, configs, extra, args, out_dir,
            ))
        for f in concurrent.futures.as_completed(futures):
            f.result()

    merged = REPO_ROOT / "outputs" / args.experiment / "eval" / f"{scenario}.csv"
    merge_csvs(merged, per_run_csvs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("experiment")
    p.add_argument("scenario", nargs="?", default="arena",
                   choices=(*SCENARIOS.keys(), "all"))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--runner", choices=("docker", "local"), default="docker")
    p.add_argument("--seed-start", type=int, default=50000)
    p.add_argument("--step-budget-factor", type=float, default=3.0)
    p.add_argument("--action-noise-std", type=float, default=0.0)
    p.add_argument("--configs", default=None,
                   help="override the scenario's config glob (e.g. for smoke tests)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = REPO_ROOT / "outputs" / args.experiment / "models"
    if not model_dir.is_dir():
        raise SystemExit(f"no models at {model_dir}")

    scenarios = DEFAULT_ALL if args.scenario == "all" else (args.scenario,)
    for scenario in scenarios:
        run_scenario(scenario, args, model_dir)


if __name__ == "__main__":
    main()
