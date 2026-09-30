"""Train every policy config against every expert config, in parallel.

Wraps ``learning/train_dagger.py``: this script only iterates the (policy, expert, seed)
grid and manages parallelism. All DAgger schedule flags stay in the policy config's
``training:`` block, so a schedule flag here would silently override the config.

Usage:
    python learning/train_grid.py <experiment> <policy_dir_or_glob> <expert_dir_or_glob>
        [--seeds 0 1 2] [--max-parallel 8] [--runner {docker,local}]

Example -- the 4-robot cell of the encoder study:
    python learning/train_grid.py study2 \\
        learning/config/study/data_mid_n04 test/config/study/unicycle2_fleet_04.yaml

Each run <policy>_<expert>_s<seed> writes to
    outputs/<experiment>/models/<run>/    checkpoints and saved configs
    outputs/<experiment>/logs/<run>.log

Docker mode (default): -u sets the host uid/gid so bind-mounted files stay host-owned,
HOME=/tmp and USER=csvil satisfy code that calls getpass.getuser(), and *_NUM_THREADS=1
prevents IPOPT/BLAS/torch from oversubscribing when many runs share the machine.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def resolve_yaml_files(arg: str) -> list[Path]:
    """Files a path/glob argument names, one per entry, sorted, all inside the repo."""
    abs_arg = os.path.abspath(arg)
    if not os.path.exists(abs_arg) and any(c in arg for c in "*?["):
        matches = sorted(glob.glob(arg))
        if not matches:
            raise SystemExit(f"no files match: {arg}")
        return [p for m in matches for p in resolve_yaml_files(m)]

    if not os.path.exists(abs_arg):
        raise SystemExit(f"no such path: {arg}")
    if not abs_arg.startswith(str(REPO_ROOT) + os.sep) and abs_arg != str(REPO_ROOT):
        raise SystemExit(f"{arg} is outside {REPO_ROOT}; a container cannot see it")

    path = Path(abs_arg)
    if path.is_dir():
        found = sorted(p for p in path.iterdir()
                       if p.is_file() and p.suffix in (".yaml", ".yml"))
        if not found:
            raise SystemExit(f"no .yaml files in {arg}")
        return found
    if path.suffix not in (".yaml", ".yml"):
        raise SystemExit(f"not a .yaml file: {arg}")
    return [path]


def repo_relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


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


def train_one(
    policy: Path, expert: Path, seed: int, experiment: str, runner: str, log_dir: Path,
    model_dir: Path,
) -> tuple[str, bool, str]:
    name = f"{policy.stem}_{expert.stem}_s{seed}"
    log_path = log_dir / f"{name}.log"

    inner_cmd = [
        "python", "learning/train_dagger.py",
        "--experiment-name", name,
        "--system", "multi_robot",
        "--expert-config", repo_relative(expert),
        "--policy-config", repo_relative(policy),
        "--seed", str(seed),
        "--checkpoint-dir", repo_relative(model_dir),
    ]
    if runner == "docker":
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
        print(f"[{stamp}] {name} OK", flush=True)
        return name, True, ""
    tail = log_path.read_text().splitlines()[-5:]
    tail_lines = "\n".join(f"    {name}| {line}" for line in tail)
    print(f"[{stamp}] {name} FAILED (exit {completed.returncode}) -- last log lines:\n{tail_lines}",
          flush=True)
    return name, False, tail_lines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("experiment", help="name for outputs/<experiment>/ subdirs")
    p.add_argument("policy", help="policy YAML, directory, or glob")
    p.add_argument("expert", help="expert YAML, directory, or glob")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--runner", choices=("docker", "local"), default="docker")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not EXPERIMENT_NAME_RE.match(args.experiment):
        raise SystemExit(f"invalid experiment name '{args.experiment}'")

    policies = resolve_yaml_files(args.policy)
    experts = resolve_yaml_files(args.expert)

    model_dir = REPO_ROOT / "outputs" / args.experiment / "models"
    log_dir = REPO_ROOT / "outputs" / args.experiment / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(p, e, s) for e in experts for p in policies for s in args.seeds]
    print(f"experiment {args.experiment}: {len(jobs)} runs "
          f"({len(policies)} policies x {len(experts)} experts x {len(args.seeds)} seeds), "
          f"{args.max_parallel} at a time")
    print(f"  policies: {[repo_relative(p) for p in policies]}")
    print(f"  experts:  {[repo_relative(e) for e in experts]}")
    print(f"  seeds:    {args.seeds}")

    ok = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = [
            pool.submit(train_one, p, e, s, args.experiment, args.runner, log_dir, model_dir)
            for p, e, s in jobs
        ]
        for f in concurrent.futures.as_completed(futures):
            _, success, _ = f.result()
            if success:
                ok += 1
            else:
                failed += 1

    print(f"done: {ok} OK, {failed} failed. Checkpoints in {model_dir}/, logs in {log_dir}/")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
