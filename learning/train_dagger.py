from __future__ import annotations

import argparse
import ast
import os
import sys
import time
import yaml
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config
from core.factory import DynamicsFactory
from learning.config_loaders import (
    default_checkpoint_dir_for_system,
    default_dataset_root_for_system, default_repo_id_for_system,
    load_dagger_training_config,
    load_encoder_config, load_flow_config, load_mlp_hidden_dims,
    load_observation_horizon, load_policy_type, load_prediction_horizon,
)
from learning.dagger import DaggerConfig, DaggerTrainer
from systems.initial_state_utils import parse_goal_states_argument, parse_initial_states_argument

"""High-level CLI plumbing and orchestration for DAgger training.

The reusable DAgger pipeline components live under learning/dagger/:
DaggerConfig (learning/dagger/dagger_config.py) and DaggerTrainer
(learning/dagger/dagger_trainer.py) hold the actual training logic; this
file only parses CLI args/policy YAML, resolves them into a DaggerConfig,
and hands off to DaggerTrainer.
"""


def _yaml_safe(value: Any) -> Any:
    """Recursively coerce a value into plain types yaml.safe_dump can render."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    return value


def _print_yaml_block(title: str, data: dict[str, Any]) -> None:
    print(f"{title}:")
    for line in yaml.safe_dump(_yaml_safe(data), sort_keys=False).splitlines():
        print(f"  {line}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a policy with DAgger")
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--system", type=str.lower, choices=DynamicsFactory.names(), required=True)
    p.add_argument("--expert-config", required=True)
    p.add_argument("--repo-id")
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--planner", choices=["casadi"])
    p.add_argument("--dagger-iterations", type=int)
    p.add_argument("--trajectories-per-iteration", nargs="+", type=int)
    p.add_argument("--steps-per-trajectory", type=int)
    p.add_argument("--action-noise-std", type=float)
    p.add_argument(
        "--training-curriculum",
        nargs="+",
        type=str,
        default=None,
        help=(
            "one 'random' or 'config' entry per DAgger round, selecting whether that round's "
            "rollouts use random goals/states or the explicit --initial-states/--goal-states lists."
        ),
    )
    p.add_argument(
        "--round-seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "one base seed per DAgger round, controlling which seed drives that round's "
            "random initial-state/goal sampling instead of the single global --seed."
        ),
    )
    p.add_argument(
        "--restart-round-seed",
        action=argparse.BooleanOptionalAction,
        help=(
            "when set, rounds sharing the same --round-seeds value replay identical random "
            "initial-state/goal draws (a true stream restart); when unset, the round index "
            "still varies the draws even if the nominal seed repeats (today's behavior)."
        ),
    )
    p.add_argument(
        "--initial-states",
        type=str,
        default=None,
        help=(
            "explicit initial state specs. Examples: '[x, y, ...]' for one rollout, "
            "'[[...], [...]]' for multiple global states, or "
            "'[[[robot1...], [robot2...]], ...]' for multi-robot rollouts. "
            "When exhausted, collection falls back to simulator RNG sampling."
        ),
    )
    p.add_argument(
        "--goal-states",
        type=str,
        default=None,
        help=(
            "explicit goal state specs, paired index-for-index with --initial-states. "
            "Examples: '[x, y, ...]' for one rollout, '[[...], [...]]' for multiple global goals, or "
            "'[[[robot1...], [robot2...]], ...]' for multi-robot rollouts. "
            "When either list is exhausted, remaining rollouts in that round fall back to random."
        ),
    )
    p.add_argument(
        "--workspace-bounds",
        nargs=2,
        type=float,
        default=None,
        help=(
            "[min, max] shared per-coordinate square both random goals and random initial "
            "positions are drawn from (random-curriculum rounds and eval fallback), with "
            "rejection sampling enforcing d_safe between all robots' goals, and between each "
            "robot's initial position and every other robot's initial position/goal."
        ),
    )
    p.add_argument(
        "--tolerance-overrides",
        type=str,
        default=None,
        help=(
            "per-experiment override for the expert config's convergence tolerances, as a Python-literal "
            "dict matching the target system's tolerance keys, e.g. "
            "'{\"pos_tol\": 0.2, \"theta_tol\": 1.1, \"vel_tol\": 0.05, \"omega_tol\": 0.05}' for unicycle2, "
            "or '{\"error_tolerance\": 0.05}' for single_integrator/double_integrator/unicycle1."
        ),
    )
    p.add_argument(
        "--eval-tolerance-overrides",
        type=str,
        default=None,
        help=(
            "same format as --tolerance-overrides, but applied only to evaluation rollouts, not "
            "data collection/training. Falls back to --tolerance-overrides when omitted. Use this "
            "to relax the pass/fail standard for eval (e.g. 'close enough and collision-free') "
            "without changing what collection/training demonstrates against."
        ),
    )
    p.add_argument("--expert-mix-beta-start", type=float)
    p.add_argument("--expert-mix-beta-end", type=float)
    p.add_argument("--expert-mix-beta-decay-rate", type=float)
    p.add_argument("--expert-mix-decay-after-eval-success", type=float)
    p.add_argument("--adaptive-beta-recovery", action=argparse.BooleanOptionalAction)
    p.add_argument("--expert-mix-beta-recovery", type=float)
    p.add_argument("--expert-mix-beta-recovery-increment", type=float)
    p.add_argument("--target-epochs-per-round", nargs="+", type=float)
    p.add_argument("--eval-episodes", type=int)
    p.add_argument("--eval-steps", type=int)
    p.add_argument("--eval-seed-start", type=int)
    p.add_argument("--eval-action-noise-std", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument(
        "--policy-config",
        type=Path,
        default=Path(PROJECT_ROOT) / "learning/config/default_policy_config.yaml",
    )
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--max-train-steps", type=int)
    return p.parse_args()

def save_experiment_configs(args: argparse.Namespace, experiment_dir: Path,  repo_id: str, dataset_root: Path) -> None:
    """
    saves experiment configs and run arguments to experiment directory
    """

    shutil.copy2(args.expert_config, experiment_dir / "expert_config.yaml")
    shutil.copy2(args.policy_config, experiment_dir / "policy_config.yaml")

    # saves resolved run arguments:
    run_args = vars(args).copy()
    run_args["repo_id"] = repo_id
    run_args["dataset_root"] = dataset_root
    run_args["checkpoint_dir"] = experiment_dir
    run_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in run_args.items()
    }

    with (experiment_dir / "run_args.yaml").open("w") as f:
        yaml.safe_dump(run_args, f, sort_keys=False)

def main() -> None:
    args = parse_args()
    _print_yaml_block("CLI arguments (train_dagger.py, unset flags fall back to policy YAML/defaults)", vars(args))
    validated = load_and_validate_system_config(args.system, args.expert_config)
    training_config = load_dagger_training_config(args.policy_config)

    def option(name: str, default: Any) -> Any:
        cli_value = getattr(args, name)
        return cli_value if cli_value is not None else training_config.get(name, default)

    dagger_iterations = int(option("dagger_iterations", 4))
    trajectories_per_iteration = [
        int(value) for value in option("trajectories_per_iteration", [20])
    ]
    target_epochs_per_round = [
        float(value) for value in option("target_epochs_per_round", [30.0])
    ]
    initial_states_config = option("initial_states", None)
    initial_states = (
        parse_initial_states_argument(initial_states_config)
        if isinstance(initial_states_config, str)
        else initial_states_config
    )
    goal_states_config = option("goal_states", None)
    goal_states = (
        parse_goal_states_argument(goal_states_config)
        if isinstance(goal_states_config, str)
        else goal_states_config
    )
    training_curriculum_config = option("training_curriculum", None)
    round_seeds_config = option("round_seeds", None)
    restart_round_seed = bool(option("restart_round_seed", False))
    tolerance_overrides_config = option("tolerance_overrides", None)
    if isinstance(tolerance_overrides_config, str):
        try:
            tolerance_overrides = ast.literal_eval(tolerance_overrides_config)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "Unable to parse --tolerance-overrides. Use Python-literal dict syntax like "
                "'{\"pos_tol\": 0.2, \"theta_tol\": 1.1}'."
            ) from exc
        if not isinstance(tolerance_overrides, dict):
            raise ValueError("--tolerance-overrides must evaluate to a dict.")
    else:
        tolerance_overrides = tolerance_overrides_config
    eval_tolerance_overrides_config = option("eval_tolerance_overrides", None)
    if isinstance(eval_tolerance_overrides_config, str):
        try:
            eval_tolerance_overrides = ast.literal_eval(eval_tolerance_overrides_config)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "Unable to parse --eval-tolerance-overrides. Use Python-literal dict syntax like "
                "'{\"pos_tol\": 0.2, \"theta_tol\": 1.1}'."
            ) from exc
        if not isinstance(eval_tolerance_overrides, dict):
            raise ValueError("--eval-tolerance-overrides must evaluate to a dict.")
    else:
        eval_tolerance_overrides = eval_tolerance_overrides_config
    if (args.repo_id is None) != (args.dataset_root is None):
        raise ValueError("Provide both --repo-id and --dataset-root together, or omit both.")
    fresh = args.repo_id is None
    timestamp = time.time_ns()
    repo_id = (
        default_repo_id_for_system(args.system, timestamp)
        if fresh
        else str(args.repo_id)
    )
    dataset_root = (
        default_dataset_root_for_system(args.system, timestamp)
        if fresh
        else Path(args.dataset_root)
    )
    trajectories, epochs, training_curriculum, round_seeds = DaggerTrainer.schedules(
        trajectories_per_iteration,
        target_epochs_per_round,
        dagger_iterations,
        training_curriculum_config,
        round_seeds_config,
    )

    # path to experiment directory where configs and checkpoints will be saved:
    experiment_dir = (
        args.checkpoint_dir or default_checkpoint_dir_for_system(args.system)
    ) / args.experiment_name

    # allows to override existing experiment directory:
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # saves configs before training (in case of failure):
    save_experiment_configs(
        args=args,
        experiment_dir=experiment_dir,
        repo_id=repo_id,
        dataset_root=dataset_root,
    )

    cfg = DaggerConfig(
        system=args.system,
        experiment_config=validated,
        repo_id=repo_id,
        dataset_root=dataset_root,
        start_with_aggregation=fresh,
        planner_name=str(option("planner", "casadi")),
        dagger_iterations=dagger_iterations,
        trajectories_per_iteration=trajectories,
        steps_per_trajectory=int(option("steps_per_trajectory", 150)),
        action_noise_std=float(option("action_noise_std", 0.0)),
        initial_states=initial_states,
        goal_states=goal_states,
        training_curriculum=training_curriculum,
        round_seeds=round_seeds,
        restart_round_seed=restart_round_seed,
        workspace_bounds=option("workspace_bounds", None),
        tolerance_overrides=tolerance_overrides,
        eval_tolerance_overrides=eval_tolerance_overrides,
        expert_mix_beta_start=float(option("expert_mix_beta_start", 0.8)),
        expert_mix_beta_end=float(option("expert_mix_beta_end", 0.0)),
        expert_mix_beta_decay_rate=option("expert_mix_beta_decay_rate", None),
        expert_mix_decay_after_eval_success=option("expert_mix_decay_after_eval_success", None),
        adaptive_beta_recovery=bool(option("adaptive_beta_recovery", False)),
        expert_mix_beta_recovery=float(option("expert_mix_beta_recovery", 1.0)),
        expert_mix_beta_recovery_increment=float(option("expert_mix_beta_recovery_increment", 1.0)),
        target_epochs_per_round=epochs,
        eval_episodes=int(option("eval_episodes", 10)),
        eval_steps=option("eval_steps", None),
        eval_seed_start=int(option("eval_seed_start", 10000)),
        eval_action_noise_std=float(option("eval_action_noise_std", 0.0)),
        batch_size=int(option("batch_size", 64)),
        learning_rate=float(option("learning_rate", 1e-3)),
        mlp_hidden_dims=load_mlp_hidden_dims(args.policy_config),
        prediction_horizon=load_prediction_horizon(args.policy_config),
        observation_horizon=load_observation_horizon(args.policy_config),
        encoder_config=load_encoder_config(args.policy_config),
        policy_type=load_policy_type(args.policy_config),
        flow_config=load_flow_config(args.policy_config),
        checkpoint_dir=experiment_dir,
        seed=int(option("seed", 99)),
        max_train_steps=option("max_train_steps", None),
    )
    _print_yaml_block("Resolved DaggerConfig (CLI + policy YAML + hardcoded defaults, fully merged)", asdict(cfg))
    DaggerTrainer(cfg).run()

if __name__ == "__main__":
    main()
