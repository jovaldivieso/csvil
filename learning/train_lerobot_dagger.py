from __future__ import annotations

import argparse
import copy
import gc
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from learning.train_lerobot import run_training
from planning.casadi_planner import PlannerSolveError
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol


def get_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_observation_feature(feature_name: str) -> bool:
    return feature_name.startswith("observation.") or ".observation." in feature_name


def create_policy_input(
    simulator: DynamicsProtocol,
    observation: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Build policy input tensors according to simulator dataset feature ordering.
    """
    policy_input: dict[str, torch.Tensor] = {}
    features = simulator.get_dataset_features()

    dummy_action = np.zeros(int(simulator.nu), dtype=np.float32)
    packed_frame = simulator.format_dataset_frame(observation, dummy_action)

    for feature_name in features.keys():
        if is_observation_feature(feature_name):
            if feature_name not in packed_frame:
                raise KeyError(
                    f"Missing observation feature '{feature_name}' in packed frame. "
                    "Check simulator.format_dataset_frame() and dataset schema alignment."
                )
            policy_input[feature_name] = torch.as_tensor(
                packed_frame[feature_name],
                dtype=torch.float32,
                device=device,
            ).view(1, -1)

    return policy_input


def load_lerobot_policy(
    policy_type: str,
    model_dir: Path,
    device: torch.device,
) -> DiffusionPolicy | ACTPolicy:
    normalized = policy_type.strip().lower()

    if normalized == "diffusion":
        policy = DiffusionPolicy.from_pretrained(str(model_dir))
    elif normalized == "act":
        policy = ACTPolicy.from_pretrained(str(model_dir))
    else:
        raise ValueError("'policy_type' must be one of {'diffusion', 'act'}.")

    policy.eval()
    policy.to(device)
    return policy


def list_pretrained_model_dirs(search_root: Path) -> set[Path]:
    if not search_root.exists():
        return set()

    matches = {
        path.resolve()
        for path in search_root.glob("**/checkpoints/last/pretrained_model")
        if path.is_dir()
    }
    return matches


def newest_pretrained_model_dir(paths: set[Path]) -> Path | None:
    if len(paths) == 0:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def detect_new_pretrained_model_dir(
    search_root: Path,
    before_paths: set[Path],
) -> Path:
    after_paths = list_pretrained_model_dirs(search_root)
    new_paths = after_paths.difference(before_paths)

    newest_new = newest_pretrained_model_dir(new_paths)
    if newest_new is not None:
        return newest_new

    newest_any = newest_pretrained_model_dir(after_paths)
    if newest_any is not None:
        return newest_any

    raise FileNotFoundError(
        "Could not find a LeRobot checkpoint directory named 'checkpoints/last/pretrained_model' "
        f"under {search_root}."
    )


def read_training_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("LeRobot training config must be a YAML mapping at top level.")
    return config


def write_training_config(config: dict[str, Any]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_lerobot_dagger_train.yaml",
        delete=False,
        encoding="utf-8",
    )
    temp_path = Path(handle.name)
    try:
        yaml.safe_dump(config, handle, sort_keys=False)
    finally:
        handle.close()
    return temp_path


def run_lerobot_training_isolated_args(config_path: str) -> None:
    """
    Run LeRobot training while shielding Draccus from this script's CLI args.
    """
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        run_training(config_path=config_path)
    finally:
        sys.argv = original_argv


def build_iteration_training_config(
    base_config: Mapping[str, Any],
    dataset_root: Path,
    repo_id: str,
    pretrained_path: Path | None,
    disable_push_to_hub: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))

    dataset_cfg = config.setdefault("dataset", {})
    if not isinstance(dataset_cfg, dict):
        raise ValueError("Expected 'dataset' section in LeRobot config to be a mapping.")
    dataset_cfg["root"] = str(dataset_root)
    dataset_cfg["repo_id"] = repo_id

    policy_cfg = config.setdefault("policy", {})
    if not isinstance(policy_cfg, dict):
        raise ValueError("Expected 'policy' section in LeRobot config to be a mapping.")

    if pretrained_path is None:
        policy_cfg.pop("pretrained_path", None)
    else:
        policy_cfg["pretrained_path"] = str(pretrained_path)

    if disable_push_to_hub:
        policy_cfg["push_to_hub"] = False

    return config


def infer_policy_type(
    cli_policy_type: str | None,
    base_config: Mapping[str, Any],
) -> str:
    if cli_policy_type is not None:
        return cli_policy_type.lower()

    policy_cfg = base_config.get("policy", {})
    if not isinstance(policy_cfg, Mapping):
        raise ValueError("Expected 'policy' section in LeRobot config to be a mapping.")

    policy_type = policy_cfg.get("type")
    if not isinstance(policy_type, str):
        raise ValueError(
            "Could not infer policy type from LeRobot config. "
            "Provide --policy-type explicitly."
        )

    normalized = policy_type.lower()
    if normalized not in {"act", "diffusion"}:
        raise ValueError("Only 'act' and 'diffusion' are supported for LeRobot DAgger.")
    return normalized


def run_training_round(
    *,
    base_training_config: Mapping[str, Any],
    dataset_root: Path,
    repo_id: str,
    pretrained_path: Path | None,
    disable_push_to_hub: bool,
    train_output_root: Path,
) -> Path:
    train_cfg = build_iteration_training_config(
        base_config=base_training_config,
        dataset_root=dataset_root,
        repo_id=repo_id,
        pretrained_path=pretrained_path,
        disable_push_to_hub=disable_push_to_hub,
    )

    temp_train_cfg_path = write_training_config(train_cfg)
    print(f"Training config for this round: {temp_train_cfg_path}")
    if pretrained_path is None:
        print("Fine-tuning starts from scratch for this round.")
    else:
        print(f"Fine-tuning warm-start path: {pretrained_path}")

    before_paths = list_pretrained_model_dirs(train_output_root)
    try:
        run_lerobot_training_isolated_args(config_path=str(temp_train_cfg_path))
    finally:
        temp_train_cfg_path.unlink(missing_ok=True)

    trained_model_dir = detect_new_pretrained_model_dir(
        search_root=train_output_root,
        before_paths=before_paths,
    )
    print(f"Using trained policy checkpoint: {trained_model_dir}")
    return trained_model_dir


def collect_lerobot_dagger_data(
    simulator: DynamicsProtocol,
    expert_planner: PlannerProtocol,
    policy: DiffusionPolicy | ACTPolicy,
    dataset_writer: LeRobotDataset,
    trajectories_per_iteration: int,
    steps_per_trajectory: int,
    device: torch.device,
) -> int:
    """
    Roll out LeRobot policy, query expert at visited states, append corrective labels.
    """
    successful_episodes = 0
    attempted_episodes = 0
    max_attempts = max(trajectories_per_iteration * 3, trajectories_per_iteration)

    while successful_episodes < trajectories_per_iteration:
        attempted_episodes += 1
        if attempted_episodes > max_attempts:
            raise RuntimeError(
                "Too many failed DAgger rollout attempts. "
                f"Collected {successful_episodes}/{trajectories_per_iteration} episodes."
            )

        state = simulator.reset_random()
        done_counter = 0
        planner_failed = False

        if hasattr(policy, "reset"):
            policy.reset()
        if hasattr(expert_planner, "reset"):
            expert_planner.reset()

        for _ in range(steps_per_trajectory):
            observation = simulator.observe(state)

            policy_input = create_policy_input(
                simulator=simulator,
                observation=observation,
                device=device,
            )

            with torch.inference_mode():
                action_tensor = policy.select_action(policy_input)
            policy_action = action_tensor.squeeze(0).cpu().numpy()

            try:
                expert_action = expert_planner(observation)
            except PlannerSolveError as exc:
                print(f"Skipping episode due to planner failure: {exc}")
                planner_failed = True
                break

            frame_data = simulator.format_dataset_frame(observation, expert_action)
            frame_data["task"] = "reach target"
            dataset_writer.add_frame(frame_data)

            # Environment advances with learner action; labels come from expert correction.
            state = simulator.step(state, policy_action)

            if simulator.is_done(state):
                done_counter += 1
                if done_counter >= 5:
                    break

        if planner_failed:
            continue

        dataset_writer.save_episode()
        successful_episodes += 1

    return successful_episodes


@dataclass(frozen=True)
class LeRobotDaggerConfig:
    system: str
    experiment_config_path: Path
    lerobot_training_config_path: Path
    repo_id: str
    dataset_root: Path
    planner_name: str
    dagger_iterations: int
    trajectories_per_iteration: int
    steps_per_trajectory: int
    train_output_root: Path
    seed: int
    policy_type: str | None
    initial_pretrained_path: Path | None
    disable_push_to_hub: bool


def run_lerobot_dagger(cfg: LeRobotDaggerConfig) -> None:
    if cfg.dagger_iterations < 0:
        raise ValueError("'dagger_iterations' must be non-negative.")
    if cfg.trajectories_per_iteration <= 0:
        raise ValueError("'trajectories_per_iteration' must be positive.")
    if cfg.steps_per_trajectory <= 0:
        raise ValueError("'steps_per_trajectory' must be positive.")

    if not cfg.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {cfg.dataset_root}")

    set_seed(cfg.seed)
    device = get_inference_device()
    print(f"Running LeRobot DAgger on device: {device}")

    validated_experiment_config = load_and_validate_system_config(
        system_name=cfg.system,
        config_path=str(cfg.experiment_config_path),
    )

    base_training_config = read_training_config(cfg.lerobot_training_config_path)
    policy_type = infer_policy_type(cfg.policy_type, base_training_config)

    print("\n=== Initial LeRobot training pass (no aggregation) ===")
    previous_pretrained_path = run_training_round(
        base_training_config=base_training_config,
        dataset_root=cfg.dataset_root,
        repo_id=cfg.repo_id,
        pretrained_path=cfg.initial_pretrained_path,
        disable_push_to_hub=cfg.disable_push_to_hub,
        train_output_root=cfg.train_output_root,
    )

    if cfg.dagger_iterations == 0:
        print("No DAgger refinements requested (--dagger-iterations 0).")
        print("\nLeRobot DAgger loop complete.")
        return

    for refinement in range(1, cfg.dagger_iterations + 1):
        print(f"\n=== DAgger refinement {refinement}/{cfg.dagger_iterations} ===")

        simulator = DynamicsFactory.create(system_name=cfg.system, config=validated_experiment_config)
        expert_planner = PlannerFactory.create(
            planner_name=cfg.planner_name,
            simulator=simulator,
            config=validated_experiment_config,
        )
        policy = load_lerobot_policy(
            policy_type=policy_type,
            model_dir=previous_pretrained_path,
            device=device,
        )

        dataset_writer = LeRobotDataset.resume(repo_id=cfg.repo_id, root=cfg.dataset_root)
        try:
            collected_episodes = collect_lerobot_dagger_data(
                simulator=simulator,
                expert_planner=expert_planner,
                policy=policy,
                dataset_writer=dataset_writer,
                trajectories_per_iteration=cfg.trajectories_per_iteration,
                steps_per_trajectory=cfg.steps_per_trajectory,
                device=device,
            )
        finally:
            dataset_writer.finalize()
            del dataset_writer
            del policy
            gc.collect()

        print(f"Collected {collected_episodes} new DAgger episodes")

        previous_pretrained_path = run_training_round(
            base_training_config=base_training_config,
            dataset_root=cfg.dataset_root,
            repo_id=cfg.repo_id,
            pretrained_path=previous_pretrained_path,
            disable_push_to_hub=cfg.disable_push_to_hub,
            train_output_root=cfg.train_output_root,
        )

    print("\nLeRobot DAgger loop complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative DAgger for LeRobot ACT/Diffusion policies"
    )
    parser.add_argument(
        "--system",
        type=str.lower,
        choices=DynamicsFactory.names(),
        required=True,
        help="name of system class, e.g. single_integrator, multi_robot, ...",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
        help="path to simulator/planner experiment YAML config",
    )
    parser.add_argument(
        "--lerobot-train-config",
        type=Path,
        required=True,
        help="path to LeRobot training YAML config",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot dataset repository id stored in metadata",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="path to existing local LeRobot dataset root",
    )
    parser.add_argument(
        "--planner",
        type=str.lower,
        default="casadi",
        choices=PlannerFactory.names(),
        help="expert planner used for corrective labels",
    )
    parser.add_argument(
        "--policy-type",
        type=str.lower,
        choices=["act", "diffusion"],
        default=None,
        help="optional override for policy type (inferred from LeRobot train config by default)",
    )
    parser.add_argument(
        "--dagger-iterations",
        type=int,
        default=4,
        help=(
            "number of DAgger refinement rounds (aggregate then retrain). "
            "Use 0 for pure offline training without aggregation"
        ),
    )
    parser.add_argument(
        "--trajectories-per-iteration",
        type=int,
        default=20,
        help="number of learner rollouts aggregated per DAgger iteration",
    )
    parser.add_argument(
        "--steps-per-trajectory",
        type=int,
        default=150,
        help="maximum rollout steps per learner trajectory",
    )
    parser.add_argument(
        "--train-output-root",
        type=Path,
        default=Path("outputs/train"),
        help=(
            "root directory where LeRobot training writes checkpoints; used to locate "
            "<run>/checkpoints/last/pretrained_model"
        ),
    )
    parser.add_argument(
        "--initial-pretrained-path",
        type=Path,
        default=None,
        help="optional pretrained_model directory to warm-start iteration 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--allow-push-to-hub",
        action="store_true",
        help="if set, keep push_to_hub behavior from LeRobot train config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = LeRobotDaggerConfig(
        system=args.system,
        experiment_config_path=args.experiment_config,
        lerobot_training_config_path=args.lerobot_train_config,
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        planner_name=args.planner,
        dagger_iterations=args.dagger_iterations,
        trajectories_per_iteration=args.trajectories_per_iteration,
        steps_per_trajectory=args.steps_per_trajectory,
        train_output_root=args.train_output_root,
        seed=args.seed,
        policy_type=args.policy_type,
        initial_pretrained_path=args.initial_pretrained_path,
        disable_push_to_hub=not args.allow_push_to_hub,
    )

    run_lerobot_dagger(cfg)


if __name__ == "__main__":
    main()
