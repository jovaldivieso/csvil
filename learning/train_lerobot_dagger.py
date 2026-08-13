from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
import draccus

from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from learning.dagger import (
    DaggerEvalMetrics,
    ExpertMixBetaController,
    apply_execution_noise,
    build_observation_feature_pack_cache,
    collect_dagger_rollouts,
    evaluate_policy_rollouts,
    observation_feature_names,
    pack_observation_features_from_cache,
    print_rollout_metrics,
    resolve_round_steps,
    resolve_initial_state_seed,
    set_seed,
    with_seeded_initial_state_config,
)
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol
from systems.seed_utils import (
    default_action_noise_seed_for_config,
)


def run_training(config_path: str) -> None:
    lerobot_training_config_path = os.path.abspath(config_path)
    print(f"Loading LeRobot config from: {lerobot_training_config_path}")

    train_pipeline_config = draccus.parse(
        config_class=TrainPipelineConfig,
        config_path=lerobot_training_config_path,
    )

    from lerobot.scripts.lerobot_train import train

    train(train_pipeline_config)


def get_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_policy_input(
    simulator: DynamicsProtocol,
    observation: np.ndarray,
    observation_feature_names_list: list[str],
    device: torch.device,
    add_batch_dim: bool = True,
    observation_feature_cache=None,
) -> dict[str, torch.Tensor]:
    """
    Build policy input tensors according to simulator dataset feature ordering.
    """
    if observation_feature_cache is None:
        observation_feature_cache = build_observation_feature_pack_cache(
            simulator,
            observation_feature_names_list,
        )

    packed_features = pack_observation_features_from_cache(observation, observation_feature_cache)
    policy_input: dict[str, torch.Tensor] = {}
    for feature_name in observation_feature_names_list:
        feature_tensor = torch.as_tensor(packed_features[feature_name], dtype=torch.float32)
        if add_batch_dim:
            feature_tensor = feature_tensor.view(1, -1)
        policy_input[feature_name] = feature_tensor

    if device.type != "cpu":
        policy_input = {feature_name: tensor.to(device) for feature_name, tensor in policy_input.items()}

    return policy_input


def load_lerobot_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    device: torch.device,
    pretrained_path: Path | None = None,
):
    policy_cfg = copy.deepcopy(policy_cfg)
    policy_cfg.device = str(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_path) if pretrained_path is not None else None,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
        },
    )
    return preprocessor, postprocessor


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
    return max(paths, key=lambda p: (p.stat().st_mtime, str(p)))


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
    steps_override: int | None,
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

    if steps_override is not None:
        config["steps"] = int(steps_override)
        if "save_freq" in config:
            save_freq = int(config["save_freq"])
            if save_freq > int(steps_override):
                config["save_freq"] = int(steps_override)

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


def load_policy_config_from_training_config(
    training_config_path: Path,
    policy_type: str,
):
    raw_config = read_training_config(training_config_path)

    # Full LeRobot train config path (dataset + policy + training fields)
    if "dataset" in raw_config and "policy" in raw_config:
        with draccus.config_type("yaml"):
            train_cfg = draccus.parse(
                config_class=TrainPipelineConfig,
                config_path=str(training_config_path),
                args=[],
            )

        policy_cfg = train_cfg.policy
        if policy_cfg is None:
            raise ValueError("LeRobot training config does not define a policy section.")
        return policy_cfg

    # Policy-only config path (used in this repository for some ACT/Diffusion configs)
    policy_section = raw_config.get("policy")
    if isinstance(policy_section, Mapping):
        policy_dict = dict(policy_section)
    else:
        policy_dict = dict(raw_config)

    normalized = policy_type.strip().lower()
    policy_dict.setdefault("type", normalized)
    policy_dict.pop("type", None)

    temp_path = write_training_config(policy_dict)
    try:
        with draccus.config_type("yaml"):
            if normalized == "act":
                return draccus.parse(
                    config_class=ACTConfig,
                    config_path=str(temp_path),
                    args=[],
                )
            if normalized == "diffusion":
                return draccus.parse(
                    config_class=DiffusionConfig,
                    config_path=str(temp_path),
                    args=[],
                )
    finally:
        temp_path.unlink(missing_ok=True)

    raise ValueError("'policy_type' must be one of {'act', 'diffusion'}.")


def build_uninitialized_lerobot_policy(
    policy_type: str,
    policy_config,
    device: torch.device,
) -> DiffusionPolicy | ACTPolicy:
    uninitialized_policy_config = copy.deepcopy(policy_config)
    uninitialized_policy_config.pretrained_path = None
    uninitialized_policy_config.device = str(device)

    normalized = policy_type.strip().lower()
    if normalized == "diffusion":
        policy = DiffusionPolicy(config=uninitialized_policy_config)
    elif normalized == "act":
        policy = ACTPolicy(config=uninitialized_policy_config)
    else:
        raise ValueError("'policy_type' must be one of {'diffusion', 'act'}.")

    policy.eval()
    policy.to(device)
    return policy


def default_repo_id_for_system(system: str, timestamp: int) -> str:
    return f"local/{system}_lerobot_dagger_{timestamp}"


def default_dataset_root_for_system(system: str, timestamp: int) -> Path:
    return Path(f"data/lerobot_dataset_{system}_lerobot_dagger_{timestamp}")


def run_training_round(
    *,
    base_training_config: Mapping[str, Any],
    dataset_root: Path,
    repo_id: str,
    pretrained_path: Path | None,
    disable_push_to_hub: bool,
    train_output_root: Path,
    steps_override: int | None,
) -> Path:
    train_cfg = build_iteration_training_config(
        base_config=base_training_config,
        dataset_root=dataset_root,
        repo_id=repo_id,
        pretrained_path=pretrained_path,
        disable_push_to_hub=disable_push_to_hub,
        steps_override=steps_override,
    )

    temp_train_cfg_path = write_training_config(train_cfg)
    print(f"Training config for this round: {temp_train_cfg_path}")
    if pretrained_path is None:
        print("Fine-tuning starts from scratch for this round.")
    else:
        print(f"Fine-tuning warm-start path: {pretrained_path}")
    if steps_override is not None:
        print(f"Training steps for this round: {steps_override}")

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


@dataclass(frozen=True)
class LeRobotDaggerConfig:
    system: str
    experiment_config_path: Path
    lerobot_training_config_path: Path
    repo_id: str
    dataset_root: Path
    start_with_aggregation: bool
    planner_name: str
    dagger_iterations: int
    trajectories_per_iteration: int
    steps_per_trajectory: int
    action_noise_std: float
    expert_mix_beta_start: float
    expert_mix_beta_end: float
    expert_mix_beta_decay_rate: float | None
    expert_mix_decay_after_success_rate: float | None
    adaptive_beta_recovery: bool
    train_output_root: Path
    seed: int
    policy_type: str | None
    initial_pretrained_path: Path | None
    disable_push_to_hub: bool
    target_epochs_per_round: float
    eval_episodes: int
    eval_steps: int | None
    eval_seed_start: int
    eval_action_noise_std: float
    max_train_steps: int | None


def dataset_frame_count(repo_id: str, dataset_root: Path) -> int:
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    return len(dataset)


def run_lerobot_dagger(cfg: LeRobotDaggerConfig) -> None:
    if cfg.dagger_iterations < 0:
        raise ValueError("'dagger_iterations' must be non-negative.")
    if cfg.trajectories_per_iteration <= 0:
        raise ValueError("'trajectories_per_iteration' must be positive.")
    if cfg.steps_per_trajectory <= 0:
        raise ValueError("'steps_per_trajectory' must be positive.")
    if cfg.action_noise_std < 0:
        raise ValueError("'action_noise_std' must be non-negative.")
    if not (0.0 <= cfg.expert_mix_beta_start <= 1.0):
        raise ValueError("'expert_mix_beta_start' must be in [0, 1].")
    if not (0.0 <= cfg.expert_mix_beta_end <= 1.0):
        raise ValueError("'expert_mix_beta_end' must be in [0, 1].")
    if cfg.expert_mix_beta_decay_rate is not None and cfg.expert_mix_beta_decay_rate < 0:
        raise ValueError("'expert_mix_beta_decay_rate' must be non-negative when provided.")
    if cfg.expert_mix_decay_after_success_rate is not None:
        if not (0.0 <= cfg.expert_mix_decay_after_success_rate <= 1.0):
            raise ValueError("'expert_mix_decay_after_success_rate' must be in [0, 1] when provided.")
    if cfg.target_epochs_per_round <= 0:
        raise ValueError("'target_epochs_per_round' must be positive.")
    if cfg.eval_episodes < 0:
        raise ValueError("'eval_episodes' must be non-negative.")
    if cfg.eval_steps is not None and cfg.eval_steps <= 0:
        raise ValueError("'eval_steps' must be positive when provided.")
    if cfg.eval_action_noise_std < 0:
        raise ValueError("'eval_action_noise_std' must be non-negative.")
    if cfg.max_train_steps is not None and cfg.max_train_steps <= 0:
        raise ValueError("'max_train_steps' must be positive when provided.")

    if not cfg.dataset_root.exists() and not cfg.start_with_aggregation:
        raise FileNotFoundError(f"Dataset root does not exist: {cfg.dataset_root}")

    set_seed(cfg.seed)
    device = get_inference_device()
    print(f"Running LeRobot DAgger on device: {device}")
    print(f"Aggregation action noise std: {cfg.action_noise_std:.6f}")
    print(f"Evaluation action noise std: {cfg.eval_action_noise_std:.6f}")
    if cfg.expert_mix_beta_decay_rate is not None:
        print(
            "Expert execution mixing schedule: "
            f"beta_start={cfg.expert_mix_beta_start:.3f}, "
            f"beta_decay_rate={cfg.expert_mix_beta_decay_rate:.3f}/round, "
            f"beta_floor=0.000, "
            f"decay_after_eval_success={cfg.expert_mix_decay_after_success_rate if cfg.expert_mix_decay_after_success_rate is not None else 'none'}"
        )
    else:
        print(
            "Expert execution mixing schedule: "
            f"beta_start={cfg.expert_mix_beta_start:.3f}, "
            f"beta_end={cfg.expert_mix_beta_end:.3f}, "
            f"decay_rounds={cfg.dagger_iterations}, "
            f"decay_after_eval_success={cfg.expert_mix_decay_after_success_rate if cfg.expert_mix_decay_after_success_rate is not None else 'none'}"
        )

    validated_experiment_config = load_and_validate_system_config(
        system_name=cfg.system,
        config_path=str(cfg.experiment_config_path),
    )
    seeded_experiment_config = with_seeded_initial_state_config(
        system_name=cfg.system,
        config=validated_experiment_config,
        base_seed=cfg.seed,
    )
    action_noise_seed = default_action_noise_seed_for_config(seeded_experiment_config)
    initial_state_seed = resolve_initial_state_seed(seeded_experiment_config, cfg.seed)
    print(f"Action noise seed: {action_noise_seed}")

    base_training_config = read_training_config(cfg.lerobot_training_config_path)
    base_batch_size = int(base_training_config.get("batch_size", 64))
    if base_batch_size <= 0:
        raise ValueError("LeRobot training config must define a positive 'batch_size'.")

    initial_round_steps = int(base_training_config.get("steps", 0))
    if not cfg.start_with_aggregation and initial_round_steps <= 0:
        raise ValueError(
            "LeRobot training config must define a positive 'steps' value for round-0 training."
        )

    print(
        "Epoch-target schedule: "
        f"target_epochs={cfg.target_epochs_per_round:.2f}, "
        f"max={cfg.max_train_steps if cfg.max_train_steps is not None else 'none'}"
    )
    if cfg.start_with_aggregation:
        print("Fresh DAgger mode: collecting round-0 data before any offline pretraining.")
    else:
        print(
            "Initial round schedule: "
            f"fixed optimizer_steps from config={initial_round_steps}"
        )

    policy_type = infer_policy_type(cfg.policy_type, base_training_config)
    uninitialized_policy_config = load_policy_config_from_training_config(
        training_config_path=cfg.lerobot_training_config_path,
        policy_type=policy_type,
    )
    if uninitialized_policy_config.type != policy_type:
        raise ValueError(
            "Policy type mismatch between inferred CLI/base config and parsed LeRobot config: "
            f"expected '{policy_type}', got '{uninitialized_policy_config.type}'."
        )
    if not cfg.start_with_aggregation:
        initial_num_frames = dataset_frame_count(repo_id=cfg.repo_id, dataset_root=cfg.dataset_root)
        initial_epochs = (
            float(initial_round_steps) * float(base_batch_size) / float(initial_num_frames)
        )
        print(
            "Initial round dataset schedule: "
            f"frames={initial_num_frames}, optimizer_steps={initial_round_steps}, "
            f"approx_epochs={initial_epochs:.2f}"
        )

    rollout_schema_simulator = DynamicsFactory.create(system_name=cfg.system, config=seeded_experiment_config)
    rollout_observation_feature_names = observation_feature_names(rollout_schema_simulator)
    rollout_observation_feature_cache = build_observation_feature_pack_cache(
        rollout_schema_simulator,
        rollout_observation_feature_names,
    )

    def evaluate_trained_policy(
        label: str,
        model_dir: Path,
    ) -> DaggerEvalMetrics | None:
        if cfg.eval_episodes == 0:
            return None

        eval_simulator = DynamicsFactory.create(system_name=cfg.system, config=seeded_experiment_config)
        eval_steps = cfg.eval_steps if cfg.eval_steps is not None else cfg.steps_per_trajectory
        eval_policy = load_lerobot_policy(
            policy_type=policy_type,
            model_dir=model_dir,
            device=device,
        )
        eval_preprocessor, eval_postprocessor = load_lerobot_pre_post_processors(
            policy_cfg=PreTrainedConfig.from_pretrained(str(model_dir)),
            device=device,
            pretrained_path=model_dir,
        )

        def action_fn(observation: np.ndarray) -> np.ndarray:
            policy_input = create_policy_input(
                simulator=eval_simulator,
                observation=observation,
                observation_feature_names_list=rollout_observation_feature_names,
                device=device,
                add_batch_dim=False,
                observation_feature_cache=rollout_observation_feature_cache,
            )
            policy_input = eval_preprocessor(policy_input)
            with torch.inference_mode():
                action_tensor = eval_policy.select_action(policy_input)
            action_tensor = eval_postprocessor(action_tensor)
            return action_tensor.squeeze(0).cpu().numpy()

        reset_fn = eval_policy.reset if hasattr(eval_policy, "reset") else None
        metrics = evaluate_policy_rollouts(
            simulator=eval_simulator,
            num_episodes=cfg.eval_episodes,
            num_steps=eval_steps,
            seed_start=cfg.eval_seed_start,
            action_fn=action_fn,
            reset_fn=reset_fn,
            action_noise_std=cfg.eval_action_noise_std,
            action_noise_seed=action_noise_seed,
        )
        if metrics is None:
            return None
        print_rollout_metrics(label=label, prefix="eval", metrics=metrics)
        del eval_policy
        gc.collect()
        return metrics

    previous_pretrained_path: Path | None = cfg.initial_pretrained_path
    initial_eval_success_rate: float | None = None
    if not cfg.start_with_aggregation:
        print("\n=== Initial LeRobot training pass (no aggregation) ===")
        previous_pretrained_path = run_training_round(
            base_training_config=base_training_config,
            dataset_root=cfg.dataset_root,
            repo_id=cfg.repo_id,
            pretrained_path=cfg.initial_pretrained_path,
            disable_push_to_hub=cfg.disable_push_to_hub,
            train_output_root=cfg.train_output_root,
            steps_override=initial_round_steps,
        )
        initial_eval_metrics = evaluate_trained_policy(
            label="Round 0 evaluation",
            model_dir=previous_pretrained_path,
        )
        if initial_eval_metrics is not None:
            initial_eval_success_rate = initial_eval_metrics.success_rate
        if cfg.dagger_iterations == 0:
            print("No DAgger refinements requested (--dagger-iterations 0).")
            print("\nLeRobot DAgger loop complete.")
            return

        round_indices = range(1, cfg.dagger_iterations + 1)
    else:
        if cfg.dagger_iterations == 0:
            raise ValueError(
                "Fresh DAgger mode requires at least one aggregation round; "
                "set --dagger-iterations to a positive value."
            )
        round_indices = range(0, cfg.dagger_iterations)

    beta_controller = ExpertMixBetaController(
        beta_start=cfg.expert_mix_beta_start,
        beta_end=cfg.expert_mix_beta_end,
        decay_rounds=max(1, cfg.dagger_iterations),
        beta_decay_rate=cfg.expert_mix_beta_decay_rate,
        decay_after_success_rate=cfg.expert_mix_decay_after_success_rate,
        adaptive_recovery=cfg.adaptive_beta_recovery,
    )

    if cfg.expert_mix_decay_after_success_rate is not None and initial_eval_success_rate is not None:
        beta_controller.prime_from_evaluation(initial_eval_success_rate)

    for round_index in round_indices:
        if cfg.start_with_aggregation:
            print(f"\n=== DAgger round {round_index + 1}/{cfg.dagger_iterations} ===")
        else:
            print(f"\n=== DAgger refinement {round_index}/{cfg.dagger_iterations} ===")

        simulator = DynamicsFactory.create(system_name=cfg.system, config=seeded_experiment_config)
        expert_planner = PlannerFactory.create(
            planner_name=cfg.planner_name,
            simulator=simulator,
            config=seeded_experiment_config,
        )

        round_beta = beta_controller.current_beta

        print(
            "Aggregation execution policy: "
            f"expert_beta={round_beta:.3f}, "
            f"decay_active={'yes' if beta_controller.decay_active else 'no'}"
        )

        policy: DiffusionPolicy | ACTPolicy | None = None
        policy_preprocessor = None
        policy_postprocessor = None
        if previous_pretrained_path is None:
            policy = build_uninitialized_lerobot_policy(
                policy_type=policy_type,
                policy_config=uninitialized_policy_config,
                device=device,
            )
            policy_preprocessor, policy_postprocessor = load_lerobot_pre_post_processors(
                policy_cfg=uninitialized_policy_config,
                device=device,
                pretrained_path=None,
            )
            print(
                "No pretrained checkpoint yet; using the uninitialized policy and matching processors "
                "for fresh-mode round-0 aggregation."
            )
        else:
            policy = load_lerobot_policy(
                policy_type=policy_type,
                model_dir=previous_pretrained_path,
                device=device,
            )
            policy_preprocessor, policy_postprocessor = load_lerobot_pre_post_processors(
                policy_cfg=PreTrainedConfig.from_pretrained(str(previous_pretrained_path)),
                device=device,
                pretrained_path=previous_pretrained_path,
            )

        if cfg.start_with_aggregation and not cfg.dataset_root.exists():
            dataset_writer = LeRobotDataset.create(
                repo_id=cfg.repo_id,
                fps=int(1 / simulator.dt),
                root=cfg.dataset_root,
                features=simulator.get_dataset_features(),
            )
        else:
            dataset_writer = LeRobotDataset.resume(repo_id=cfg.repo_id, root=cfg.dataset_root)
        try:
            policy_action_fn = None
            policy_reset_fn = None
            if policy is not None and policy_preprocessor is not None and policy_postprocessor is not None:
                def policy_action_fn(observation: np.ndarray) -> np.ndarray:
                    policy_input = create_policy_input(
                        simulator=simulator,
                        observation=observation,
                        observation_feature_names_list=rollout_observation_feature_names,
                        device=device,
                        add_batch_dim=False,
                        observation_feature_cache=rollout_observation_feature_cache,
                    )
                    policy_input = policy_preprocessor(policy_input)
                    with torch.inference_mode():
                        action_tensor = policy.select_action(policy_input)
                    action_tensor = policy_postprocessor(action_tensor)
                    return action_tensor.squeeze(0).cpu().numpy()

                policy_reset_fn = policy.reset if hasattr(policy, "reset") else None

            aggregation_metrics = collect_dagger_rollouts(
                simulator=simulator,
                expert_planner=expert_planner,
                dataset_writer=dataset_writer,
                trajectories_per_iteration=cfg.trajectories_per_iteration,
                steps_per_trajectory=cfg.steps_per_trajectory,
                action_noise_std=cfg.action_noise_std,
                action_noise_seed=action_noise_seed,
                initial_state_seed=initial_state_seed,
                expert_mixing_beta=round_beta,
                policy_action_fn=policy_action_fn,
                policy_reset_fn=policy_reset_fn,
            )
        finally:
            dataset_writer.finalize()
            del dataset_writer
            if policy is not None:
                del policy
            gc.collect()

        print_rollout_metrics(
            label=(
                f"Round {round_index + 1} aggregation"
                if cfg.start_with_aggregation
                else f"Refinement {round_index} aggregation"
            ),
            prefix="aggregation",
            metrics=aggregation_metrics,
        )

        refinement_num_frames = dataset_frame_count(repo_id=cfg.repo_id, dataset_root=cfg.dataset_root)
        refinement_steps, refinement_epochs = resolve_round_steps(
            num_frames=refinement_num_frames,
            batch_size=base_batch_size,
            target_epochs_per_round=cfg.target_epochs_per_round,
            max_train_steps=cfg.max_train_steps,
        )
        print(
            "Refinement dataset schedule: "
            f"frames={refinement_num_frames}, optimizer_steps={refinement_steps}, "
            f"approx_epochs={refinement_epochs:.2f}"
        )

        previous_pretrained_path = run_training_round(
            base_training_config=base_training_config,
            dataset_root=cfg.dataset_root,
            repo_id=cfg.repo_id,
            pretrained_path=previous_pretrained_path,
            disable_push_to_hub=cfg.disable_push_to_hub,
            train_output_root=cfg.train_output_root,
            steps_override=refinement_steps,
        )
        eval_metrics = evaluate_trained_policy(
            label=(
                f"Round {round_index + 1} evaluation"
                if cfg.start_with_aggregation
                else f"Refinement {round_index} evaluation"
            ),
            model_dir=previous_pretrained_path,
        )
        beta_controller.update_after_evaluation(
            eval_metrics.success_rate if eval_metrics is not None else None
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
        "--expert-config",
        dest="expert_config",
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
        default=None,
        help=(
            "LeRobot dataset repository id stored in metadata. If both --repo-id and --dataset-root "
            "are omitted, fresh DAgger mode auto-creates them"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "path to local LeRobot dataset root. If both --dataset-root and --repo-id are omitted, "
            "fresh DAgger mode auto-creates them and starts with aggregation"
        ),
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
        "--action-noise-std",
        type=float,
        default=0.0,
        help=(
            "std-dev of Gaussian action noise applied during DAgger rollout execution; "
            "expert labels remain noise-free"
        ),
    )
    parser.add_argument(
        "--expert-mix-beta-start",
        type=float,
        default=0.8,
        help=(
            "initial probability of executing the expert action during aggregation rollouts; "
            "set start=end to disable decay, set start=end=0.0 for pre-mixing behavior"
        ),
    )
    parser.add_argument(
        "--expert-mix-beta-end",
        type=float,
        default=0.0,
        help="final expert-action probability at the last DAgger aggregation round",
    )
    parser.add_argument(
        "--expert-mix-beta-decay-rate",
        type=float,
        default=None,
        help=(
            "optional additive expert-mix decay per aggregation round (beta_t = max(0, beta_start - rate*t)); "
            "when set, this overrides --expert-mix-beta-end"
        ),
    )
    parser.add_argument(
        "--expert-mix-decay-after-success-rate",
        type=float,
        default=None,
        help=(
            "optional gate: start beta decay only after latest eval success_rate exceeds this threshold "
            "(e.g. 0.0)"
        ),
    )
    parser.add_argument(
        "--adaptive-beta-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "toggle adaptive recovery on eval regressions: when enabled, beta is increased by one "
            "schedule step after a success-rate drop; when disabled, beta follows a "
            "monotonic schedule"
        ),
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
        default=99,
        help="random seed",
    )
    parser.add_argument(
        "--target-epochs-per-round",
        type=float,
        default=30.0,
        help="target number of dataset epochs to train per round",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="number of seeded rollouts used for in-loop policy evaluation after each retrain",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="maximum rollout steps used for in-loop evaluation (defaults to --steps-per-trajectory)",
    )
    parser.add_argument(
        "--eval-seed-start",
        type=int,
        default=10000,
        help="first deterministic seed used to generate evaluation initial states",
    )
    parser.add_argument(
        "--eval-action-noise-std",
        type=float,
        default=0.0,
        help="std-dev of Gaussian action noise applied during in-loop evaluation rollouts",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help=(
            "optional upper bound on per-round training steps"
        ),
    )
    parser.add_argument(
        "--allow-push-to-hub",
        action="store_true",
        help="if set, keep push_to_hub behavior from LeRobot train config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (args.repo_id is None) != (args.dataset_root is None):
        raise ValueError("Provide both --repo-id and --dataset-root together, or omit both for fresh DAgger mode.")

    if args.repo_id is None and args.dataset_root is None:
        timestamp = time.time_ns()
        repo_id = default_repo_id_for_system(args.system, timestamp)
        dataset_root = default_dataset_root_for_system(args.system, timestamp)
        start_with_aggregation = True
    else:
        repo_id = str(args.repo_id)
        dataset_root = Path(args.dataset_root)
        start_with_aggregation = False

    cfg = LeRobotDaggerConfig(
        system=args.system,
        experiment_config_path=args.expert_config,
        lerobot_training_config_path=args.lerobot_train_config,
        repo_id=repo_id,
        dataset_root=dataset_root,
        start_with_aggregation=start_with_aggregation,
        planner_name=args.planner,
        dagger_iterations=args.dagger_iterations,
        trajectories_per_iteration=args.trajectories_per_iteration,
        steps_per_trajectory=args.steps_per_trajectory,
        action_noise_std=args.action_noise_std,
        expert_mix_beta_start=args.expert_mix_beta_start,
        expert_mix_beta_end=args.expert_mix_beta_end,
        expert_mix_beta_decay_rate=args.expert_mix_beta_decay_rate,
        expert_mix_decay_after_success_rate=args.expert_mix_decay_after_success_rate,
        adaptive_beta_recovery=args.adaptive_beta_recovery,
        train_output_root=args.train_output_root,
        seed=args.seed,
        policy_type=args.policy_type,
        initial_pretrained_path=args.initial_pretrained_path,
        disable_push_to_hub=not args.allow_push_to_hub,
        target_epochs_per_round=args.target_epochs_per_round,
        eval_episodes=args.eval_episodes,
        eval_steps=args.eval_steps,
        eval_seed_start=args.eval_seed_start,
        eval_action_noise_std=args.eval_action_noise_std,
        max_train_steps=args.max_train_steps,
    )

    run_lerobot_dagger(cfg)


if __name__ == "__main__":
    main()
