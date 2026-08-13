from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import (
    load_and_validate_mlp_architecture_config,
    load_and_validate_system_config,
)
from core.factory import DynamicsFactory, PlannerFactory
from learning.dagger import (
    DaggerEvalMetrics,
    ExpertMixBetaController,
    action_feature_names,
    apply_execution_noise,
    collect_dagger_rollouts,
    evaluate_policy_rollouts,
    observation_dim_from_features,
    observation_feature_names,
    pack_observation_features,
    print_rollout_metrics,
    resolve_round_steps,
    set_seed,
    with_seeded_initial_state_config,
)
from learning.models.mlp import MLPPolicy
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol
from systems.seed_utils import (
    default_action_noise_seed_for_config,
)


DEFAULT_MLP_HIDDEN_DIMS: tuple[int, ...] = (256, 256, 128)


def get_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flatten_observation_for_policy(
    simulator: DynamicsProtocol,
    observation: np.ndarray,
    observation_feature_names_list: list[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Pack runtime observation exactly like dataset frames, then flatten for MLP.
    """
    packed_features = pack_observation_features(
        simulator=simulator,
        observation=observation,
        feature_names=observation_feature_names_list,
    )

    chunks: list[torch.Tensor] = []
    for feature_name in observation_feature_names_list:
        chunks.append(
            torch.as_tensor(packed_features[feature_name], dtype=torch.float32, device=device).view(-1)
        )

    return torch.cat(chunks, dim=0).unsqueeze(0)


class LeRobotMLPDataset(Dataset):
    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        obs_feature_names: list[str],
        action_feature_names: list[str],
    ):
        self.dataset = lerobot_dataset
        self.obs_feature_names = obs_feature_names
        self.action_feature_names = action_feature_names

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        obs_parts = [sample[name].float().view(-1) for name in self.obs_feature_names]
        observation = torch.cat(obs_parts, dim=0)
        action_parts = [sample[name].float().view(-1) for name in self.action_feature_names]
        action = torch.cat(action_parts, dim=0)
        return observation, action


@dataclass(frozen=True)
class DaggerConfig:
    system: str
    experiment_config: Mapping[str, Any]
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
    target_epochs_per_round: float
    eval_episodes: int
    eval_steps: int | None
    eval_seed_start: int
    eval_action_noise_std: float
    batch_size: int
    learning_rate: float
    mlp_hidden_dims: tuple[int, ...]
    checkpoint_dir: Path
    seed: int
    max_train_steps: int | None


def load_mlp_hidden_dims(mlp_config_path: Path | None) -> tuple[int, ...]:
    if mlp_config_path is None:
        return DEFAULT_MLP_HIDDEN_DIMS

    validated = load_and_validate_mlp_architecture_config(mlp_config_path)
    return validated.hidden_dims


def default_checkpoint_dir_for_system(system: str) -> Path:
    if system == "multi_robot":
        return Path("outputs/train_dagger_multi_robot")
    return Path("outputs/train_dagger")


def default_repo_id_for_system(system: str, timestamp: int) -> str:
    return f"local/{system}_dagger_{timestamp}"


def default_dataset_root_for_system(system: str, timestamp: int) -> Path:
    return Path(f"data/lerobot_dataset_{system}_dagger_{timestamp}")


def train_policy_steps(
    policy: MLPPolicy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int,
) -> float:
    if num_steps <= 0:
        raise ValueError("'num_steps' must be positive.")

    policy.train()
    mse_loss = nn.MSELoss()
    running_loss = 0.0
    data_iterator = iter(dataloader)
    progress = None
    if tqdm is not None:
        progress = tqdm(
            range(1, num_steps + 1),
            desc="Train steps",
            unit="step",
            leave=False,
            dynamic_ncols=True,
        )
        step_iterator = progress
    else:
        step_iterator = range(1, num_steps + 1)
        print("tqdm not installed; showing periodic step progress.")

    for step in step_iterator:
        try:
            observations, actions = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            observations, actions = next(data_iterator)

        observations = observations.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()
        predictions = policy(observations)
        loss = mse_loss(predictions, actions)
        loss.backward()
        optimizer.step()

        step_loss = float(loss.item())
        running_loss += step_loss
        running_mean = running_loss / float(step)
        if progress is not None:
            progress.set_postfix(loss=f"{step_loss:.6f}", mean=f"{running_mean:.6f}")
        elif step == 1 or step == num_steps or step % max(1, num_steps // 10) == 0:
            print(
                f"  step {step}/{num_steps} "
                f"loss={step_loss:.6f} mean_loss={running_mean:.6f}"
            )

    if progress is not None:
        progress.close()

    return running_loss / float(num_steps)


def run_dagger(cfg: DaggerConfig) -> None:
    set_seed(cfg.seed)

    seeded_experiment_config = with_seeded_initial_state_config(
        system_name=cfg.system,
        config=cfg.experiment_config,
        base_seed=cfg.seed,
    )

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
    if cfg.batch_size <= 0:
        raise ValueError("'batch_size' must be positive.")
    if cfg.learning_rate <= 0:
        raise ValueError("'learning_rate' must be positive.")
    if cfg.max_train_steps is not None and cfg.max_train_steps <= 0:
        raise ValueError("'max_train_steps' must be positive when provided.")

    if not cfg.dataset_root.exists() and not cfg.start_with_aggregation:
        raise FileNotFoundError(f"Dataset root does not exist: {cfg.dataset_root}")

    simulator = DynamicsFactory.create(system_name=cfg.system, config=seeded_experiment_config)
    device = get_training_device()
    action_noise_seed = default_action_noise_seed_for_config(seeded_experiment_config)

    obs_feature_names = observation_feature_names(simulator)
    act_feature_names = action_feature_names(simulator)
    if len(obs_feature_names) == 0:
        raise ValueError("No observation features found in simulator dataset schema.")
    if len(act_feature_names) == 0:
        raise ValueError("No action features found in simulator dataset schema.")

    state_dim = observation_dim_from_features(simulator)
    action_dim = int(simulator.nu)

    policy = MLPPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=cfg.mlp_hidden_dims,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Starting DAgger training")
    print(f"Device: {device}")
    print(f"Initial dataset root: {cfg.dataset_root}")
    print(f"Aggregation action noise std: {cfg.action_noise_std:.6f}")
    print(f"Evaluation action noise std: {cfg.eval_action_noise_std:.6f}")
    print(f"Action noise seed: {action_noise_seed}")
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
    print(f"MLP hidden dims: {list(cfg.mlp_hidden_dims)}")
    if cfg.start_with_aggregation:
        print("Fresh DAgger mode: collecting round-0 data before any offline pretraining.")
    else:
        print("Initial offline training pass starts from the current expert dataset.")

    print(
        "Epoch-target schedule: "
        f"target_epochs={cfg.target_epochs_per_round:.2f}, "
        f"max={cfg.max_train_steps if cfg.max_train_steps is not None else 'none'}"
    )

    def train_on_aggregate(label: str, training_round: int) -> None:
        print(f"\n=== {label} ===")

        aggregate_dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.dataset_root)
        train_dataset = LeRobotMLPDataset(
            lerobot_dataset=aggregate_dataset,
            obs_feature_names=obs_feature_names,
            action_feature_names=act_feature_names,
        )

        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(int(cfg.seed) + int(training_round))
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            generator=dataloader_generator,
        )

        print(f"Training on {len(train_dataset)} aggregated frames")
        round_steps, approx_epochs = resolve_round_steps(
            num_frames=len(train_dataset),
            batch_size=cfg.batch_size,
            target_epochs_per_round=cfg.target_epochs_per_round,
            max_train_steps=cfg.max_train_steps,
        )
        print(
            f"  optimizer_steps={round_steps} "
            f"(~{approx_epochs:.2f} epochs at batch_size={cfg.batch_size})"
        )
        step_loss = train_policy_steps(
            policy=policy,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            num_steps=round_steps,
        )
        print(f"  mean_step_loss={step_loss:.6f}")

    def evaluate_current_policy(label: str) -> DaggerEvalMetrics | None:
        if cfg.eval_episodes == 0:
            return None

        eval_simulator = DynamicsFactory.create(system_name=cfg.system, config=seeded_experiment_config)
        eval_steps = cfg.eval_steps if cfg.eval_steps is not None else cfg.steps_per_trajectory

        def action_fn(observation: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                model_input = flatten_observation_for_policy(
                    eval_simulator,
                    observation,
                    observation_feature_names_list=obs_feature_names,
                    device=device,
                )
                action = policy.select_action(model_input).squeeze(0).cpu().numpy()
            return action

        metrics = evaluate_policy_rollouts(
            simulator=eval_simulator,
            num_episodes=cfg.eval_episodes,
            num_steps=eval_steps,
            seed_start=cfg.eval_seed_start,
            action_fn=action_fn,
            reset_fn=None,
            action_noise_std=cfg.eval_action_noise_std,
            action_noise_seed=action_noise_seed,
        )
        if metrics is None:
            return None
        print_rollout_metrics(label=label, prefix="eval", metrics=metrics)
        return metrics

    def save_checkpoints(training_round: int) -> None:
        checkpoint_data = {
            "iteration": training_round,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dims": list(cfg.mlp_hidden_dims),
            "obs_feature_names": obs_feature_names,
            "system": cfg.system,
        }

        latest_checkpoint = cfg.checkpoint_dir / "mlp_dagger_checkpoint.pt"
        iteration_checkpoint = cfg.checkpoint_dir / f"mlp_dagger_iter_{training_round:03d}.pt"
        torch.save(checkpoint_data, latest_checkpoint)
        torch.save(checkpoint_data, iteration_checkpoint)
        print(f"Saved checkpoints: {latest_checkpoint} and {iteration_checkpoint}")

    initial_eval_success_rate: float | None = None

    if not cfg.start_with_aggregation:
        train_on_aggregate("Initial offline training pass", training_round=0)
        last_eval_metrics = evaluate_current_policy("Round 0 evaluation")
        if last_eval_metrics is not None:
            initial_eval_success_rate = last_eval_metrics.success_rate
        save_checkpoints(training_round=0)

        if cfg.dagger_iterations == 0:
            print("No DAgger refinements requested (--dagger-iterations 0).")
            return

        refinement_round_indices = range(1, cfg.dagger_iterations + 1)
    else:
        if cfg.dagger_iterations == 0:
            raise ValueError(
                "Fresh DAgger mode requires at least one aggregation round; "
                "set --dagger-iterations to a positive value."
            )
        refinement_round_indices = range(0, cfg.dagger_iterations)

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

    for training_round in refinement_round_indices:
        if cfg.start_with_aggregation:
            print(
                f"\n=== DAgger round {training_round + 1}/{cfg.dagger_iterations}: aggregate ==="
            )
        else:
            print(
                f"\n=== DAgger refinement {training_round}/{cfg.dagger_iterations}: aggregate ==="
            )

        simulator_for_rollout = DynamicsFactory.create(system_name=cfg.system, config=seeded_experiment_config)
        expert_planner = PlannerFactory.create(
            planner_name=cfg.planner_name,
            simulator=simulator_for_rollout,
            config=seeded_experiment_config,
        )

        round_beta = beta_controller.current_beta

        print(
            "Aggregation execution policy: "
            f"expert_beta={round_beta:.3f}, "
            f"decay_active={'yes' if beta_controller.decay_active else 'no'}"
        )

        if cfg.start_with_aggregation and not cfg.dataset_root.exists():
            dataset_writer = LeRobotDataset.create(
                repo_id=cfg.repo_id,
                fps=int(1 / simulator_for_rollout.dt),
                root=cfg.dataset_root,
                features=simulator_for_rollout.get_dataset_features(),
            )
        else:
            dataset_writer = LeRobotDataset.resume(repo_id=cfg.repo_id, root=cfg.dataset_root)

        try:
            def policy_action_fn(observation: np.ndarray) -> np.ndarray:
                with torch.inference_mode():
                    model_input = flatten_observation_for_policy(
                        simulator_for_rollout,
                        observation,
                        observation_feature_names_list=obs_feature_names,
                        device=device,
                    )
                    return policy.select_action(model_input).squeeze(0).cpu().numpy()

            aggregation_metrics = collect_dagger_rollouts(
                simulator=simulator_for_rollout,
                expert_planner=expert_planner,
                dataset_writer=dataset_writer,
                trajectories_per_iteration=cfg.trajectories_per_iteration,
                steps_per_trajectory=cfg.steps_per_trajectory,
                action_noise_std=cfg.action_noise_std,
                action_noise_seed=action_noise_seed,
                initial_state_seed=cfg.seed,
                expert_mixing_beta=round_beta,
                policy_action_fn=policy_action_fn,
            )
        finally:
            # LeRobot writes parquet chunks lazily; finalize guarantees readable footers.
            dataset_writer.finalize()
            del dataset_writer
            gc.collect()

        print_rollout_metrics(
            label=f"Round {training_round + 1} aggregation" if cfg.start_with_aggregation else f"Refinement {training_round} aggregation",
            prefix="aggregation",
            metrics=aggregation_metrics,
        )

        if cfg.start_with_aggregation:
            train_on_aggregate(
                f"DAgger round {training_round + 1}/{cfg.dagger_iterations}: retrain",
                training_round=training_round,
            )
            eval_metrics = evaluate_current_policy(f"Round {training_round + 1} evaluation")
        else:
            train_on_aggregate(
                f"DAgger refinement {training_round}/{cfg.dagger_iterations}: retrain",
                training_round=training_round,
            )
            eval_metrics = evaluate_current_policy(f"Refinement {training_round} evaluation")

        beta_controller.update_after_evaluation(
            eval_metrics.success_rate if eval_metrics is not None else None
        )
        save_checkpoints(training_round=training_round)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom MLP policy with DAgger")
    parser.add_argument(
        "--system",
        type=str.lower,
        choices=DynamicsFactory.names(),
        required=True,
        help="name of system class, e.g. single_integrator, unicycle2, ...",
    )
    parser.add_argument(
        "--expert-config",
        dest="expert_config",
        type=str,
        required=True,
        help="path to simulator/planner experiment YAML config",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="LeRobot dataset repository id stored in metadata (e.g. local/double_integrator_casadi_expert)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "path to local LeRobot dataset root. If both --dataset-root and --repo-id are omitted, "
            "fresh DAgger mode auto-creates them and starts with aggregation from a randomly initialized policy"
        ),
    )
    parser.add_argument(
        "--planner",
        type=str.lower,
        default="casadi",
        choices=["casadi"],
        help="expert planner to query during DAgger rollouts",
    )
    parser.add_argument(
        "--dagger-iterations",
        type=int,
        default=4,
        help=(
            "number of DAgger refinement rounds (aggregate then retrain). "
            "Use 0 for pure offline training"
        ),
    )
    parser.add_argument(
        "--trajectories-per-iteration",
        type=int,
        default=20,
        help="number of learner rollouts to aggregate per DAgger iteration",
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
            "set start=end to disable decay, set start=end=0.0 for policy only behavior"
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
        default=False,
        help=(
            "toggle adaptive recovery on eval regressions: when enabled, beta is increased by one "
            "schedule step after a success-rate drop; when disabled (default), beta follows a "
            "monotonic schedule"
        ),
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
        "--batch-size",
        type=int,
        default=64,
        help="mini-batch size for MLP training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--mlp-config",
        type=Path,
        default=None,
        help=(
            "optional YAML config for MLP architecture; expected key 'model.hidden_dims' "
            "(or top-level 'hidden_dims'), e.g. [512, 256, 128]"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "directory where DAgger checkpoints are written "
            "(defaults to outputs/train_dagger for single-robot systems and "
            "outputs/train_dagger_multi_robot for multi_robot)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="random seed",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="optional upper bound on per-round optimizer steps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validated_config = load_and_validate_system_config(
        system_name=args.system,
        config_path=args.expert_config,
    )
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = default_checkpoint_dir_for_system(args.system)
    mlp_hidden_dims = load_mlp_hidden_dims(args.mlp_config)

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

    cfg = DaggerConfig(
        system=args.system,
        experiment_config=validated_config,
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
        target_epochs_per_round=args.target_epochs_per_round,
        eval_episodes=args.eval_episodes,
        eval_steps=args.eval_steps,
        eval_seed_start=args.eval_seed_start,
        eval_action_noise_std=args.eval_action_noise_std,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mlp_hidden_dims=mlp_hidden_dims,
        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
        max_train_steps=args.max_train_steps,
    )

    run_dagger(cfg)


if __name__ == "__main__":
    main()
