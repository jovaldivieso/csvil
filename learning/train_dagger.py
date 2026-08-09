from __future__ import annotations

import argparse
import gc
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    tqdm = None

from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from learning.dagger_evaluation import (
    DaggerEvalMetrics,
    apply_execution_noise,
    evaluate_policy_rollouts,
)
from learning.models.mlp import MLPPolicy
from planning.casadi_planner import PlannerSolveError
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol
from systems.seed_utils import (
    action_noise_seed_for_rollout,
    default_action_noise_seed_for_config,
)


def get_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def observation_feature_names(simulator: DynamicsProtocol) -> list[str]:
    def is_observation_feature(name: str) -> bool:
        return name.startswith("observation.") or ".observation." in name

    return [
        feature_name
        for feature_name in simulator.get_dataset_features().keys()
        if is_observation_feature(feature_name)
    ]


def action_feature_names(simulator: DynamicsProtocol) -> list[str]:
    def is_action_feature(name: str) -> bool:
        return name == "action" or name.endswith(".action")

    return [
        feature_name
        for feature_name in simulator.get_dataset_features().keys()
        if is_action_feature(feature_name)
    ]


def observation_dim_from_features(simulator: DynamicsProtocol) -> int:
    def is_observation_feature(name: str) -> bool:
        return name.startswith("observation.") or ".observation." in name

    total_dim = 0
    for feature_name, feature_info in simulator.get_dataset_features().items():
        if is_observation_feature(feature_name):
            total_dim += int(feature_info["shape"][0])
    return total_dim


def flatten_observation_for_policy(
    simulator: DynamicsProtocol,
    observation: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Pack runtime observation exactly like dataset frames, then flatten for MLP.
    """
    features = simulator.get_dataset_features()

    def is_observation_feature(name: str) -> bool:
        return name.startswith("observation.") or ".observation." in name

    dummy_action = np.zeros(int(simulator.nu), dtype=np.float32)
    packed_frame = simulator.format_dataset_frame(observation, dummy_action)

    chunks: list[torch.Tensor] = []
    for feature_name in features.keys():
        if is_observation_feature(feature_name):
            if feature_name not in packed_frame:
                raise KeyError(
                    f"Missing observation feature '{feature_name}' in packed frame. "
                    "Check simulator.format_dataset_frame() and dataset schema alignment."
                )
            chunks.append(torch.as_tensor(packed_frame[feature_name], dtype=torch.float32, device=device).view(-1))

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
    planner_name: str
    dagger_iterations: int
    trajectories_per_iteration: int
    steps_per_trajectory: int
    action_noise_std: float
    target_epochs_per_round: float
    eval_episodes: int
    eval_steps: int | None
    eval_seed_start: int
    eval_action_noise_std: float
    batch_size: int
    learning_rate: float
    checkpoint_dir: Path
    seed: int
    max_train_steps: int | None


def default_checkpoint_dir_for_system(system: str) -> Path:
    if system == "multi_robot":
        return Path("outputs/train_dagger_multi_robot")
    return Path("outputs/train_dagger")


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


def resolve_round_steps(
    num_frames: int,
    batch_size: int,
    target_epochs_per_round: float,
    max_train_steps: int | None,
) -> tuple[int, float]:
    if num_frames <= 0:
        raise ValueError("Training dataset must contain at least one frame.")
    if batch_size <= 0:
        raise ValueError("'batch_size' must be positive.")
    if target_epochs_per_round <= 0:
        raise ValueError("'target_epochs_per_round' must be positive.")

    steps = math.ceil(float(target_epochs_per_round) * float(num_frames) / float(batch_size))
    if max_train_steps is not None:
        steps = min(steps, int(max_train_steps))
    if steps <= 0:
        raise ValueError("Per-round training steps must remain positive.")

    approx_epochs = float(steps) * float(batch_size) / float(num_frames)
    return steps, approx_epochs


def collect_dagger_data(
    simulator: DynamicsProtocol,
    expert_planner: PlannerProtocol,
    policy: MLPPolicy,
    dataset_writer: LeRobotDataset,
    trajectories_per_iteration: int,
    steps_per_trajectory: int,
    device: torch.device,
    action_noise_std: float,
    action_noise_seed: int,
) -> DaggerEvalMetrics:
    """
    Roll out learner policy, query expert at visited states, aggregate labels.
    """
    successful_episodes = 0
    attempted_episodes = 0
    max_attempts = max(trajectories_per_iteration * 3, trajectories_per_iteration)
    reached_goal_count = 0
    steps_taken: list[int] = []

    policy.eval()

    while successful_episodes < trajectories_per_iteration:
        attempted_episodes += 1
        if attempted_episodes > max_attempts:
            raise RuntimeError(
                "Too many failed DAgger rollout attempts. "
                f"Collected {successful_episodes}/{trajectories_per_iteration} episodes."
            )

        state = simulator.reset_random()
        episode_initial_state = state.copy()
        episode_noise_seed = action_noise_seed_for_rollout(
            action_noise_seed,
            rollout_index=attempted_episodes,
        )
        episode_action_noise_rng = np.random.default_rng(episode_noise_seed)
        planner_failed = False
        expert_planner.reset()

        reached_goal = False
        rollout_steps = steps_per_trajectory
        episode_frames: list[dict[str, object]] = []

        for step in range(1, steps_per_trajectory + 1):
            observation = simulator.observe(state)

            with torch.inference_mode():
                model_input = flatten_observation_for_policy(simulator, observation, device=device)
                policy_action = policy.select_action(model_input).squeeze(0).cpu().numpy()

            try:
                expert_action = expert_planner(observation)
            except PlannerSolveError as exc:
                print(
                    "Skipping episode due to planner failure "
                    f"(attempt={attempted_episodes}, step={step}, action_noise_std={action_noise_std:.6f}, "
                    f"noise_seed={episode_noise_seed})."
                )
                print(
                    "Planner failure context: "
                    f"initial_state={np.array2string(np.asarray(episode_initial_state), precision=6)}, "
                    f"current_state={np.array2string(np.asarray(state), precision=6)}, "
                    f"goal_state={np.array2string(np.asarray(simulator.goal_state), precision=6)}"
                )
                print(f"Underlying solver error: {exc}")
                planner_failed = True
                break

            frame_data = simulator.format_dataset_frame(observation, expert_action)
            frame_data["task"] = "reach target"
            episode_frames.append(frame_data)

            # Environment advances with noisy learner action; labels remain clean expert corrections.
            executed_action = apply_execution_noise(
                simulator=simulator,
                action=policy_action,
                action_noise_std=action_noise_std,
                rng=episode_action_noise_rng,
            )
            state = simulator.step(state, executed_action)

            if simulator.should_terminate_rollout(state):
                reached_goal = True
                rollout_steps = step
                break

        if planner_failed:
            continue

        for frame_data in episode_frames:
            dataset_writer.add_frame(frame_data)
        dataset_writer.save_episode()
        successful_episodes += 1
        reached_goal_count += int(reached_goal)
        steps_taken.append(int(rollout_steps))

        if successful_episodes % 10 == 0:
            print(
                "Collected "
                f"{successful_episodes}/{trajectories_per_iteration} trajectories"
            )

    return DaggerEvalMetrics(
        success_rate=float(reached_goal_count) / float(successful_episodes),
        mean_steps=float(np.mean(np.asarray(steps_taken, dtype=float))),
        min_steps=min(steps_taken),
        max_steps=max(steps_taken),
        num_episodes=successful_episodes,
    )


def run_dagger(cfg: DaggerConfig) -> None:
    set_seed(cfg.seed)

    if cfg.dagger_iterations < 0:
        raise ValueError("'dagger_iterations' must be non-negative.")
    if cfg.trajectories_per_iteration <= 0:
        raise ValueError("'trajectories_per_iteration' must be positive.")
    if cfg.steps_per_trajectory <= 0:
        raise ValueError("'steps_per_trajectory' must be positive.")
    if cfg.action_noise_std < 0:
        raise ValueError("'action_noise_std' must be non-negative.")
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

    if not cfg.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {cfg.dataset_root}")

    simulator = DynamicsFactory.create(system_name=cfg.system, config=cfg.experiment_config)
    device = get_training_device()
    action_noise_seed = default_action_noise_seed_for_config(cfg.experiment_config)

    obs_feature_names = observation_feature_names(simulator)
    act_feature_names = action_feature_names(simulator)
    if len(obs_feature_names) == 0:
        raise ValueError("No observation features found in simulator dataset schema.")
    if len(act_feature_names) == 0:
        raise ValueError("No action features found in simulator dataset schema.")

    state_dim = observation_dim_from_features(simulator)
    action_dim = int(simulator.nu)

    policy = MLPPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Starting DAgger training")
    print(f"Device: {device}")
    print(f"Initial dataset root: {cfg.dataset_root}")
    print(f"Aggregation action noise std: {cfg.action_noise_std:.6f}")
    print(f"Evaluation action noise std: {cfg.eval_action_noise_std:.6f}")
    print(f"Action noise seed: {action_noise_seed}")
    print("Initial offline training pass starts from the current expert dataset.")

    print(
        "Epoch-target schedule: "
        f"target_epochs={cfg.target_epochs_per_round:.2f}, "
        f"max={cfg.max_train_steps if cfg.max_train_steps is not None else 'none'}"
    )

    def train_on_aggregate(label: str) -> None:
        print(f"\n=== {label} ===")

        aggregate_dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.dataset_root)
        train_dataset = LeRobotMLPDataset(
            lerobot_dataset=aggregate_dataset,
            obs_feature_names=obs_feature_names,
            action_feature_names=act_feature_names,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
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

    def evaluate_current_policy(label: str) -> None:
        if cfg.eval_episodes == 0:
            return

        eval_simulator = DynamicsFactory.create(system_name=cfg.system, config=cfg.experiment_config)
        eval_steps = cfg.eval_steps if cfg.eval_steps is not None else cfg.steps_per_trajectory

        def action_fn(observation: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                model_input = flatten_observation_for_policy(eval_simulator, observation, device=device)
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
            return
        print_rollout_metrics(label=label, prefix="eval", metrics=metrics)

    def save_checkpoints(training_round: int) -> None:
        checkpoint_data = {
            "iteration": training_round,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "obs_feature_names": obs_feature_names,
            "system": cfg.system,
        }

        latest_checkpoint = cfg.checkpoint_dir / "mlp_dagger_checkpoint.pt"
        iteration_checkpoint = cfg.checkpoint_dir / f"mlp_dagger_iter_{training_round:03d}.pt"
        torch.save(checkpoint_data, latest_checkpoint)
        torch.save(checkpoint_data, iteration_checkpoint)
        print(f"Saved checkpoints: {latest_checkpoint} and {iteration_checkpoint}")

    train_on_aggregate("Initial offline training pass")
    evaluate_current_policy("Round 0 evaluation")
    save_checkpoints(training_round=0)

    if cfg.dagger_iterations == 0:
        print("No DAgger refinements requested (--dagger-iterations 0).")
        return

    for refinement in range(1, cfg.dagger_iterations + 1):
        print(f"\n=== DAgger refinement {refinement}/{cfg.dagger_iterations}: aggregate ===")

        simulator_for_rollout = DynamicsFactory.create(system_name=cfg.system, config=cfg.experiment_config)
        expert_planner = PlannerFactory.create(
            planner_name=cfg.planner_name,
            simulator=simulator_for_rollout,
            config=cfg.experiment_config,
        )
        dataset_writer = LeRobotDataset.resume(repo_id=cfg.repo_id, root=cfg.dataset_root)

        try:
            aggregation_metrics = collect_dagger_data(
                simulator=simulator_for_rollout,
                expert_planner=expert_planner,
                policy=policy,
                dataset_writer=dataset_writer,
                trajectories_per_iteration=cfg.trajectories_per_iteration,
                steps_per_trajectory=cfg.steps_per_trajectory,
                device=device,
                action_noise_std=cfg.action_noise_std,
                action_noise_seed=action_noise_seed,
            )
        finally:
            # LeRobot writes parquet chunks lazily; finalize guarantees readable footers.
            dataset_writer.finalize()
            del dataset_writer
            gc.collect()

        print_rollout_metrics(
            label=f"Refinement {refinement} aggregation",
            prefix="aggregation",
            metrics=aggregation_metrics,
        )

        train_on_aggregate(f"DAgger refinement {refinement}/{cfg.dagger_iterations}: retrain")
        evaluate_current_policy(f"Refinement {refinement} evaluation")
        save_checkpoints(training_round=refinement)


def print_rollout_metrics(label: str, prefix: str, metrics: DaggerEvalMetrics) -> None:
    print(
        f"{label}: {prefix}_success_rate={100.0 * metrics.success_rate:.1f}% "
        f"{prefix}_mean_steps={metrics.mean_steps:.2f} "
        f"{prefix}_min_steps={metrics.min_steps} {prefix}_max_steps={metrics.max_steps} "
        f"episodes={metrics.num_episodes}"
    )


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
        "--config",
        type=str,
        required=True,
        help="path to yaml config file for experiment",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobot dataset repository id stored in metadata (e.g. local/double_integrator_casadi_expert)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="path to the existing local LeRobot dataset root",
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
        default=256,
        help="mini-batch size for MLP training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate for Adam optimizer",
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
        config_path=args.config,
    )
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = default_checkpoint_dir_for_system(args.system)

    cfg = DaggerConfig(
        system=args.system,
        experiment_config=validated_config,
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        planner_name=args.planner,
        dagger_iterations=args.dagger_iterations,
        trajectories_per_iteration=args.trajectories_per_iteration,
        steps_per_trajectory=args.steps_per_trajectory,
        action_noise_std=args.action_noise_std,
        target_epochs_per_round=args.target_epochs_per_round,
        eval_episodes=args.eval_episodes,
        eval_steps=args.eval_steps,
        eval_seed_start=args.eval_seed_start,
        eval_action_noise_std=args.eval_action_noise_std,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
        max_train_steps=args.max_train_steps,
    )

    run_dagger(cfg)


if __name__ == "__main__":
    main()
