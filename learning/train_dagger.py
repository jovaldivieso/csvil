from __future__ import annotations

import argparse
import gc
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

from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from learning.models.mlp import CustomMLPPolicy
from planning.casadi_planner import PlannerSolveError


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


def observation_feature_names(simulator) -> list[str]:
    return [
        feature_name
        for feature_name in simulator.get_dataset_features().keys()
        if feature_name.startswith("observation.")
    ]


def observation_dim_from_features(simulator) -> int:
    total_dim = 0
    for feature_name, feature_info in simulator.get_dataset_features().items():
        if feature_name.startswith("observation."):
            total_dim += int(feature_info["shape"][0])
    return total_dim


def flatten_observation_for_policy(simulator, observation: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Reproduces evaluate_lerobot dynamic slicing logic, then flattens for MLP.
    """
    features = simulator.get_dataset_features()

    current_idx = 0
    chunks: list[torch.Tensor] = []
    for feature_name, feature_info in features.items():
        if feature_name.startswith("observation."):
            dim = int(feature_info["shape"][0])
            sliced_obs = observation[current_idx: current_idx + dim]
            chunks.append(torch.as_tensor(sliced_obs, dtype=torch.float32, device=device))
            current_idx += dim

    if current_idx != observation.shape[0]:
        raise ValueError(
            "Observation slicing mismatch. "
            f"Consumed {current_idx} values but observation has length {observation.shape[0]}."
        )

    return torch.cat(chunks, dim=0).unsqueeze(0)


class LeRobotMLPDataset(Dataset):
    def __init__(self, lerobot_dataset: LeRobotDataset, obs_feature_names: list[str]):
        self.dataset = lerobot_dataset
        self.obs_feature_names = obs_feature_names

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        obs_parts = [sample[name].float().view(-1) for name in self.obs_feature_names]
        observation = torch.cat(obs_parts, dim=0)
        action = sample["action"].float().view(-1)
        return observation, action


@dataclass(frozen=True)
class DaggerConfig:
    system: str
    experiment_config: Mapping[str, Any]
    repo_id: str
    dataset_root: Path
    planner_name: str
    dagger_iterations: int
    epochs_per_iteration: int
    trajectories_per_iteration: int
    steps_per_trajectory: int
    batch_size: int
    learning_rate: float
    checkpoint_dir: Path
    seed: int


def train_policy_epoch(
    policy: CustomMLPPolicy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    policy.train()
    mse_loss = nn.MSELoss()
    running_loss = 0.0

    for observations, actions in dataloader:
        observations = observations.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()
        predictions = policy(observations)
        loss = mse_loss(predictions, actions)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())

    return running_loss / max(len(dataloader), 1)


def collect_dagger_data(
    simulator,
    expert_planner,
    policy: CustomMLPPolicy,
    dataset_writer: LeRobotDataset,
    trajectories_per_iteration: int,
    steps_per_trajectory: int,
    device: torch.device,
) -> int:
    """
    Roll out learner policy, query expert at visited states, aggregate labels.
    """
    successful_episodes = 0
    attempted_episodes = 0
    max_attempts = max(trajectories_per_iteration * 3, trajectories_per_iteration)

    policy.eval()

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
        expert_planner.reset()

        for _ in range(steps_per_trajectory):
            observation = simulator.observe(state)

            with torch.inference_mode():
                model_input = flatten_observation_for_policy(simulator, observation, device=device)
                policy_action = policy.select_action(model_input).squeeze(0).cpu().numpy()

            try:
                expert_action = expert_planner(observation)
            except PlannerSolveError as exc:
                print(f"Skipping episode due to planner failure: {exc}")
                planner_failed = True
                break

            frame_data = simulator.format_dataset_frame(observation, expert_action)
            frame_data["task"] = "reach target"
            dataset_writer.add_frame(frame_data)

            # Environment advances using learner action; label is expert correction.
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


def run_dagger(cfg: DaggerConfig) -> None:
    set_seed(cfg.seed)

    if cfg.dagger_iterations <= 0:
        raise ValueError("'dagger_iterations' must be positive.")
    if cfg.epochs_per_iteration <= 0:
        raise ValueError("'epochs_per_iteration' must be positive.")
    if cfg.trajectories_per_iteration <= 0:
        raise ValueError("'trajectories_per_iteration' must be positive.")
    if cfg.steps_per_trajectory <= 0:
        raise ValueError("'steps_per_trajectory' must be positive.")
    if cfg.batch_size <= 0:
        raise ValueError("'batch_size' must be positive.")
    if cfg.learning_rate <= 0:
        raise ValueError("'learning_rate' must be positive.")

    if not cfg.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {cfg.dataset_root}")

    simulator = DynamicsFactory.create(system_name=cfg.system, config=cfg.experiment_config)
    device = get_training_device()

    obs_feature_names = observation_feature_names(simulator)
    state_dim = observation_dim_from_features(simulator)
    action_dim = int(simulator.nu)

    policy = CustomMLPPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Starting DAgger training")
    print(f"Device: {device}")
    print(f"Initial dataset root: {cfg.dataset_root}")
    print(
        "DAgger iteration 1 starts from offline expert data "
        "(parameter-free setup with beta_1 = 1)."
    )

    for iteration in range(1, cfg.dagger_iterations + 1):
        print(f"\n=== DAgger iteration {iteration}/{cfg.dagger_iterations} ===")

        aggregate_dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.dataset_root)
        train_dataset = LeRobotMLPDataset(
            lerobot_dataset=aggregate_dataset,
            obs_feature_names=obs_feature_names,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )

        print(f"Training on {len(train_dataset)} aggregated frames")
        for epoch in range(1, cfg.epochs_per_iteration + 1):
            epoch_loss = train_policy_epoch(
                policy=policy,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
            )
            print(f"  epoch {epoch:03d}/{cfg.epochs_per_iteration:03d} | mse={epoch_loss:.6f}")

        simulator_for_rollout = DynamicsFactory.create(system_name=cfg.system, config=cfg.experiment_config)
        expert_planner = PlannerFactory.create(
            planner_name=cfg.planner_name,
            simulator=simulator_for_rollout,
            config=cfg.experiment_config,
        )
        dataset_writer = LeRobotDataset.resume(repo_id=cfg.repo_id, root=cfg.dataset_root)

        try:
            collected_episodes = collect_dagger_data(
                simulator=simulator_for_rollout,
                expert_planner=expert_planner,
                policy=policy,
                dataset_writer=dataset_writer,
                trajectories_per_iteration=cfg.trajectories_per_iteration,
                steps_per_trajectory=cfg.steps_per_trajectory,
                device=device,
            )
        finally:
            # LeRobot writes parquet chunks lazily; finalize guarantees readable footers.
            dataset_writer.finalize()
            del dataset_writer
            gc.collect()

        print(f"Collected {collected_episodes} new DAgger episodes")

        checkpoint_data = {
            "iteration": iteration,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "obs_feature_names": obs_feature_names,
            "system": cfg.system,
        }

        latest_checkpoint = cfg.checkpoint_dir / "mlp_dagger_checkpoint.pt"
        iteration_checkpoint = cfg.checkpoint_dir / f"mlp_dagger_iter_{iteration:03d}.pt"
        torch.save(checkpoint_data, latest_checkpoint)
        torch.save(checkpoint_data, iteration_checkpoint)

        print(f"Saved checkpoints: {latest_checkpoint} and {iteration_checkpoint}")


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
        default=5,
        help="number of outer DAgger iterations",
    )
    parser.add_argument(
        "--epochs-per-iteration",
        type=int,
        default=20,
        help="supervised training epochs per DAgger iteration",
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
        default=Path("outputs/train_dagger"),
        help="directory where DAgger checkpoints are written",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validated_config = load_and_validate_system_config(
        system_name=args.system,
        config_path=args.config,
    )

    cfg = DaggerConfig(
        system=args.system,
        experiment_config=validated_config,
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        planner_name=args.planner,
        dagger_iterations=args.dagger_iterations,
        epochs_per_iteration=args.epochs_per_iteration,
        trajectories_per_iteration=args.trajectories_per_iteration,
        steps_per_trajectory=args.steps_per_trajectory,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    run_dagger(cfg)


if __name__ == "__main__":
    main()
