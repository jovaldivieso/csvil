from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from learning.config_loaders import (
    EncoderConfig, FlowConfig, default_checkpoint_dir_for_system,
    default_dataset_root_for_system, default_repo_id_for_system,
    load_dagger_training_config,
    load_encoder_config, load_flow_config, load_mlp_hidden_dims,
    load_observation_horizon, load_policy_type, load_prediction_horizon,
)
from learning.data_utils import collate_batch_for_policy, create_collate_fn_with_dataset
from learning.dagger import (
    DaggerEvalMetrics, ExpertMixBetaController, build_decentralized_joint_action,
    collect_dagger_rollouts, evaluate_policy_rollouts, print_rollout_metrics,
    ObservationHistoryBuffer, resolve_initial_state_seed, resolve_round_steps, set_seed,
    with_seeded_initial_state_config,
)
from learning.models.encoder import EncoderFactory
from learning.models.policy import ActionPolicy, PolicyFactory
from systems.dynamics import DynamicsProtocol
from systems.initial_state_utils import parse_initial_states_argument
from systems.seed_utils import default_action_noise_seed_for_config


def _validate_resumable_dataset_schema(
    existing_features: Mapping[str, Any],
    expected_features: Mapping[str, Any],
) -> None:
    """Fail fast when a dataset being resumed doesn't match the current observation schema."""
    for name, expected_info in expected_features.items():
        expected_shape = tuple(int(value) for value in expected_info.get("shape", ()))
        existing_info = existing_features.get(name)
        existing_shape = (
            tuple(int(value) for value in existing_info["shape"]) if existing_info else None
        )
        if existing_shape != expected_shape:
            raise ValueError(
                "Cannot resume DAgger collection: the on-disk dataset's "
                f"'{name}' feature has shape {existing_shape}, but the current run's observation "
                f"schema expects shape {expected_shape}. The dataset was likely recorded with a "
                "different neighbor/observation feature layout. Start a fresh dataset "
                "(omit --repo-id/--dataset-root) or resume a dataset recorded with the current schema."
            )


@dataclass(frozen=True)
class DaggerConfig:
    system: str
    experiment_config: dict[str, Any]
    repo_id: str
    dataset_root: Path
    start_with_aggregation: bool
    planner_name: str
    dagger_iterations: int
    trajectories_per_iteration: list[int]
    steps_per_trajectory: int
    action_noise_std: float
    expert_mix_beta_start: float
    expert_mix_beta_end: float
    expert_mix_beta_decay_rate: float | None
    expert_mix_decay_after_success_rate: float | None
    adaptive_beta_recovery: bool
    target_epochs_per_round: list[float]
    eval_episodes: int
    eval_steps: int | None
    eval_seed_start: int
    eval_action_noise_std: float
    batch_size: int
    learning_rate: float
    mlp_hidden_dims: tuple[int, ...]
    prediction_horizon: int
    observation_horizon: int
    encoder_config: EncoderConfig
    policy_type: str
    flow_config: FlowConfig
    checkpoint_dir: Path
    seed: int
    max_train_steps: int | None
    initial_states: list[np.ndarray] | None = None
    randomize_goal_with_eval_success: float | None = None

    def __post_init__(self) -> None:
        if self.dagger_iterations < 0:
            raise ValueError("'dagger_iterations' must be non-negative.")
        if len(self.trajectories_per_iteration) not in {1, self.dagger_iterations}:
            raise ValueError(
                "'trajectories_per_iteration' must contain one or one value per round."
            )
        if any(v <= 0 for v in self.trajectories_per_iteration):
            raise ValueError("Trajectory targets must be positive.")
        if self.steps_per_trajectory <= 0:
            raise ValueError("'steps_per_trajectory' must be positive.")
        if self.observation_horizon <= 0:
            raise ValueError("'observation_horizon' must be positive.")
        if self.action_noise_std < 0 or self.eval_action_noise_std < 0:
            raise ValueError("Action noise must be non-negative.")
        if not 0 <= self.expert_mix_beta_start <= 1 or not 0 <= self.expert_mix_beta_end <= 1:
            raise ValueError("Expert beta values must be in [0, 1].")
        if (
            self.expert_mix_beta_decay_rate is not None
            and self.expert_mix_beta_decay_rate < 0
        ):
            raise ValueError("Beta decay rate must be non-negative.")
        if (
            self.expert_mix_decay_after_success_rate is not None
            and not 0 <= self.expert_mix_decay_after_success_rate <= 1
        ):
            raise ValueError("Beta gate must be in [0, 1].")
        if len(self.target_epochs_per_round) not in {1, self.dagger_iterations}:
            raise ValueError(
                "'target_epochs_per_round' must contain one or one value per round."
            )
        if any(v <= 0 for v in self.target_epochs_per_round):
            raise ValueError("Epoch targets must be positive.")
        if self.eval_episodes < 0 or (
            self.eval_steps is not None and self.eval_steps <= 0
        ):
            raise ValueError("Evaluation counts must be valid.")
        if self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("Batch size and learning rate must be positive.")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("Max train steps must be positive.")
        if self.randomize_goal_with_eval_success is not None and not 0.0 <= self.randomize_goal_with_eval_success <= 100.0:
            raise ValueError("'randomize_goal_with_eval_success' must be between 0 and 100 percent.")
        if not self.dataset_root.exists() and not self.start_with_aggregation:
            raise FileNotFoundError(self.dataset_root)


class DaggerTrainer:
    def __init__(self, cfg: DaggerConfig) -> None:
        self.cfg = cfg
        self.device: torch.device | None = None
        self.simulator: DynamicsProtocol | None = None
        self.seeded_config: Mapping[str, Any] | None = None
        self.policy: ActionPolicy | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.action_noise_seed = 0
        self.initial_state_seed = 0
        self.dynamic_goal_randomization = False
        self.obs_feature_names: list[str] = []
        self.state_dim = self.action_dim = self.neighbor_slots = 0
        self.neighbor_feature_dim: int | None = None
        self.observation_horizon = 1

    @staticmethod
    def schedules(
        trajectories: list[int],
        epochs: list[float],
        rounds: int,
    ) -> tuple[list[int], list[float]]:
        def expand(values: list[Any], name: str) -> list[Any]:
            if len(values) == 1:
                return values if rounds == 0 else values * rounds
            if len(values) != rounds:
                raise ValueError(
                    f"{name} must contain one or exactly {rounds} values."
                )
            return values
        return (
            expand(trajectories, "trajectories-per-iteration"),
            expand(epochs, "target-epochs-per-round"),
        )

    def setup(self) -> None:
        set_seed(self.cfg.seed)
        self.seeded_config = with_seeded_initial_state_config(
            self.cfg.system,
            self.cfg.experiment_config,
            self.cfg.seed,
        )
        self.simulator = DynamicsFactory.create(
            system_name=self.cfg.system,
            config=self.seeded_config,
        )
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.action_noise_seed = default_action_noise_seed_for_config(self.seeded_config)
        self.initial_state_seed = resolve_initial_state_seed(self.seeded_config, self.cfg.seed)
        self.observation_horizon = self.cfg.observation_horizon
        features = self.simulator.get_dataset_features()
        self.obs_feature_names = [n for n in features if n.startswith("observation.")]
        base_ego_dim = sum(
            int(features[name]["shape"][0])
            for name in ("observation.environment_state", "observation.state")
        )
        self.action_dim = int(features["action"]["shape"][0])
        self.neighbor_slots = max(0, int(self.simulator.num_robots) - 1)
        neighbor_state_dim = int(features["observation.neighbor_state"]["shape"][0])
        if self.neighbor_slots > 0:
            if neighbor_state_dim <= 0 or neighbor_state_dim % self.neighbor_slots != 0:
                raise ValueError(
                    "observation.neighbor_state dimension must be a positive multiple of the neighbor count; "
                    f"got dimension {neighbor_state_dim} for {self.neighbor_slots} neighbors."
                )
            self.neighbor_feature_dim = (
                neighbor_state_dim // self.neighbor_slots
            ) * self.observation_horizon
            stacked_neighbor_mask_dim = self.neighbor_slots * self.observation_horizon
        else:
            # The encoder still requires a valid input width when there are no slots.
            self.neighbor_feature_dim = max(1, neighbor_state_dim) * self.observation_horizon
            stacked_neighbor_mask_dim = 0

        self.state_dim = (
            base_ego_dim * self.observation_horizon
            + self.neighbor_slots * self.neighbor_feature_dim
            + stacked_neighbor_mask_dim
        )

        if self.neighbor_feature_dim is None:
            raise RuntimeError("Neighbor feature dimension was not initialized from the dataset schema.")

        enc = EncoderFactory.create(
            self.cfg.encoder_config.encoder_type,
            self.state_dim,
            self.neighbor_feature_dim,
            self.neighbor_slots,
            observation_horizon=self.observation_horizon,
            **self.cfg.encoder_config.kwargs,
        )
        flow = (
            {"num_inference_steps": self.cfg.flow_config.num_inference_steps}
            if self.cfg.policy_type == "flow"
            else {}
        )
        self.policy = PolicyFactory.create(
            self.cfg.policy_type,
            action_dim=self.action_dim,
            obs_encoder=enc,
            hidden_dims=self.cfg.mlp_hidden_dims,
            prediction_horizon=self.cfg.prediction_horizon,
            **flow,
        ).to(self.device).eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate)
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Print rich startup diagnostic logs
        print("Starting DAgger training")
        print(f"Device: {self.device}")
        print(f"Initial dataset root: {self.cfg.dataset_root}")
        print(f"Aggregation action noise std: {self.cfg.action_noise_std:.6f}")
        print(f"Evaluation action noise std: {self.cfg.eval_action_noise_std:.6f}")
        print(f"Action noise seed: {self.action_noise_seed}")
        if self.cfg.expert_mix_beta_decay_rate is not None:
            print(
                "Expert execution mixing schedule: "
                f"beta_start={self.cfg.expert_mix_beta_start:.3f}, "
                f"beta_decay_rate={self.cfg.expert_mix_beta_decay_rate:.3f}/round, "
                f"beta_floor=0.000, "
                f"decay_after_eval_success={self.cfg.expert_mix_decay_after_success_rate if self.cfg.expert_mix_decay_after_success_rate is not None else 'none'}"
            )
        else:
            print(
                "Expert execution mixing schedule: "
                f"beta_start={self.cfg.expert_mix_beta_start:.3f}, "
                f"beta_end={self.cfg.expert_mix_beta_end:.3f}, "
                f"decay_rounds={self.cfg.dagger_iterations}, "
                f"decay_after_eval_success={self.cfg.expert_mix_decay_after_success_rate if self.cfg.expert_mix_decay_after_success_rate is not None else 'none'}"
            )
        print(f"MLP hidden dims: {list(self.cfg.mlp_hidden_dims)}")
        print(f"Prediction horizon: {self.cfg.prediction_horizon}")
        print(f"Policy type: {self.cfg.policy_type}")
        if self.cfg.policy_type == "flow":
            print(
                "Flow inference: "
                f"num_inference_steps={self.cfg.flow_config.num_inference_steps}"
            )
        if self.cfg.start_with_aggregation:
            print("Fresh DAgger mode: collecting round-0 data before any offline pretraining.")
        else:
            print("Initial offline training pass starts from the current expert dataset.")
        print(
            "Epoch-target schedule: "
            f"target_epochs={self.cfg.target_epochs_per_round}, "
            f"max={self.cfg.max_train_steps if self.cfg.max_train_steps is not None else 'none'}"
        )
        print(f"Decentralized policy neighbor slots: {self.neighbor_slots}")

    def train_policy_steps(self, dataloader: DataLoader, num_steps: int) -> float:
        if self.policy is None or self.optimizer is None or self.device is None:
            raise RuntimeError("Trainer is not set up.")
        self.policy.train()
        loss_fn = nn.MSELoss()
        running_loss = 0.0
        iterator = iter(dataloader)
        progress = tqdm(
            range(1, num_steps + 1),
            desc="Train steps",
            leave=False,
            dynamic_ncols=True,
        ) if tqdm is not None else None
        step_iterator = progress if progress is not None else range(1, num_steps + 1)
        if progress is None:
            print("tqdm not installed; showing periodic step progress.")

        for step in step_iterator:
            try:
                observations, actions = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                observations, actions = next(iterator)
            if isinstance(observations, dict):
                observations = {
                    key: value.to(self.device)
                    for key, value in observations.items()
                }
            else:
                observations = observations.to(self.device)
            actions = actions.to(self.device)
            self.optimizer.zero_grad()
            try:
                loss = self.policy.compute_loss(observations, actions)
            except NotImplementedError:
                loss = loss_fn(self.policy(observations), actions)
            loss.backward()
            self.optimizer.step()

            step_loss = float(loss.item())
            running_loss += step_loss
            running_mean = running_loss / float(step)
            if progress is not None:
                progress.set_postfix(loss=f"{step_loss:.6f}", mean=f"{running_mean:.6f}")
            elif step == 1 or step == num_steps or step % max(1, num_steps // 10) == 0:
                print(f"  step {step}/{num_steps} loss={step_loss:.6f} mean_loss={running_mean:.6f}")

        if progress is not None:
            progress.close()
        self.policy.eval()
        return running_loss / float(num_steps)

    def train_on_aggregate(self, label: str, training_round: int, target_epochs: float) -> None:
        assert self.simulator is not None
        assert self.policy is not None
        assert self.optimizer is not None
        assert self.device is not None
        print(f"\n=== {label} ===")
        dataset = LeRobotDataset(
            repo_id=self.cfg.repo_id,
            root=self.cfg.dataset_root,
        )
        generator = torch.Generator().manual_seed(self.cfg.seed + training_round)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=create_collate_fn_with_dataset(
                dataset=dataset,
                simulator=self.simulator,
                prediction_horizon=self.cfg.prediction_horizon,
                observation_horizon=self.cfg.observation_horizon,
            ),
        )
        steps, approx = resolve_round_steps(
            len(dataset),
            self.cfg.batch_size,
            target_epochs,
            self.cfg.max_train_steps,
        )
        print(f"Training on {len(dataset)} aggregated frames")
        print(
            f"  optimizer_steps={steps} "
            f"(~{approx:.2f} epochs at batch_size={self.cfg.batch_size})"
        )
        mean_loss = self.train_policy_steps(loader, steps)
        print(f"  mean_step_loss={mean_loss:.6f}")

    def evaluate_current_policy(self, label: str) -> DaggerEvalMetrics | None:
        assert self.policy is not None
        assert self.device is not None
        assert self.seeded_config is not None
        if self.cfg.eval_episodes == 0:
            return None
        eval_config = copy.deepcopy(dict(self.seeded_config))
        if self.dynamic_goal_randomization:
            if "robots" in eval_config:
                for robot_entry in eval_config["robots"]:
                    if isinstance(robot_entry.get("config"), dict):
                        robot_entry["config"]["randomize_goal"] = True
            else:
                eval_config["randomize_goal"] = True
        simulator = DynamicsFactory.create(
            system_name=self.cfg.system,
            config=eval_config,
        )
        history_buffer = ObservationHistoryBuffer(
            self.cfg.observation_horizon,
            int(simulator.num_robots),
        )

        def action_fn(obs: np.ndarray) -> np.ndarray:
            return build_decentralized_joint_action(
                simulator,
                self.policy,
                obs,
                self.device,
                observation_horizon=self.cfg.observation_horizon,
                history_buffer=history_buffer,
            )

        def reset_policy_state() -> None:
            history_buffer.reset()
            self.policy.reset()

        metrics = evaluate_policy_rollouts(
            simulator,
            self.cfg.eval_episodes,
            self.cfg.eval_steps or self.cfg.steps_per_trajectory,
            self.cfg.eval_seed_start,
            action_fn,
            reset_fn=reset_policy_state,
            action_noise_std=self.cfg.eval_action_noise_std,
            action_noise_seed=self.action_noise_seed,
        )
        if metrics is not None:
            print_rollout_metrics(label, "eval", metrics)
        return metrics

    def save_checkpoints(self, training_round: int) -> None:
        assert self.policy is not None
        assert self.optimizer is not None
        data: dict[str, Any] = {
            "iteration": training_round,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "prediction_horizon": self.cfg.prediction_horizon,
            "observation_horizon": self.cfg.observation_horizon,
            "hidden_dims": list(self.cfg.mlp_hidden_dims),
            "obs_feature_names": self.obs_feature_names,
            "system": self.cfg.system,
            "neighbor_feature_dim": self.neighbor_feature_dim,
            "neighbor_slots": self.neighbor_slots,
            "encoder_type": self.cfg.encoder_config.encoder_type,
            "encoder_kwargs": self.cfg.encoder_config.kwargs,
            "policy_type": self.cfg.policy_type,
        }
        if self.cfg.policy_type == "flow":
            data["flow_config"] = {
                "num_inference_steps": self.cfg.flow_config.num_inference_steps,
            }
        prefix = "flow_dagger" if self.cfg.policy_type == "flow" else "mlp_dagger"
        latest_checkpoint = self.cfg.checkpoint_dir / f"{prefix}_checkpoint.pt"
        iteration_checkpoint = self.cfg.checkpoint_dir / f"{prefix}_iter_{training_round:03d}.pt"
        torch.save(data, latest_checkpoint)
        torch.save(data, iteration_checkpoint)
        print(f"Saved checkpoints: {latest_checkpoint} and {iteration_checkpoint}")

    def run(self) -> None:
        self.setup()
        assert self.simulator is not None
        assert self.policy is not None
        assert self.device is not None
        assert self.seeded_config is not None

        beta = ExpertMixBetaController(
            beta_start=self.cfg.expert_mix_beta_start,
            beta_end=self.cfg.expert_mix_beta_end,
            decay_rounds=max(1, self.cfg.dagger_iterations),
            beta_decay_rate=self.cfg.expert_mix_beta_decay_rate,
            decay_after_success_rate=self.cfg.expert_mix_decay_after_success_rate,
            adaptive_recovery=self.cfg.adaptive_beta_recovery,
        )

        initial = None
        if not self.cfg.start_with_aggregation:
            self.train_on_aggregate(
                "Initial offline training pass",
                0,
                self.cfg.target_epochs_per_round[0],
            )
            initial = self.evaluate_current_policy("Round 0 evaluation")
            self.save_checkpoints(0)
            if self.cfg.dagger_iterations == 0:
                print("No DAgger refinements requested (--dagger-iterations 0).")
                return
            rounds = range(1, self.cfg.dagger_iterations + 1)
            if initial is not None:
                beta.prime_from_evaluation(initial.success_rate)
        else:
            if self.cfg.dagger_iterations == 0:
                raise ValueError(
                    "Fresh DAgger mode requires at least one aggregation round; "
                    "set --dagger-iterations to a positive value."
                )
            rounds = range(self.cfg.dagger_iterations)

        for index in rounds:
            if self.cfg.start_with_aggregation:
                schedule = index
                display = index + 1
                print(f"\n=== DAgger round {display}/{self.cfg.dagger_iterations}: aggregate ===")
            else:
                schedule = index - 1
                display = index
                print(f"\n=== DAgger refinement {display}/{self.cfg.dagger_iterations}: aggregate ===")

            collection_config = copy.deepcopy(dict(self.seeded_config))
            if self.dynamic_goal_randomization:
                if "robots" in collection_config:
                    for robot_entry in collection_config["robots"]:
                        if isinstance(robot_entry.get("config"), dict):
                            robot_entry["config"]["randomize_goal"] = True
                else:
                    collection_config["randomize_goal"] = True

            if "robots" in collection_config:
                aggregation_goals_randomized = any(
                    isinstance(robot_entry.get("config"), dict)
                    and bool(robot_entry["config"].get("randomize_goal", False))
                    for robot_entry in collection_config["robots"]
                )
            else:
                aggregation_goals_randomized = bool(
                    collection_config.get("randomize_goal", False)
                )
            aggregation_goal_source = "random" if aggregation_goals_randomized else "config"
            print(f"Aggregation goal source: {aggregation_goal_source}")

            simulator = DynamicsFactory.create(
                system_name=self.cfg.system,
                config=collection_config,
            )
            planner = PlannerFactory.create(
                self.cfg.planner_name,
                simulator,
                collection_config,
            )

            round_beta = beta.current_beta
            print(
                "Aggregation execution policy: "
                f"expert_beta={round_beta:.3f}, "
                f"decay_active={'yes' if beta.decay_active else 'no'}"
            )

            if self.cfg.start_with_aggregation and not self.cfg.dataset_root.exists():
                features = simulator.get_dataset_features()
                writer = LeRobotDataset.create(
                    repo_id=self.cfg.repo_id,
                    fps=int(1 / simulator.dt),
                    root=self.cfg.dataset_root,
                    features=features,
                )
            else:
                existing_meta = LeRobotDatasetMetadata(repo_id=self.cfg.repo_id, root=self.cfg.dataset_root)
                _validate_resumable_dataset_schema(existing_meta.features, simulator.get_dataset_features())
                writer = LeRobotDataset.resume(
                    repo_id=self.cfg.repo_id,
                    root=self.cfg.dataset_root,
                )

            try:
                history_buffer = ObservationHistoryBuffer(
                    self.cfg.observation_horizon,
                    int(simulator.num_robots),
                )

                def action_fn(obs: np.ndarray) -> np.ndarray:
                    return build_decentralized_joint_action(
                        simulator,
                        self.policy,
                        obs,
                        self.device,
                        observation_horizon=self.cfg.observation_horizon,
                        history_buffer=history_buffer,
                    )

                def reset_policy_state() -> None:
                    history_buffer.reset()
                    self.policy.reset()

                frames = simulator.format_dataset_frame
                metrics = collect_dagger_rollouts(
                    simulator=simulator,
                    expert_planner=planner,
                    dataset_writer=writer,
                    trajectories_per_iteration=self.cfg.trajectories_per_iteration[schedule],
                    steps_per_trajectory=self.cfg.steps_per_trajectory,
                    action_noise_std=self.cfg.action_noise_std,
                    action_noise_seed=self.action_noise_seed,
                    initial_state_seed=self.initial_state_seed,
                    initial_states=self.cfg.initial_states,
                    expert_mixing_beta=round_beta,
                    policy_action_fn=action_fn,
                    policy_reset_fn=reset_policy_state,
                    frame_builder=frames,
                )
            finally:
                writer.finalize()
                del writer
                gc.collect()

            print_rollout_metrics(
                label=f"Round {display} aggregation"
                if self.cfg.start_with_aggregation
                else f"Refinement {display} aggregation",
                prefix="aggregation",
                metrics=metrics,
            )
            print(f"aggregation_goal_source: {aggregation_goal_source}")

            success_threshold = self.cfg.randomize_goal_with_eval_success
            self.train_on_aggregate(
                f"DAgger round {display}/{self.cfg.dagger_iterations}: retrain"
                if self.cfg.start_with_aggregation
                else f"DAgger refinement {display}/{self.cfg.dagger_iterations}: retrain",
                training_round=index,
                target_epochs=self.cfg.target_epochs_per_round[schedule],
            )
            eval_metrics = self.evaluate_current_policy(
                f"Round {display} evaluation"
                if self.cfg.start_with_aggregation
                else f"Refinement {display} evaluation"
            )

            if eval_metrics is not None:
                eval_success_pct = eval_metrics.success_rate * 100.0
                if (
                    success_threshold is not None
                    and round_beta == 0.0
                    and eval_success_pct >= success_threshold
                ):
                    if not self.dynamic_goal_randomization:
                        print(
                            "Curriculum milestone reached during evaluation: "
                            f"success rate {eval_success_pct:.1f}% >= threshold {success_threshold}% "
                            "with beta=0.0. Enabling goal randomization."
                        )
                    self.dynamic_goal_randomization = True

            beta.update_after_evaluation(
                eval_metrics.success_rate if eval_metrics is not None else None
            )
            self.save_checkpoints(training_round=index)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a policy with DAgger")
    p.add_argument("--system", type=str.lower, choices=DynamicsFactory.names(), required=True)
    p.add_argument("--expert-config", required=True)
    p.add_argument("--repo-id")
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--planner", default="casadi", choices=["casadi"])
    p.add_argument("--dagger-iterations", type=int, default=4)
    p.add_argument("--trajectories-per-iteration", nargs="+", type=int, default=[20])
    p.add_argument("--steps-per-trajectory", type=int, default=150)
    p.add_argument("--action-noise-std", type=float, default=0.0)
    p.add_argument(
        "--randomize-goal-with-eval-success",
        type=float,
        default=None,
        help="enable goal randomization when a beta=0.0 round reaches this evaluation success percentage",
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
    p.add_argument("--expert-mix-beta-start", type=float, default=0.8)
    p.add_argument("--expert-mix-beta-end", type=float, default=0.0)
    p.add_argument("--expert-mix-beta-decay-rate", type=float)
    p.add_argument("--expert-mix-decay-after-success-rate", type=float)
    p.add_argument("--adaptive-beta-recovery", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--target-epochs-per-round", nargs="+", type=float, default=[30.0])
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-steps", type=int)
    p.add_argument("--eval-seed-start", type=int, default=10000)
    p.add_argument("--eval-action-noise-std", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--policy-config", type=Path)
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--max-train-steps", type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
    trajectories, epochs = DaggerTrainer.schedules(
        args.trajectories_per_iteration,
        args.target_epochs_per_round,
        args.dagger_iterations,
    )
    cfg = DaggerConfig(
        system=args.system,
        experiment_config=validated,
        repo_id=repo_id,
        dataset_root=dataset_root,
        start_with_aggregation=fresh,
        planner_name=args.planner,
        dagger_iterations=args.dagger_iterations,
        trajectories_per_iteration=trajectories,
        steps_per_trajectory=args.steps_per_trajectory,
        action_noise_std=args.action_noise_std,
        initial_states=parse_initial_states_argument(args.initial_states),
        expert_mix_beta_start=args.expert_mix_beta_start,
        expert_mix_beta_end=args.expert_mix_beta_end,
        expert_mix_beta_decay_rate=args.expert_mix_beta_decay_rate,
        expert_mix_decay_after_success_rate=args.expert_mix_decay_after_success_rate,
        adaptive_beta_recovery=args.adaptive_beta_recovery,
        target_epochs_per_round=epochs,
        eval_episodes=args.eval_episodes,
        eval_steps=args.eval_steps,
        eval_seed_start=args.eval_seed_start,
        eval_action_noise_std=args.eval_action_noise_std,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mlp_hidden_dims=load_mlp_hidden_dims(args.policy_config),
        prediction_horizon=load_prediction_horizon(args.policy_config),
        encoder_config=load_encoder_config(args.policy_config),
        policy_type=load_policy_type(args.policy_config),
        flow_config=load_flow_config(args.policy_config),
        checkpoint_dir=args.checkpoint_dir or default_checkpoint_dir_for_system(args.system),
        seed=args.seed,
        max_train_steps=args.max_train_steps,
        randomize_goal_with_eval_success=args.randomize_goal_with_eval_success,
    )
    DaggerTrainer(cfg).run()


if __name__ == "__main__":
    main()