from __future__ import annotations

import copy
import csv
import gc
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from core.config import validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from learning.data_utils import (
    build_action_window_cache,
    build_observation_history_cache,
    create_collate_fn_with_dataset,
)
from learning.models.encoder import EncoderFactory
from learning.models.policy import ActionPolicy, PolicyFactory
from systems.dynamics import DynamicsProtocol
from systems.seed_utils import default_action_noise_seed_for_config

from .beta_controller import ExpertMixBetaController
from .dagger_config import DaggerConfig
from .metrics import DaggerEvalMetrics
from .rollouts import ObservationHistoryBuffer, build_decentralized_joint_action, collect_dagger_rollouts, evaluate_policy_rollouts
from .utils import (
    apply_config_overrides,
    print_rollout_metrics,
    resolve_initial_state_seed,
    resolve_round_steps,
    set_seed,
    with_seeded_initial_state_config,
)


def _validate_resumable_dataset_schema(
    existing_features: Mapping[str, Any],
    expected_features: Mapping[str, Any],
) -> None:
    """Fail fast when a dataset being resumed doesn't match the current observation schema."""
    for name, expected_info in expected_features.items():
        expected_shape = tuple(int(value) for value in expected_info.get("shape", ()))
        expected_dtype = expected_info.get("dtype")
        expected_names = list(expected_info.get("names") or [])
        existing_info = existing_features.get(name)
        if existing_info is None:
            existing_shape = None
            existing_dtype = None
            existing_names = None
        else:
            existing_shape = tuple(int(value) for value in existing_info.get("shape", ()))
            existing_dtype = existing_info.get("dtype")
            existing_names = list(existing_info.get("names") or [])
        if (existing_shape, existing_dtype, existing_names) != (expected_shape, expected_dtype, expected_names):
            raise ValueError(
                "Cannot resume DAgger collection: the on-disk dataset's "
                f"'{name}' feature is (shape={existing_shape}, dtype={existing_dtype}, names={existing_names}), "
                f"but the current run's observation schema expects "
                f"(shape={expected_shape}, dtype={expected_dtype}, names={expected_names}). "
                "The dataset was likely recorded with a different neighbor/observation feature layout. "
                "Start a fresh dataset (omit --repo-id/--dataset-root) or resume a dataset recorded "
                "with the current schema."
            )


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
        self.obs_feature_names: list[str] = []
        self.state_dim = self.action_dim = self.neighbor_slots = 0
        self.neighbor_feature_dim: int | None = None
        self.observation_horizon = 1

    @staticmethod
    def schedules(
        trajectories: list[int],
        epochs: list[float],
        rounds: int,
        training_curriculum: list[str] | None = None,
        round_seeds: list[int] | None = None,
    ) -> tuple[list[int], list[float], list[str] | None, list[int] | None]:
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
            expand(training_curriculum, "training-curriculum") if training_curriculum is not None else None,
            expand(round_seeds, "round-seeds") if round_seeds is not None else None,
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
        if self.cfg.dataset_root.exists():
            existing_meta = LeRobotDatasetMetadata(repo_id=self.cfg.repo_id, root=self.cfg.dataset_root)
            _validate_resumable_dataset_schema(existing_meta.features, features)
        self.obs_feature_names = [n for n in features if n.startswith("observation.")]
        # observation.state (proprioception) and its companion
        # observation.state_mask are stacked across observation_horizon like
        # the neighbor tensors; observation.environment_state (goal-relative
        # encoding) stays single-frame.
        environment_state_dim = int(features["observation.environment_state"]["shape"][0])
        proprioception_dim = int(features["observation.state"]["shape"][0])
        state_mask_dim = int(features["observation.state_mask"]["shape"][0])
        base_ego_dim = environment_state_dim + (proprioception_dim + state_mask_dim) * self.observation_horizon
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
            base_ego_dim
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
            {
                "num_inference_steps": self.cfg.flow_config.num_inference_steps,
                # Per-robot physical action bound, e.g. unicycle2's
                # [max_linear_accel, max_angular_accel] -- see FlowPolicy's
                # own docstring for why its diffusion-style training needs
                # this to keep every action dimension on comparable footing
                # against its isotropic unit-scale noise prior. Homogeneous
                # fleet (validated elsewhere), so simulators[0] speaks for
                # every robot's own action bound.
                "action_scale": np.broadcast_to(
                    np.asarray(self.simulator.simulators[0].max_action, dtype=float),
                    (self.action_dim,),
                ).tolist(),
            }
            if self.cfg.policy_type in {"flow", "safeflow"}
            else {}
        )
        safeflow = (
            {"simulator": self.simulator, "planner_config": self.seeded_config}
            if self.cfg.policy_type == "safeflow"
            else {}
        )
        self.policy = PolicyFactory.create(
            self.cfg.policy_type,
            action_dim=self.action_dim,
            obs_encoder=enc,
            hidden_dims=self.cfg.mlp_hidden_dims,
            prediction_horizon=self.cfg.prediction_horizon,
            **flow,
            **safeflow,
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
                f"decay_after_eval_success={self.cfg.expert_mix_decay_after_eval_success if self.cfg.expert_mix_decay_after_eval_success is not None else 'none'}"
            )
        else:
            print(
                "Expert execution mixing schedule: "
                f"beta_start={self.cfg.expert_mix_beta_start:.3f}, "
                f"beta_end={self.cfg.expert_mix_beta_end:.3f}, "
                f"decay_rounds={self.cfg.dagger_iterations}, "
                f"decay_after_eval_success={self.cfg.expert_mix_decay_after_eval_success if self.cfg.expert_mix_decay_after_eval_success is not None else 'none'}"
            )
        print(
            "Backtrack recovery schedule: "
            f"beta_recovery={self.cfg.expert_mix_beta_recovery:.3f}, "
            f"beta_recovery_increment={self.cfg.expert_mix_beta_recovery_increment:.3f} "
            "(escalates toward 1.0 per retry at the same backtracked state; "
            "resets to beta_recovery fresh on every new backtrack)"
        )
        print(f"MLP hidden dims: {list(self.cfg.mlp_hidden_dims)}")
        print(f"Prediction horizon: {self.cfg.prediction_horizon}")
        print(f"Policy type: {self.cfg.policy_type}")
        if self.cfg.policy_type in {"flow", "safeflow"}:
            print(
                "Flow inference: "
                f"num_inference_steps={self.cfg.flow_config.num_inference_steps}"
            )
            action_scale = np.broadcast_to(
                np.asarray(self.simulator.simulators[0].max_action, dtype=float),
                (self.action_dim,),
            ).tolist()
            print(
                "Flow action normalization: "
                f"action_scale={action_scale} "
                "(per-dimension divisor against the flow-matching noise prior, from this robot's own max_action)"
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
            loss = self.policy.compute_loss(observations, actions)
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

    def train_on_aggregate(self, label: str, training_round: int, target_epochs: float) -> float:
        assert self.simulator is not None
        assert self.policy is not None
        assert self.optimizer is not None
        assert self.device is not None
        print(f"\n=== {label} ===")
        dataset = LeRobotDataset(
            repo_id=self.cfg.repo_id,
            root=self.cfg.dataset_root,
        )
        action_window_cache = build_action_window_cache(dataset, self.cfg.prediction_horizon)
        observation_history_cache = build_observation_history_cache(dataset, self.cfg.observation_horizon)
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
                action_window_cache=action_window_cache,
                observation_history_cache=observation_history_cache,
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
        return mean_loss

    def _apply_runtime_config_overrides(
        self, config: dict[str, Any], *, tolerance_overrides: Mapping[str, float] | None = None
    ) -> dict[str, Any]:
        """Inject the training config's solver/dynamics tuning knobs (sampling bounds, tolerances), if set.

        tolerance_overrides defaults to self.cfg.tolerance_overrides (the
        collection/training value) when not given -- evaluate_current_policy
        passes self.cfg.eval_tolerance_overrides here instead, so eval can
        use a more relaxed pass/fail standard without touching what
        collection/training actually demonstrates against.
        """
        overrides: dict[str, Any] = {}
        if self.cfg.workspace_bounds is not None:
            overrides["workspace_bounds"] = list(self.cfg.workspace_bounds)
        effective_tolerance_overrides = (
            tolerance_overrides if tolerance_overrides is not None else self.cfg.tolerance_overrides
        )
        if effective_tolerance_overrides:
            overrides.update(effective_tolerance_overrides)
        merged_config = apply_config_overrides(config, overrides)
        return validate_system_config(self.cfg.system, merged_config)

    def evaluate_current_policy(self, label: str) -> DaggerEvalMetrics | None:
        assert self.policy is not None
        assert self.device is not None
        assert self.seeded_config is not None
        if self.cfg.eval_episodes == 0:
            return None
        eval_config = self._apply_runtime_config_overrides(
            copy.deepcopy(dict(self.seeded_config)),
            tolerance_overrides=self.cfg.eval_tolerance_overrides,
        )
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
            initial_states=self.cfg.initial_states,
            goal_states=self.cfg.goal_states,
        )
        if metrics is not None:
            print_rollout_metrics(label, "eval", metrics)
        return metrics

    def save_results(
        self,
        train_loss: float,
        eval_metrics: DaggerEvalMetrics | None,
        aggregation_metrics: DaggerEvalMetrics | None = None,
    ) -> None:
        """
        saves training and evaluation results to results.csv
        """

        path_to_results = self.cfg.checkpoint_dir / "results.csv"

        results = {
            "train_loss": train_loss,
            "aggregation_success_rate": (
                aggregation_metrics.success_rate
                if aggregation_metrics is not None
                else None
            ),
            "eval_success_rate": (
                eval_metrics.success_rate
                if eval_metrics is not None
                else None
            ),
            "eval_mean_steps": (
                eval_metrics.mean_steps
                if eval_metrics is not None
                else None
            ),
        }

        with path_to_results.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)

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
        if self.cfg.policy_type in {"flow", "safeflow"}:
            data["flow_config"] = {
                "num_inference_steps": self.cfg.flow_config.num_inference_steps,
                # Saved explicitly (not re-derived from a live simulator at
                # eval time) so a checkpoint's normalized action space stays
                # tied to whatever max_action trained it, even if the
                # evaluating caller's --config later changes those physical
                # bounds -- see FlowPolicy's own docstring for why training
                # and inference must agree on this exactly.
                "action_scale": np.broadcast_to(
                    np.asarray(self.simulator.simulators[0].max_action, dtype=float),
                    (self.action_dim,),
                ).tolist(),
            }
        prefix = "flow_dagger" if self.cfg.policy_type in {"flow", "safeflow"} else "mlp_dagger"
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
            decay_after_success_rate=self.cfg.expert_mix_decay_after_eval_success,
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

            mode = (
                self.cfg.training_curriculum[schedule]
                if self.cfg.training_curriculum is not None
                else ("config" if self.cfg.initial_states or self.cfg.goal_states else "random")
            )
            round_initial_states = self.cfg.initial_states if mode == "config" else None
            round_goal_states = self.cfg.goal_states if mode == "config" else None
            round_initial_state_seed = (
                self.cfg.round_seeds[schedule]
                if self.cfg.round_seeds is not None
                else self.initial_state_seed
            )
            collection_config = self._apply_runtime_config_overrides(
                copy.deepcopy(dict(self.seeded_config))
            )
            collection_config = apply_config_overrides(
                collection_config, {"randomize_goal": True}
            )
            print(f"Aggregation goal source: {mode}")

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
                    initial_state_seed=round_initial_state_seed,
                    initial_states=round_initial_states,
                    goal_states=round_goal_states,
                    expert_mixing_beta=round_beta,
                    round_index=schedule,
                    restart_initial_state_round=self.cfg.restart_round_seed,
                    beta_recovery=self.cfg.expert_mix_beta_recovery,
                    beta_recovery_increment=self.cfg.expert_mix_beta_recovery_increment,
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
            print(f"aggregation_goal_source: {mode}")

            train_loss = self.train_on_aggregate(
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

            beta.update_after_evaluation(
                eval_metrics.success_rate if eval_metrics is not None else None
            )
            self.save_checkpoints(training_round=index)

        # saves final results:
        self.save_results(
            train_loss=train_loss,
            eval_metrics=eval_metrics,
            aggregation_metrics=metrics,
        )
