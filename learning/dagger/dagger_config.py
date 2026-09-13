from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from learning.config_loaders import EncoderConfig, FlowConfig


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
    expert_mix_decay_after_eval_success: float | None
    adaptive_beta_recovery: bool
    expert_mix_beta_recovery: float
    expert_mix_beta_recovery_increment: float
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
    goal_states: list[np.ndarray] | None = None
    training_curriculum: list[str] | None = None
    round_seeds: list[int] | None = None
    restart_round_seed: bool = False
    workspace_bounds: list[float] | None = None
    tolerance_overrides: dict[str, float] | None = None
    # Independent of tolerance_overrides: that one shapes what gets
    # demonstrated/trained on (tight tolerances during collection encourage
    # precise demonstrations), while this one only affects whether an eval
    # rollout counts as a success -- eval cares about "close enough and
    # collision-free", not exact final-pose matching, so it's reasonable for
    # this to be more relaxed without touching training dynamics or
    # demonstration length. Falls back to tolerance_overrides when unset.
    eval_tolerance_overrides: dict[str, float] | None = None

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
            self.expert_mix_decay_after_eval_success is not None
            and not 0 <= self.expert_mix_decay_after_eval_success <= 1
        ):
            raise ValueError("Beta gate must be in [0, 1].")
        if not 0 <= self.expert_mix_beta_recovery <= 1:
            raise ValueError("'expert_mix_beta_recovery' must be in [0, 1].")
        if self.expert_mix_beta_recovery_increment <= 0:
            raise ValueError(
                "'expert_mix_beta_recovery_increment' must be positive, so backtrack "
                "recovery always reaches beta=1.0 (the safe, guaranteed-recoverable "
                "fallback) in finitely many retries."
            )
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
        if self.training_curriculum is not None:
            if len(self.training_curriculum) != self.dagger_iterations:
                raise ValueError(
                    "'training_curriculum' must contain exactly one entry per DAgger round "
                    f"({self.dagger_iterations}), got {len(self.training_curriculum)}."
                )
            if any(mode not in {"random", "config"} for mode in self.training_curriculum):
                raise ValueError("'training_curriculum' entries must be 'random' or 'config'.")
        if self.round_seeds is not None and len(self.round_seeds) != self.dagger_iterations:
            raise ValueError(
                "'round_seeds' must contain exactly one entry per DAgger round "
                f"({self.dagger_iterations}), got {len(self.round_seeds)}."
            )
        if self.workspace_bounds is not None:
            if len(self.workspace_bounds) != 2:
                raise ValueError("'workspace_bounds' must contain exactly two values.")
            if self.workspace_bounds[1] <= self.workspace_bounds[0]:
                raise ValueError("'workspace_bounds[1]' must exceed 'workspace_bounds[0]'.")
        if self.tolerance_overrides is not None and any(
            value <= 0 for value in self.tolerance_overrides.values()
        ):
            raise ValueError("'tolerance_overrides' values must be positive.")
        if self.eval_tolerance_overrides is not None and any(
            value <= 0 for value in self.eval_tolerance_overrides.values()
        ):
            raise ValueError("'eval_tolerance_overrides' values must be positive.")
        if not self.dataset_root.exists() and not self.start_with_aggregation:
            raise FileNotFoundError(self.dataset_root)
