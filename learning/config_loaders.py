from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping

from core.config import load_and_validate_mlp_architecture_config, load_yaml_config
from learning.models.encoder import DEFAULT_ENCODER_TYPE

DEFAULT_MLP_HIDDEN_DIMS: tuple[int, ...] = (256, 256, 128)
DEFAULT_FLOW_HIDDEN_DIMS: tuple[int, ...] = (256, 256, 256)
DEFAULT_DAGGER_TRAINING_CONFIG: dict[str, object] = {
    "planner": "casadi",
    "dagger_iterations": 4,
    "trajectories_per_iteration": [20],
    "steps_per_trajectory": 150,
    "action_noise_std": 0.0,
    "expert_mix_beta_start": 0.8,
    "expert_mix_beta_end": 0.0,
    "expert_mix_beta_decay_rate": None,
    "expert_mix_decay_after_eval_success": None,
    "adaptive_beta_recovery": False,
    "target_epochs_per_round": [30.0],
    "eval_episodes": 10,
    "eval_steps": None,
    "eval_seed_start": 10000,
    "eval_action_noise_std": 0.0,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "seed": 99,
    "max_train_steps": None,
    "randomize_goal_after_eval_success": None,
    "initial_states": None,
}


@dataclass(frozen=True)
class EncoderConfig:
    encoder_type: str
    kwargs: dict[str, object]


@dataclass(frozen=True)
class FlowConfig:
    num_inference_steps: int = 10
    observation_horizon: int = 1


@lru_cache(maxsize=None)
def _cached_yaml_config(policy_config_path: Path) -> dict[str, object]:
    """Avoid re-parsing the same policy YAML once per load_* call in a training run."""
    return load_yaml_config(policy_config_path)


def load_dagger_training_config(policy_config_path: Path | None) -> dict[str, object]:
    """Load optional DAgger CLI defaults from a policy configuration file."""
    config = dict(DEFAULT_DAGGER_TRAINING_CONFIG)
    if policy_config_path is None:
        return config

    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if not isinstance(model_section, Mapping):
        raise ValueError("Policy config model section must be a mapping.")
    training_section = raw_config.get("training", model_section.get("training", {}))
    if not isinstance(training_section, Mapping):
        raise ValueError("Policy config 'training' section must be a mapping.")
    config.update(training_section)
    return config


def load_mlp_hidden_dims(policy_config_path: Path | None) -> tuple[int, ...]:
    if policy_config_path is None:
        return DEFAULT_MLP_HIDDEN_DIMS

    policy_type = load_policy_type(policy_config_path)
    if policy_type == "flow":
        raw_config = _cached_yaml_config(policy_config_path)
        model_section = raw_config.get("model", raw_config)
        hidden_dims_raw = model_section.get("hidden_dims") if isinstance(model_section, Mapping) else None
        if isinstance(hidden_dims_raw, list) and hidden_dims_raw:
            return tuple(int(width) for width in hidden_dims_raw)
        return DEFAULT_FLOW_HIDDEN_DIMS

    validated = load_and_validate_mlp_architecture_config(policy_config_path)
    return validated.hidden_dims


def load_policy_type(policy_config_path: Path | None) -> str:
    if policy_config_path is None:
        return "mlp"
    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if not isinstance(model_section, Mapping):
        raise ValueError("Model config 'model' section must be a mapping.")
    policy_type_raw = model_section.get("policy_type", "mlp")
    if not isinstance(policy_type_raw, str) or policy_type_raw.strip().lower() not in {"mlp", "flow"}:
        raise ValueError("'model.policy_type' must be one of {'mlp', 'flow'}.")
    return policy_type_raw.strip().lower()


def load_flow_config(policy_config_path: Path | None) -> FlowConfig:
    if policy_config_path is None:
        return FlowConfig()

    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if not isinstance(model_section, Mapping):
        raise ValueError("Model config 'model' section must be a mapping.")

    flow_section = model_section.get("flow", {})
    if not isinstance(flow_section, Mapping):
        raise ValueError("Model config 'model.flow' must be a mapping.")

    return FlowConfig(
        num_inference_steps=int(flow_section.get("num_inference_steps", 10)),
        observation_horizon=int(model_section.get("observation_horizon", 1)),
    )


def load_prediction_horizon(policy_config_path: Path | None) -> int:
    if policy_config_path is None:
        return 1
    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if isinstance(model_section, Mapping):
        return int(model_section.get("prediction_horizon", 1))
    return 1


def load_observation_horizon(policy_config_path: Path | None) -> int:
    if policy_config_path is None:
        return 1
    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if not isinstance(model_section, Mapping):
        raise ValueError("Policy config model section must be a mapping.")
    horizon = int(model_section.get("observation_horizon", 1))
    if horizon <= 0:
        raise ValueError("'model.observation_horizon' must be positive.")
    return horizon


def load_encoder_config(policy_config_path: Path | None) -> EncoderConfig:
    if policy_config_path is None:
        return EncoderConfig(encoder_type=DEFAULT_ENCODER_TYPE, kwargs={})

    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if not isinstance(model_section, Mapping):
        raise ValueError("Policy config model section must be a mapping.")

    encoder_type_raw = model_section.get("encoder", DEFAULT_ENCODER_TYPE)
    if not isinstance(encoder_type_raw, str) or not encoder_type_raw.strip():
        raise ValueError("Policy config 'model.encoder' must be a non-empty string.")

    normalized_type = encoder_type_raw.strip().lower()
    if normalized_type == DEFAULT_ENCODER_TYPE:
        raw_kwargs = model_section.get(DEFAULT_ENCODER_TYPE, {})
        if not isinstance(raw_kwargs, Mapping):
            raise ValueError("Policy config 'model.deepset' must be a mapping.")
        kwargs: dict[str, object] = {
            "phi_dims": tuple(int(width) for width in raw_kwargs.get("phi_dims", (128, 128))),
            "rho_dims": tuple(int(width) for width in raw_kwargs.get("rho_dims", (128,))),
            "pool_type": str(raw_kwargs.get("pool_type", "max")),
        }
        return EncoderConfig(encoder_type=normalized_type, kwargs=kwargs)

    if normalized_type == "transformer":
        raw_kwargs = model_section.get("transformer", {})
        if not isinstance(raw_kwargs, Mapping):
            raise ValueError("Policy config 'model.transformer' must be a mapping.")
        kwargs = {
            "hidden_dim": int(raw_kwargs.get("hidden_dim", 64)),
            "num_heads": int(raw_kwargs.get("num_heads", 4)),
            "num_layers": int(raw_kwargs.get("num_layers", 1)),
            "dropout": float(raw_kwargs.get("dropout", 0.1)),
        }
        return EncoderConfig(encoder_type=normalized_type, kwargs=kwargs)

    return EncoderConfig(encoder_type=normalized_type, kwargs={})


def default_checkpoint_dir_for_system(system: str) -> Path:
    if system == "multi_robot":
        return Path("outputs/train_dagger_multi_robot")
    return Path("outputs/train_dagger")


def default_repo_id_for_system(system: str, timestamp: int) -> str:
    return f"local/{system}_dagger_{timestamp}"


def default_dataset_root_for_system(system: str, timestamp: int) -> Path:
    return Path(f"data/lerobot_dataset_{system}_dagger_{timestamp}")
