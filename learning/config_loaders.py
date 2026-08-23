from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

from core.config import load_and_validate_mlp_architecture_config, load_yaml_config
from learning.models.encoder import DEFAULT_ENCODER_TYPE

DEFAULT_MLP_HIDDEN_DIMS: tuple[int, ...] = (256, 256, 128)
DEFAULT_FLOW_HIDDEN_DIMS: tuple[int, ...] = (256, 256, 256)


@dataclass(frozen=True)
class EncoderConfig:
    encoder_type: str
    kwargs: dict[str, object]


@dataclass(frozen=True)
class FlowConfig:
    num_inference_steps: int = 10


def load_mlp_hidden_dims(policy_config_path: Path | None) -> tuple[int, ...]:
    if policy_config_path is None:
        return DEFAULT_MLP_HIDDEN_DIMS

    policy_type = load_policy_type(policy_config_path)
    if policy_type == "flow":
        raw_config = load_yaml_config(policy_config_path)
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
    raw_config = load_yaml_config(policy_config_path)
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

    raw_config = load_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if not isinstance(model_section, Mapping):
        raise ValueError("Model config 'model' section must be a mapping.")

    flow_section = model_section.get("flow", {})
    if not isinstance(flow_section, Mapping):
        raise ValueError("Model config 'model.flow' must be a mapping.")

    return FlowConfig(num_inference_steps=int(flow_section.get("num_inference_steps", 10)))


def load_prediction_horizon(policy_config_path: Path | None) -> int:
    if policy_config_path is None:
        return 1
    raw_config = load_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if isinstance(model_section, Mapping):
        return int(model_section.get("prediction_horizon", 1))
    return 1


def load_encoder_config(policy_config_path: Path | None) -> EncoderConfig:
    if policy_config_path is None:
        return EncoderConfig(encoder_type=DEFAULT_ENCODER_TYPE, kwargs={})

    raw_config = load_yaml_config(policy_config_path)
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

    return EncoderConfig(encoder_type=normalized_type, kwargs={})


def default_checkpoint_dir_for_system(system: str) -> Path:
    if system == "multi_robot":
        return Path("outputs/train_dagger_multi_robot")
    return Path("outputs/train_dagger")


def default_repo_id_for_system(system: str, timestamp: int) -> str:
    return f"local/{system}_dagger_{timestamp}"


def default_dataset_root_for_system(system: str, timestamp: int) -> Path:
    return Path(f"data/lerobot_dataset_{system}_dagger_{timestamp}")
