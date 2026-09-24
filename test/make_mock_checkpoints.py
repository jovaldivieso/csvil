"""Write untrained checkpoints, so the evaluation pipeline can be tested without training.

A real grid costs days of expert time, and checkpoints from before the observation layout
changed no longer load (their ``state_dim`` is 16/36/56/76 where the current schema gives
20/40/60/80). This builds the same metadata checkpoint ``learning/train_dagger.py`` writes,
with the real encoder and policy classes and the dimensions derived from the simulator, but
with the weights as initialized.

The numbers such a checkpoint produces are meaningless -- an untrained policy reaches no
goal -- so this is for the plumbing and the plot layout only, never for results.

Usage:
    python test/make_mock_checkpoints.py
    python test/make_mock_checkpoints.py --experiment mock --fleets 2 4 --encoders deepset
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_and_validate_system_config  # noqa: E402
from core.factory import DynamicsFactory  # noqa: E402
from learning.config_loaders import (  # noqa: E402
    load_encoder_config,
    load_mlp_hidden_dims,
    load_observation_horizon,
    load_prediction_horizon,
)
from learning.models.encoder import EncoderFactory  # noqa: E402
from learning.models.mlp_policy import MLPPolicy  # noqa: E402

POLICY_CONFIG = "learning/config/study/data_mid_n{n:02d}/{encoder}_mlp.yaml"
EXPERT_CONFIG = "test/config/study/unicycle2_fleet_{n:02d}.yaml"


def observation_dimensions(simulator, observation_horizon: int) -> tuple[int, int, int, int]:
    """(state_dim, action_dim, neighbor_slots, neighbor_feature_dim) as DaggerTrainer.setup.

    Kept identical to learning/dagger/dagger_trainer.py: a mock whose widths differ from
    the trainer's would be rejected by the evaluator for the very reason this script
    exists -- to prove that rejection does not happen.
    """
    features = simulator.get_dataset_features()
    environment_state_dim = int(features["observation.environment_state"]["shape"][0])
    proprioception_dim = int(features["observation.state"]["shape"][0])
    state_mask_dim = int(features["observation.state_mask"]["shape"][0])
    base_ego_dim = environment_state_dim + (proprioception_dim + state_mask_dim) * observation_horizon

    action_dim = int(features["action"]["shape"][0])
    neighbor_slots = max(0, int(simulator.num_robots) - 1)
    neighbor_state_dim = int(features["observation.neighbor_state"]["shape"][0])
    if neighbor_slots > 0:
        neighbor_feature_dim = (neighbor_state_dim // neighbor_slots) * observation_horizon
        stacked_neighbor_mask_dim = neighbor_slots * observation_horizon
    else:
        neighbor_feature_dim = max(1, neighbor_state_dim) * observation_horizon
        stacked_neighbor_mask_dim = 0

    state_dim = base_ego_dim + neighbor_slots * neighbor_feature_dim + stacked_neighbor_mask_dim
    return state_dim, action_dim, neighbor_slots, neighbor_feature_dim


def write_mock(encoder: str, num_robots: int, experiment: str, seed: int) -> Path:
    policy_config = PROJECT_ROOT / POLICY_CONFIG.format(n=num_robots, encoder=encoder)
    expert_config = PROJECT_ROOT / EXPERT_CONFIG.format(n=num_robots)
    if not policy_config.exists():
        raise SystemExit(f"no policy config at {policy_config}")

    validated = load_and_validate_system_config("multi_robot", expert_config)
    simulator = DynamicsFactory.create(system_name="multi_robot", config=validated)

    observation_horizon = load_observation_horizon(policy_config)
    prediction_horizon = load_prediction_horizon(policy_config)
    hidden_dims = load_mlp_hidden_dims(policy_config)
    encoder_config = load_encoder_config(policy_config)
    state_dim, action_dim, neighbor_slots, neighbor_feature_dim = observation_dimensions(
        simulator, observation_horizon
    )

    torch.manual_seed(seed + num_robots)
    obs_encoder = EncoderFactory.create(
        encoder_config.encoder_type,
        state_dim,
        neighbor_feature_dim,
        neighbor_slots,
        observation_horizon=observation_horizon,
        **encoder_config.kwargs,
    )
    policy = MLPPolicy(
        action_dim=action_dim,
        obs_encoder=obs_encoder,
        hidden_dims=tuple(hidden_dims),
        prediction_horizon=prediction_horizon,
    )

    run = f"{encoder}_mlp_unicycle2_fleet_{num_robots:02d}_s{seed}"
    out_dir = PROJECT_ROOT / "outputs" / experiment / "models" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mlp_dagger_checkpoint.pt"
    torch.save(
        {
            "iteration": 0,
            "model_state_dict": policy.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "prediction_horizon": prediction_horizon,
            "observation_horizon": observation_horizon,
            "hidden_dims": list(hidden_dims),
            "system": "multi_robot",
            "neighbor_feature_dim": neighbor_feature_dim,
            "neighbor_slots": neighbor_slots,
            "encoder_type": encoder_config.encoder_type,
            "encoder_kwargs": encoder_config.kwargs,
            "policy_type": "mlp",
            "mock": True,
        },
        out_path,
    )
    print(f"wrote {out_path.relative_to(PROJECT_ROOT)} "
          f"(state_dim {state_dim}, slots {neighbor_slots}, {encoder})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="mock", help="outputs/<experiment>/models/")
    parser.add_argument("--fleets", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--encoders", nargs="+", default=["deepset", "transformer", "gnn"])
    parser.add_argument("--seed", type=int, default=0, help="run-name seed suffix")
    args = parser.parse_args()

    for num_robots in args.fleets:
        for encoder in args.encoders:
            write_mock(encoder, num_robots, args.experiment, args.seed)


if __name__ == "__main__":
    main()
