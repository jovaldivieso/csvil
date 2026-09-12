"""Evaluate one trained decentralized policy across several fleet sizes.

``test/evaluate_policy.py`` rolls the CasADi expert out alongside the policy for
every seed, which is what makes its side-by-side plots possible but also makes it
unusable past roughly eight robots (the MPC solve grows super-quadratically in the
fleet size). This CLI runs the policy only, so a checkpoint trained on a small fleet
can be benchmarked on much larger ones, and appends one row per fleet size to a CSV.

Seeding matches the in-training evaluation in ``learning/dagger/rollouts.py``:
the same ``--eval-seed-start`` and fleet config reproduce the same episodes.

Usage:
    python test/evaluate_scaling.py \
    --checkpoint outputs/train_dagger_multi_robot/deepset_n02/mlp_dagger_checkpoint.pt \
    --configs test/config/study/unicycle2_fleet_*.yaml \
    --episodes 50 --steps 200 --output-csv outputs/study/deepset_n02.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config
from core.factory import DynamicsFactory
from learning.dagger import (
    ObservationHistoryBuffer, apply_execution_noise, build_decentralized_joint_action,
    evaluation_seed_specs, sample_initial_state,
)
from learning.models.encoder import DEFAULT_ENCODER_TYPE, EncoderFactory
from learning.models.flow_policy import FlowPolicy
from learning.models.mlp_policy import MLPPolicy
from learning.models.policy import ActionPolicy
from systems.dynamics import DynamicsProtocol
from systems.goal_metrics import fleet_goal_errors
from systems.seed_utils import (
    action_noise_rng_for_rollout, default_action_noise_seed_for_config,
)

# Convergence tolerances are written into every row so a result is self-describing:
# they define what "success" means, and a silent change to them moves success_rate
# without moving anything about the policy. unicycle2's four; a system that uses a
# single 'error_tolerance' instead leaves these blank.
TOLERANCE_FIELDS = ("pos_tol", "theta_tol", "vel_tol", "omega_tol")

CSV_FIELDS = (
    "checkpoint", "encoder_type", "policy_type", "train_seed", "train_fleet_size", "eval_fleet_size",
    "config", "episodes", "steps", "action_noise_std", *TOLERANCE_FIELDS,
    "success_rate",
    "collision_rate", "timeout_rate", "mean_steps", "mean_goal_position_error",
    "mean_goal_heading_error", "mean_min_pair_distance",
    "mean_action_ms", "mean_action_ms_per_robot", "wall_time_s",
)


def load_policy(checkpoint_path: str, device: torch.device) -> tuple[ActionPolicy, dict[str, Any]]:
    """Rebuild a policy from a train_dagger.py metadata checkpoint.

    ``state_dim`` and ``neighbor_slots`` are taken from the checkpoint rather than
    from the evaluation simulator: the encoders derive their ego-feature width from
    that pair, and only the training-time pair reproduces the trained shapes. The
    neighbour branch itself is slot-count agnostic, so the rebuilt policy accepts
    any fleet size at rollout time.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"'{checkpoint_path}' is not a train_dagger.py metadata checkpoint.")

    # Observation history widens each neighbour's feature vector and the ego block,
    # so the encoder cannot be rebuilt without it. Older checkpoints predate the key.
    raw_horizon = checkpoint.get("observation_horizon", 1)
    observation_horizon = 1 if raw_horizon is None else int(raw_horizon)
    if observation_horizon <= 0:
        raise ValueError("Checkpoint 'observation_horizon' must be positive.")

    encoder_kwargs_raw = checkpoint.get("encoder_kwargs") or {}
    encoder = EncoderFactory.create(
        encoder_type=str(checkpoint.get("encoder_type", DEFAULT_ENCODER_TYPE)),
        state_dim=int(checkpoint["state_dim"]),
        neighbor_feature_dim=int(checkpoint.get("neighbor_feature_dim", 2)),
        neighbor_slots=int(checkpoint["neighbor_slots"]),
        observation_horizon=observation_horizon,
        **(dict(encoder_kwargs_raw) if isinstance(encoder_kwargs_raw, Mapping) else {}),
    )

    policy_type = str(checkpoint.get("policy_type", "mlp"))
    shared = {
        "action_dim": int(checkpoint["action_dim"]),
        "obs_encoder": encoder,
        "hidden_dims": tuple(int(width) for width in checkpoint["hidden_dims"]),
        "prediction_horizon": int(checkpoint.get("prediction_horizon", 1)),
    }
    if policy_type == "flow":
        flow_config = checkpoint.get("flow_config") or {}
        policy = FlowPolicy(**shared, num_inference_steps=int(flow_config.get("num_inference_steps", 10)))
    elif policy_type == "mlp":
        policy = MLPPolicy(**shared)
    else:
        raise ValueError(f"Unsupported policy type '{policy_type}'.")

    policy.load_state_dict(checkpoint["model_state_dict"])
    checkpoint["observation_horizon"] = observation_horizon
    return policy.to(device).eval(), checkpoint


def min_pair_distance(simulator: DynamicsProtocol, state: np.ndarray) -> float:
    positions = np.stack([
        np.asarray(robot_state)[list(simulator.simulators[0].position_indices)]
        for robot_state in np.split(np.asarray(state), simulator.num_robots)
    ])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def tolerance_columns(simulator: DynamicsProtocol) -> dict[str, float | str]:
    """The fleet's convergence tolerances, read off robot 0 (fleets are homogeneous)."""
    robot_simulator = simulator.simulators[0]
    return {name: getattr(robot_simulator, name, "") for name in TOLERANCE_FIELDS}


def config_start_state(raw_config: Mapping[str, Any]) -> np.ndarray:
    """Concatenate the per-robot 'start' entries into one fleet state."""
    robots = raw_config.get("robots")
    if not isinstance(robots, list) or not robots:
        raise ValueError("--use-config-start needs a non-empty 'robots' list.")
    starts = []
    for robot_idx, robot_entry in enumerate(robots):
        start = robot_entry.get("start")
        if start is None:
            robot_config = robot_entry.get("config")
            start = robot_config.get("start") if isinstance(robot_config, Mapping) else None
        if start is None:
            raise ValueError(
                f"--use-config-start given, but robots[{robot_idx}] defines no 'start'."
            )
        starts.append(np.asarray(start, dtype=np.float32))
    return np.concatenate(starts)


def evaluate_fleet(
    simulator: DynamicsProtocol,
    policy: ActionPolicy,
    device: torch.device,
    episodes: int,
    steps: int,
    seed_start: int,
    action_noise_std: float,
    action_noise_seed: int,
    fixed_initial_state: np.ndarray | None = None,
    observation_horizon: int = 1,
) -> dict[str, float]:
    """Roll the policy out over seeded episodes and summarize the outcomes.

    With ``fixed_initial_state`` the layout is identical every episode, so the only
    thing separating episodes is the action-noise draw. Without noise the scenario
    is fully deterministic and one episode is the whole result.
    """
    successes = collisions = 0
    # One entry per control step: the wall time of the policy call that produced that
    # step's joint action. Kept separate from wall_time_s, which also covers the
    # simulator, the collision checks and the observation construction.
    action_times_ms: list[float] = []
    steps_taken: list[int] = []
    position_errors: list[float] = []
    heading_errors: list[float] = []
    min_distances: list[float] = []

    for seed_spec in evaluation_seed_specs(simulator, episodes, seed_start):
        if fixed_initial_state is not None:
            state = simulator.reset(fixed_initial_state.copy())
        else:
            state = simulator.reset(sample_initial_state(simulator, seed_spec))
        goal_state = simulator.goal_state.copy()
        # Fresh per episode: history must not leak across rollouts.
        history_buffer = (
            ObservationHistoryBuffer(observation_horizon, int(simulator.num_robots))
            if observation_horizon > 1 else None
        )
        noise_rng = action_noise_rng_for_rollout(action_noise_seed, seed_spec=seed_spec)
        episode_min_distance = min_pair_distance(simulator, state)
        reached_goal = collided = False
        rollout_steps = 0

        for step in range(1, steps + 1):
            observation = simulator.observe(state, validate=False)
            # CUDA launches are async, so the timer would measure queueing rather than
            # compute without a sync on either side.
            if device.type == "cuda":
                torch.cuda.synchronize()
            action_start = time.perf_counter()
            action = build_decentralized_joint_action(
                simulator, policy, observation, device,
                observation_horizon=observation_horizon, history_buffer=history_buffer,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            action_times_ms.append((time.perf_counter() - action_start) * 1000.0)
            state = simulator.step(
                state,
                apply_execution_noise(simulator, action, action_noise_std, noise_rng),
                validate=False,
            )
            rollout_steps = step
            episode_min_distance = min(episode_min_distance, min_pair_distance(simulator, state))
            if simulator.is_collision(state):
                collided = True
                break
            if simulator.should_terminate_rollout(state):
                reached_goal = True
                break

        successes += int(reached_goal)
        collisions += int(collided)
        steps_taken.append(rollout_steps)
        # Split by coordinate geometry: a raw L2 over the state vector scores a
        # correct-but-wrapped heading as an error of 2*pi. See systems/goal_metrics.py.
        position_error, heading_error = fleet_goal_errors(simulator, state, goal_state)
        position_errors.append(position_error)
        heading_errors.append(heading_error)
        min_distances.append(episode_min_distance)

    return {
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "timeout_rate": (episodes - successes - collisions) / episodes,
        "mean_steps": float(np.mean(steps_taken)),
        "mean_goal_position_error": float(np.mean(position_errors)),
        "mean_goal_heading_error": float(np.mean(heading_errors)),
        "mean_min_pair_distance": float(np.mean(min_distances)),
        # The fleet is one batched forward pass, so this is the latency of a whole
        # control step, not of a single robot deciding on its own hardware. The
        # per-robot figure divides that batch cost evenly and therefore understates
        # true decentralized latency -- use it to compare policies, not to size a
        # real controller.
        "mean_action_ms": float(np.mean(action_times_ms)),
        "mean_action_ms_per_robot": float(np.mean(action_times_ms)) / float(simulator.num_robots),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="metadata .pt written by learning/train_dagger.py")
    parser.add_argument("--configs", nargs="+", required=True, help="one multi_robot YAML config per fleet size")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=50000, help="disjoint from the trainer's --eval-seed-start")
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--use-config-start",
        action="store_true",
        help=(
            "start every episode from the per-robot 'start' in the config instead of "
            "sampling one, for deterministic scenarios such as the antipodal-circle swap"
        ),
    )
    parser.add_argument(
        "--train-seed", default="",
        help="training seed of this checkpoint, copied into the CSV so results can be "
             "grouped by seed without parsing the checkpoint path",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default=None, help="cpu, cuda, mps; autodetected when omitted")
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    policy, checkpoint = load_policy(args.checkpoint, device)
    train_fleet_size = int(checkpoint["neighbor_slots"]) + 1
    print(
        f"checkpoint: {args.checkpoint}\n"
        f"encoder: {checkpoint.get('encoder_type')} | policy: {checkpoint.get('policy_type')} | "
        f"trained on {train_fleet_size} robots | device: {device}"
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output_csv.exists()
    with args.output_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        episodes = args.episodes
        if args.use_config_start and args.action_noise_std == 0.0 and episodes > 1:
            print(
                "  note: --use-config-start with no action noise is deterministic; "
                f"collapsing {episodes} identical episodes to 1"
            )
            episodes = 1

        for config_path in args.configs:
            config = load_and_validate_system_config("multi_robot", config_path)
            simulator = DynamicsFactory.create(system_name="multi_robot", config=config)
            fixed_start = config_start_state(config) if args.use_config_start else None
            if fixed_start is not None and simulator.is_collision(fixed_start):
                raise SystemExit(f"{config_path}: configured start state is already in collision.")
            start_time = time.perf_counter()
            metrics = evaluate_fleet(
                simulator=simulator,
                policy=policy,
                device=device,
                episodes=episodes,
                steps=args.steps,
                seed_start=args.seed_start,
                action_noise_std=args.action_noise_std,
                action_noise_seed=default_action_noise_seed_for_config(config),
                fixed_initial_state=fixed_start,
                observation_horizon=int(checkpoint.get("observation_horizon", 1)),
            )
            row = {
                "checkpoint": args.checkpoint,
                "encoder_type": checkpoint.get("encoder_type"),
                "policy_type": checkpoint.get("policy_type"),
                "train_seed": args.train_seed,
                "train_fleet_size": train_fleet_size,
                "eval_fleet_size": int(simulator.num_robots),
                "config": config_path,
                "episodes": episodes,
                "steps": args.steps,
                "action_noise_std": args.action_noise_std,
                **tolerance_columns(simulator),
                "wall_time_s": round(time.perf_counter() - start_time, 2),
                **{key: round(value, 6) for key, value in metrics.items()},
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"  eval_fleet={row['eval_fleet_size']:>2}  "
                f"success={metrics['success_rate']:.3f}  collision={metrics['collision_rate']:.3f}  "
                f"timeout={metrics['timeout_rate']:.3f}  mean_steps={metrics['mean_steps']:.1f}  "
                f"pos_err={metrics['mean_goal_position_error']:.3f}  "
                f"head_err={metrics['mean_goal_heading_error']:.3f}  "
                f"min_pair_dist={metrics['mean_min_pair_distance']:.3f}  "
                f"action={metrics['mean_action_ms']:.2f}ms  ({row['wall_time_s']}s)"
            )

    print(f"\nwrote {args.output_csv}")


if __name__ == "__main__":
    main()
