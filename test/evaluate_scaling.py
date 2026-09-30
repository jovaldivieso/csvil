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
    --episodes 50 --steps 200 --output-csv outputs/study2/eval/random/deepset_n02.csv
"""

from __future__ import annotations

import argparse
import csv
import math
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
from learning.models.policy import ActionPolicy, PolicyFactory
from planning.casadi_planner import PlannerSolveError
from systems.dynamics import DynamicsProtocol
from systems.goal_metrics import fleet_goal_errors
from systems.seed_utils import (
    action_noise_rng_for_rollout, default_action_noise_seed_for_config,
)

TOLERANCE_FIELDS = ("pos_tol", "theta_tol", "vel_tol", "omega_tol")

CSV_FIELDS = (
    "checkpoint", "encoder_type", "policy_type", "train_seed", "train_fleet_size", "eval_fleet_size",
    "density", "config", "episodes", "steps", "action_noise_std", *TOLERANCE_FIELDS,
    "success_rate", "collision_rate", "timeout_rate",
    "infeasible_rate",
    "robot_success_rate", "robot_collision_rate", "robot_timeout_rate",
    "mean_steps", "mean_goal_position_error",
    "mean_goal_heading_error", "mean_min_pair_distance",
    "min_min_pair_distance", "p05_min_pair_distance",
    "mean_visible_neighbours", "mean_action_ms", "mean_action_ms_per_robot", "wall_time_s",
)


def read_checkpoint(checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    """Load and sanity-check a train_dagger.py metadata checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"'{checkpoint_path}' is not a train_dagger.py metadata checkpoint.")
    raw_horizon = checkpoint.get("observation_horizon", 1)
    observation_horizon = 1 if raw_horizon is None else int(raw_horizon)
    if observation_horizon <= 0:
        raise ValueError("Checkpoint 'observation_horizon' must be positive.")
    checkpoint["observation_horizon"] = observation_horizon
    return checkpoint


def build_policy(
    checkpoint: dict[str, Any],
    device: torch.device,
    simulator: DynamicsProtocol | None = None,
    planner_config: Mapping[str, Any] | None = None,
) -> ActionPolicy:
    """Rebuild a policy from checkpoint metadata."""
    observation_horizon = int(checkpoint.get("observation_horizon", 1))

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
    if policy_type in {"flow", "safeflow"}:
        flow_config = checkpoint.get("flow_config") or {}
        flow_kwargs: dict[str, Any] = {
            "num_inference_steps": int(flow_config.get("num_inference_steps", 10)),
        }
        # action_scale is a non-persistent buffer, so it is missing from model_state_dict.
        if flow_config.get("action_scale") is not None:
            flow_kwargs["action_scale"] = list(flow_config["action_scale"])

        if policy_type == "safeflow":
            if simulator is None or planner_config is None:
                raise ValueError(
                    "policy_type 'safeflow' needs 'simulator' and 'planner_config' to "
                    "build its CasADi projector. Its projectors are sized to the fleet "
                    "(one Opti per robot), so it must be rebuilt per evaluation config "
                    "rather than once per checkpoint."
                )
            policy = PolicyFactory.create(
                "safeflow", **shared, **flow_kwargs,
                simulator=simulator, planner_config=planner_config,
            )
        else:
            policy = FlowPolicy(**shared, **flow_kwargs)
    elif policy_type == "mlp":
        policy = MLPPolicy(**shared)
    else:
        raise ValueError(f"Unsupported policy type '{policy_type}'.")

    policy.load_state_dict(checkpoint["model_state_dict"])
    return policy.to(device).eval()


def load_policy(
    checkpoint_path: str,
    device: torch.device,
    simulator: DynamicsProtocol | None = None,
    planner_config: Mapping[str, Any] | None = None,
) -> tuple[ActionPolicy, dict[str, Any]]:
    """Read a checkpoint and rebuild its policy, for callers that want both."""
    checkpoint = read_checkpoint(checkpoint_path, device)
    policy = build_policy(checkpoint, device, simulator, planner_config)
    return policy, checkpoint


def min_pair_distance(simulator: DynamicsProtocol, state: np.ndarray) -> float:
    positions = np.stack([
        np.asarray(robot_state)[list(simulator.simulators[0].position_indices)]
        for robot_state in np.split(np.asarray(state), simulator.num_robots)
    ])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def collided_robots(simulator: DynamicsProtocol, state: np.ndarray) -> np.ndarray:
    """Per-robot mask: is this robot within d_collision of any other right now?"""
    positions = np.stack([
        np.asarray(robot_state)[list(simulator.simulators[0].position_indices)]
        for robot_state in np.split(np.asarray(state), simulator.num_robots)
    ])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1) < float(simulator.d_collision)


def robots_at_goal(simulator: DynamicsProtocol, state: np.ndarray) -> np.ndarray:
    """Per-robot mask of is_done, the same criterion should_terminate_rollout uses."""
    state_array = np.asarray(state, dtype=float)
    return np.array([
        bool(sub.is_done(state_array[state_slice], validate=False))
        for sub, state_slice in zip(simulator.simulators, simulator.robot_state_slices)
    ])


def visible_neighbours(simulator: DynamicsProtocol, state: np.ndarray) -> float:
    """Mean number of neighbours inside the sensing radius, over the fleet."""
    positions = np.stack([
        np.asarray(robot_state)[list(simulator.simulators[0].position_indices)]
        for robot_state in np.split(np.asarray(state), simulator.num_robots)
    ])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    radii = np.asarray(simulator.robot_visibility_radii, dtype=float).reshape(-1, 1)
    return float((distances < radii).sum(axis=1).mean())


def tolerance_columns(simulator: DynamicsProtocol) -> dict[str, float | str]:
    """The fleet's convergence tolerances, read off robot 0 (fleets are homogeneous)."""
    robot_simulator = simulator.simulators[0]
    return {name: getattr(robot_simulator, name, "") for name in TOLERANCE_FIELDS}


STEP_BUDGET_FACTOR = 2.5


def robot_density(
    simulator: DynamicsProtocol,
    fixed_initial_state: np.ndarray | None,
) -> float:
    """Robots per m^2 of the area the scenario places them in."""
    robot_simulator = simulator.simulators[0]
    if fixed_initial_state is not None:
        positions = np.concatenate([
            np.stack([
                np.asarray(robot_state)[list(robot_simulator.position_indices)]
                for robot_state in np.split(np.asarray(fixed_initial_state), simulator.num_robots)
            ]),
            np.stack([
                np.asarray(goal)[:2]
                for goal in np.split(np.asarray(simulator.goal_state), simulator.num_robots)
            ]),
        ])
        side = float(np.max(positions.max(axis=0) - positions.min(axis=0)))
    else:
        low, high = robot_simulator.workspace_bounds
        side = float(high - low)
    if side <= 0.0:
        return 0.0
    return round(float(simulator.num_robots) / side ** 2, 4)


def step_budget(
    simulator: DynamicsProtocol,
    factor: float,
    fixed_initial_state: np.ndarray | None,
) -> int:
    """Steps allowed per episode, from the longest distance the config can produce."""
    robot_simulator = simulator.simulators[0]
    position_indices = list(robot_simulator.position_indices)
    if fixed_initial_state is not None:
        starts = np.split(np.asarray(fixed_initial_state), simulator.num_robots)
        goals = np.split(np.asarray(simulator.goal_state), simulator.num_robots)
        max_travel = max(
            float(np.linalg.norm(np.asarray(start)[position_indices] - np.asarray(goal)[:2]))
            for start, goal in zip(starts, goals)
        )
    else:
        low, high = robot_simulator.workspace_bounds
        max_travel = float(np.hypot(high - low, high - low))
    speed = float(robot_simulator.max_linear_vel)
    return int(math.ceil(factor * max_travel / (speed * float(robot_simulator.dt))))


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
    stop_on_collision: bool = False,
) -> dict[str, float]:
    """Roll the policy out over seeded episodes and summarize the outcomes."""
    successes = collisions = infeasibles = 0
    robot_successes = robot_collisions = robot_total = 0
    action_times_ms: list[float] = []
    steps_taken: list[int] = []
    position_errors: list[float] = []
    heading_errors: list[float] = []
    min_distances: list[float] = []
    visible_counts: list[float] = []

    for episode_index, seed_spec in enumerate(evaluation_seed_specs(simulator, episodes, seed_start)):
        # Flow samples actions from noise; seeding per episode makes runs reproducible.
        torch.manual_seed(seed_start + episode_index)
        if fixed_initial_state is not None:
            state = simulator.reset(fixed_initial_state.copy())
        else:
            state = simulator.reset(sample_initial_state(simulator, seed_spec))
        goal_state = simulator.goal_state.copy()
        history_buffer = (
            ObservationHistoryBuffer(observation_horizon, int(simulator.num_robots))
            if observation_horizon > 1 else None
        )
        noise_rng = action_noise_rng_for_rollout(action_noise_seed, seed_spec=seed_spec)
        episode_min_distance = min_pair_distance(simulator, state)
        episode_visible = [visible_neighbours(simulator, state)]
        ever_collided = collided_robots(simulator, state)
        reached_goal = collided = False
        rollout_steps = 0
        infeasible = False

        for step in range(1, steps + 1):
            observation = simulator.observe(state, validate=False)
            # CUDA launches are async; sync so the timer measures compute, not queueing.
            if device.type == "cuda":
                torch.cuda.synchronize()
            action_start = time.perf_counter()
            try:
                action = build_decentralized_joint_action(
                    simulator, policy, observation, device,
                    observation_horizon=observation_horizon, history_buffer=history_buffer,
                )
            except PlannerSolveError:
                # safeflow's projector could not certify an action: an outcome, not a crash.
                infeasible = True
                break
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
            episode_visible.append(visible_neighbours(simulator, state))
            ever_collided |= collided_robots(simulator, state)
            if ever_collided.any():
                collided = True
                if stop_on_collision:
                    break
            if simulator.should_terminate_rollout(state):
                reached_goal = True
                break

        successes += int(reached_goal and not collided)
        collisions += int(collided)
        infeasibles += int(infeasible)
        at_goal = robots_at_goal(simulator, state)
        robot_successes += int((at_goal & ~ever_collided).sum())
        robot_collisions += int(ever_collided.sum())
        robot_total += int(simulator.num_robots)
        steps_taken.append(rollout_steps)
        position_error, heading_error = fleet_goal_errors(simulator, state, goal_state)
        position_errors.append(position_error)
        heading_errors.append(heading_error)
        min_distances.append(episode_min_distance)
        visible_counts.append(float(np.mean(episode_visible)))

    return {
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "timeout_rate": (episodes - successes - collisions) / episodes,
        "infeasible_rate": infeasibles / episodes,
        "robot_success_rate": robot_successes / robot_total,
        "robot_collision_rate": robot_collisions / robot_total,
        "robot_timeout_rate": (robot_total - robot_successes - robot_collisions) / robot_total,
        "mean_steps": float(np.mean(steps_taken)),
        "mean_goal_position_error": float(np.mean(position_errors)),
        "mean_goal_heading_error": float(np.mean(heading_errors)),
        "mean_min_pair_distance": float(np.mean(min_distances)),
        "min_min_pair_distance": float(np.min(min_distances)),
        "p05_min_pair_distance": float(np.percentile(min_distances, 5)),
        "mean_visible_neighbours": float(np.mean(visible_counts)),
        "mean_action_ms": float(np.mean(action_times_ms)),
        "mean_action_ms_per_robot": float(np.mean(action_times_ms)) / float(simulator.num_robots),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="metadata .pt written by learning/train_dagger.py")
    parser.add_argument("--configs", nargs="+", required=True, help="one multi_robot YAML config per fleet size")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--steps", type=int,
        help="fixed step budget for every config; mutually exclusive with --step-budget-factor",
    )
    parser.add_argument(
        "--step-budget-factor", type=float,
        help=(
            "derive the step budget per config from the distances it produces: "
            f"{STEP_BUDGET_FACTOR} is the usual setting (see step_budget)"
        ),
    )
    parser.add_argument("--seed-start", type=int, default=50000, help="disjoint from the trainer's --eval-seed-start")
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--stop-on-collision", action="store_true",
        help="end an episode at the first collision. Cheaper, but the per-robot rates then "
             "count every robot of the fleet as failed because two of them touched",
    )
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
    if args.steps is not None and args.step_budget_factor is not None:
        parser.error("pass either --steps or --step-budget-factor, not both.")
    if args.steps is None and args.step_budget_factor is None:
        args.step_budget_factor = STEP_BUDGET_FACTOR

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = read_checkpoint(args.checkpoint, device)
    policy_type = str(checkpoint.get("policy_type", "mlp")).lower()
    # safeflow builds one CasADi Opti per robot, so it is rebuilt per config below.
    policy = None if policy_type == "safeflow" else build_policy(checkpoint, device)
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
        policy_is_deterministic = policy_type not in {"flow", "safeflow"}
        if (
            args.use_config_start
            and args.action_noise_std == 0.0
            and episodes > 1
            and policy_is_deterministic
        ):
            print(
                "  note: --use-config-start with no action noise is deterministic for "
                f"this policy; collapsing {episodes} identical episodes to 1"
            )
            episodes = 1

        for config_path in args.configs:
            config = load_and_validate_system_config("multi_robot", config_path)
            simulator = DynamicsFactory.create(system_name="multi_robot", config=config)
            if policy_type == "safeflow":
                policy = build_policy(
                    checkpoint, device, simulator=simulator, planner_config=config
                )
            fixed_start = config_start_state(config) if args.use_config_start else None
            if fixed_start is not None and simulator.is_collision(fixed_start):
                raise SystemExit(f"{config_path}: configured start state is already in collision.")
            steps = (
                args.steps
                if args.steps is not None
                else step_budget(simulator, args.step_budget_factor, fixed_start)
            )
            start_time = time.perf_counter()
            metrics = evaluate_fleet(
                simulator=simulator,
                policy=policy,
                device=device,
                episodes=episodes,
                steps=steps,
                seed_start=args.seed_start,
                action_noise_std=args.action_noise_std,
                action_noise_seed=default_action_noise_seed_for_config(config),
                fixed_initial_state=fixed_start,
                observation_horizon=int(checkpoint.get("observation_horizon", 1)),
                stop_on_collision=args.stop_on_collision,
            )
            row = {
                "checkpoint": args.checkpoint,
                "encoder_type": checkpoint.get("encoder_type"),
                "policy_type": checkpoint.get("policy_type"),
                "train_seed": args.train_seed,
                "train_fleet_size": train_fleet_size,
                "eval_fleet_size": int(simulator.num_robots),
                "density": robot_density(simulator, fixed_start),
                "config": config_path,
                "episodes": episodes,
                "steps": steps,
                "action_noise_std": args.action_noise_std,
                **tolerance_columns(simulator),
                "wall_time_s": round(time.perf_counter() - start_time, 2),
                **{key: round(value, 6) for key, value in metrics.items()},
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"  eval_fleet={row['eval_fleet_size']:>2}  "
                f"success={metrics['success_rate']:.3f} (robots {metrics['robot_success_rate']:.3f})  "
                f"collision={metrics['collision_rate']:.3f}  "
                f"timeout={metrics['timeout_rate']:.3f}  mean_steps={metrics['mean_steps']:.1f}  "
                f"pos_err={metrics['mean_goal_position_error']:.3f}  "
                f"head_err={metrics['mean_goal_heading_error']:.3f}  "
                f"min_pair_dist={metrics['mean_min_pair_distance']:.3f}  "
                f"action={metrics['mean_action_ms']:.2f}ms  ({row['wall_time_s']}s)"
            )

    print(f"\nwrote {args.output_csv}")


if __name__ == "__main__":
    main()
