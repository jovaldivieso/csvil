"""Compare multiple trained policy checkpoints head-to-head.

Unlike evaluate_policy.py (one checkpoint vs the expert, with trajectory
plots/videos), this runs any number of policy_type/checkpoint combinations
-- including the same checkpoint evaluated under a different policy_type,
e.g. a 'flow' checkpoint run in 'safeflow' mode -- on matched-size
evaluation splits (a fixed config-scenario set plus randomly sampled
episodes) and reports aggregate success/failure-mode rates plus
per-control-tick inference-time statistics against --dt, to answer whether
each is real-time (per-tick computation time <= dt).

Edit MODES below to choose what to compare; everything else (system,
configs, sample sizes, step budget, dt) is a CLI flag. Example:

    python test/evaluate_checkpoints.py --num-steps 250 --dt 0.05
"""
import os
import sys
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import yaml

from core.config import load_and_validate_system_config, validate_system_config
from core.factory import DynamicsFactory
from learning.dagger import (
    ObservationHistoryBuffer,
    apply_config_overrides,
    build_decentralized_joint_action,
    evaluate_policy_rollouts,
)
from learning.models.policy import PolicyFactory
from planning.planner import PlannerSolveError

from evaluate_policy import _load_checkpoint_policy_components, get_inference_device, _synchronize_device

FLOW_CKPT = "outputs/train_dagger_multi_robot/2_unicycle2_casadi_deepset_flow_v8/flow_dagger_iter_003.pt"
SAFEFLOW_CKPT = "outputs/train_dagger_multi_robot/2_unicycle2_casadi_deepset_safeflow_v8/flow_dagger_iter_001.pt"
MLP_CKPT = "outputs/train_dagger_multi_robot/2_unicycle2_casadi_deepset_mlp_v8/mlp_dagger_iter_002.pt"

MODES: dict[str, tuple[str, str]] = {
    "flow_native": ("flow", FLOW_CKPT),
    "flow_in_safeflow_mode": ("safeflow", FLOW_CKPT),
    "safeflow_native": ("safeflow", SAFEFLOW_CKPT),
    "mlp": ("mlp", MLP_CKPT),
}


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_eval_config(expert_config_path: str, eval_config_path: str, system: str, workspace_bounds: tuple[float, float]):
    with open(eval_config_path) as f:
        training = yaml.safe_load(f)["training"]
    tolerance_overrides = training["eval_tolerance_overrides"]
    initial_states = training["initial_states"]
    goal_states = training["goal_states"]

    raw_expert = load_and_validate_system_config(system, expert_config_path)
    overridden = apply_config_overrides(
        dict(raw_expert),
        {"workspace_bounds": list(workspace_bounds), **tolerance_overrides},
    )
    validated = validate_system_config(system, overridden)
    return validated, initial_states, goal_states


def build_policy(policy_type: str, checkpoint_path: str, simulator, device, validated_config):
    checkpoint, state_dict, obs_encoder, action_dim, hidden_dims, prediction_horizon, observation_horizon = (
        _load_checkpoint_policy_components(checkpoint_path, simulator, policy_type, device)
    )
    flow_config_raw = checkpoint.get("flow_config", {})
    policy_kwargs = {
        "action_dim": action_dim,
        "obs_encoder": obs_encoder,
        "hidden_dims": hidden_dims,
        "prediction_horizon": prediction_horizon,
    }
    if policy_type in {"flow", "safeflow"}:
        policy_kwargs["num_inference_steps"] = int(flow_config_raw.get("num_inference_steps", 10))
        if flow_config_raw.get("action_scale") is not None:
            policy_kwargs["action_scale"] = list(flow_config_raw["action_scale"])
    if policy_type == "safeflow":
        policy_kwargs["simulator"] = simulator
        policy_kwargs["planner_config"] = validated_config
    policy = PolicyFactory.create(policy_type, **policy_kwargs)
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to(device)
    return policy, observation_horizon


def evaluate_mode(
    name: str,
    policy_type: str,
    checkpoint_path: str,
    validated_config,
    initial_states,
    goal_states,
    device: torch.device,
    num_robots: int,
    num_steps: int,
    config_seed_starts: list[int],
    random_seed_starts: list[int],
    random_episodes_per_rep: int,
    dt: float,
) -> dict[str, float]:
    simulator = DynamicsFactory.create(system_name="multi_robot", config=validated_config)
    policy, observation_horizon = build_policy(policy_type, checkpoint_path, simulator, device, validated_config)
    history_buffer = ObservationHistoryBuffer(observation_horizon, int(simulator.num_robots))
    step_times: list[float] = []

    # Untimed warm-up: triggers FlowPolicy.net's lazy torch.compile and (for
    # safeflow) the projector thread pool's lazy spawn, before any timed
    # sample is collected. All-zero state collides every robot at the
    # origin, which is fine for flow (ignored) but can make a cold-start
    # SafeFlow projection infeasible -- both goals of the warm-up (compile,
    # thread spawn) already happen before that failure, so it's safe to
    # swallow.
    try:
        build_decentralized_joint_action(
            simulator, policy, simulator.observe(np.zeros(simulator.nx)), device,
            observation_horizon=observation_horizon, history_buffer=None if observation_horizon <= 1 else history_buffer,
        )
    except PlannerSolveError:
        pass
    policy.reset()
    history_buffer.reset()

    def action_fn(obs: np.ndarray) -> np.ndarray:
        _synchronize_device(device)
        t0 = time.perf_counter()
        action = build_decentralized_joint_action(
            simulator, policy, obs, device,
            observation_horizon=observation_horizon,
            history_buffer=history_buffer,
        )
        _synchronize_device(device)
        step_times.append(time.perf_counter() - t0)
        return action

    def reset_fn() -> None:
        history_buffer.reset()
        policy.reset()

    config_successes = config_total = config_collisions = config_timeouts = 0
    for seed_start in config_seed_starts:
        metrics = evaluate_policy_rollouts(
            simulator, len(initial_states), num_steps, seed_start, action_fn,
            reset_fn=reset_fn, initial_states=initial_states, goal_states=goal_states,
        )
        config_successes += metrics.config_successes
        config_total += metrics.config_num_episodes
        config_collisions += metrics.collision_failures
        config_timeouts += metrics.timeout_failures

    random_successes = random_total = random_collisions = random_timeouts = 0
    for seed_start in random_seed_starts:
        metrics = evaluate_policy_rollouts(
            simulator, random_episodes_per_rep, num_steps, seed_start, action_fn,
            reset_fn=reset_fn, initial_states=None, goal_states=None,
        )
        random_successes += metrics.random_successes
        random_total += metrics.random_num_episodes
        random_collisions += metrics.collision_failures
        random_timeouts += metrics.timeout_failures

    step_times_arr = np.asarray(step_times)
    mean_t = float(step_times_arr.mean())
    median_t = float(np.median(step_times_arr))
    p95_t = float(np.percentile(step_times_arr, 95))
    max_t = float(step_times_arr.max())
    per_robot_mean = mean_t / num_robots

    print(f"\n=== {name} ({policy_type}) ===")
    print(f"config:  {config_successes}/{config_total} = {100*config_successes/config_total:.1f}%  "
          f"(collision={config_collisions} timeout={config_timeouts})")
    print(f"random:  {random_successes}/{random_total} = {100*random_successes/random_total:.1f}%  "
          f"(collision={random_collisions} timeout={random_timeouts})")
    overall_s = config_successes + random_successes
    overall_t = config_total + random_total
    print(f"overall: {overall_s}/{overall_t} = {100*overall_s/overall_t:.1f}%")
    print(
        f"inference time (joint, both robots): mean={mean_t*1000:.2f}ms median={median_t*1000:.2f}ms "
        f"p95={p95_t*1000:.2f}ms max={max_t*1000:.2f}ms  (n={len(step_times)} ticks)"
    )
    print(f"inference time (per-robot, amortized): mean={per_robot_mean*1000:.2f}ms")
    print(
        f"real-time (dt={dt*1000:.0f}ms)? joint mean: {'YES' if mean_t <= dt else 'NO'} "
        f"({mean_t/dt:.2f}x dt) | joint p95: {'YES' if p95_t <= dt else 'NO'} ({p95_t/dt:.2f}x dt) | "
        f"per-robot mean: {'YES' if per_robot_mean <= dt else 'NO'} ({per_robot_mean/dt:.2f}x dt)"
    )
    return {
        "overall_rate": overall_s / overall_t,
        "mean_ms": mean_t * 1000,
        "p95_ms": p95_t * 1000,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the checkpoints in MODES (edit that dict to change what's "
            "compared) on equal-sized config + random evaluation splits, with "
            "per-control-tick inference-time benchmarking against --dt."
        )
    )
    parser.add_argument(
        "--system",
        type=str.lower,
        default="multi_robot",
        choices=DynamicsFactory.names(),
        help="name of system class shared by every checkpoint in MODES",
    )
    parser.add_argument(
        "--expert-config",
        type=str,
        default="test/config/2_multi_unicycle2_casadi_config.yaml",
        help="path to the expert/system yaml (robot count, dynamics, collision radii, ...)",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default="learning/config/2_multi_unicycle2_casadi_flow_config.yaml",
        help=(
            "policy-training yaml to read eval_tolerance_overrides/initial_states/"
            "goal_states from, i.e. which scenarios to evaluate on -- not which "
            "checkpoints to compare (see MODES). Any policy-family config for the "
            "same experiment works interchangeably here, since these fields are "
            "shared across that experiment's flow/safeflow/mlp configs."
        ),
    )
    parser.add_argument(
        "--workspace-bounds",
        type=float,
        nargs=2,
        default=(-3.0, 3.0),
        metavar=("LOW", "HIGH"),
        help="workspace bounds override applied to the expert config",
    )
    parser.add_argument("--num-steps", type=int, default=250, help="maximum steps per episode")
    parser.add_argument(
        "--config-seed-starts",
        type=str,
        default="10000",
        help="comma-separated seed starts; each contributes one pass over --eval-config's full scenario list",
    )
    parser.add_argument(
        "--random-seed-starts",
        type=str,
        default="90000",
        help="comma-separated seed starts; each contributes --random-episodes-per-rep randomly sampled episodes",
    )
    parser.add_argument("--random-episodes-per-rep", type=int, default=180, help="random episodes per seed start")
    parser.add_argument("--dt", type=float, default=0.05, help="control-tick time budget (s) for the real-time verdict")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="override automatic device selection")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_inference_device()
    print(f"device: {device}")

    validated_config, initial_states, goal_states = build_eval_config(
        args.expert_config, args.eval_config, args.system, tuple(args.workspace_bounds)
    )
    num_robots = int(DynamicsFactory.create(system_name=args.system, config=validated_config).num_robots)
    config_seed_starts = _parse_int_list(args.config_seed_starts)
    random_seed_starts = _parse_int_list(args.random_seed_starts)
    n_config = len(initial_states) * len(config_seed_starts)
    n_random = len(random_seed_starts) * args.random_episodes_per_rep
    print(f"sample size per mode: {n_config} config + {n_random} random = {n_config + n_random} episodes")
    print(f"num_robots={num_robots}, dt={args.dt}s")

    results = {}
    for name, (policy_type, checkpoint_path) in MODES.items():
        results[name] = evaluate_mode(
            name, policy_type, checkpoint_path, validated_config, initial_states, goal_states, device, num_robots,
            num_steps=args.num_steps,
            config_seed_starts=config_seed_starts,
            random_seed_starts=random_seed_starts,
            random_episodes_per_rep=args.random_episodes_per_rep,
            dt=args.dt,
        )

    print("\n=== summary ===")
    for name, r in results.items():
        print(f"{name}: overall={r['overall_rate']:.1%}  mean_inference={r['mean_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms")


if __name__ == "__main__":
    main()
