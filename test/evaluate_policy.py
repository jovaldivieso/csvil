import os
import sys
import torch
import ast
import argparse
import numpy as np
from typing import Any, Mapping

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config, validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from planning.casadi_planner import PlannerSolveError
from learning.models.mlp import MLPPolicy
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol

# Import both policy types
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy

from utils import plot_xy_trajectories, save_xy_rollout_video


def is_observation_feature(feature_name: str) -> bool:
    return feature_name.startswith("observation.") or ".observation." in feature_name


def parse_seed_argument(raw_seeds: str | None) -> list[int] | list[list[int]]:
    if raw_seeds is None:
        return [42]

    text = raw_seeds.strip()
    if text == "":
        return [42]

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parts = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
        if len(parts) > 1:
            return [int(part) for part in parts]
        return [int(text)]

    if isinstance(parsed, int):
        return [parsed]
    if isinstance(parsed, list):
        if len(parsed) == 0:
            return [42]
        if all(isinstance(x, int) for x in parsed):
            return [int(x) for x in parsed]
        if all(isinstance(x, list) for x in parsed):
            nested: list[list[int]] = []
            for robot_idx, robot_seeds in enumerate(parsed):
                if not all(isinstance(seed, int) for seed in robot_seeds):
                    raise ValueError(
                        f"All nested seed entries must be integers. Invalid entry in robot {robot_idx}."
                    )
                nested.append([int(seed) for seed in robot_seeds])
            return nested

    raise ValueError(
        "Unable to parse seeds. Use an int (e.g. '42'), a list (e.g. '[42, 7]'), "
        "or nested per-robot lists (e.g. '[[10, 4], [21, 0]]')."
    )


def normalize_seed_specs(
    simulator: DynamicsProtocol,
    seeds: list[int] | list[list[int]] | None,
) -> list[int | list[int]]:
    if seeds is None:
        return [42]

    if len(seeds) == 0:
        return [42]

    if isinstance(seeds[0], list):
        seed_lists = seeds  # type: ignore[assignment]
        expected_len = len(seed_lists[0])
        for robot_idx, robot_seeds in enumerate(seed_lists[1:], start=1):
            if len(robot_seeds) != expected_len:
                raise ValueError(
                    "Seed list length mismatch across robots. "
                    f"Robot 0 has {expected_len} seeds, but Robot {robot_idx} has {len(robot_seeds)} seeds. "
                    "The number of seeds per robot must match to pair initial conditions into joint rollout scenarios."
                )

        return [list(seed_tuple) for seed_tuple in zip(*seed_lists)]

    return [int(seed) for seed in seeds]  # type: ignore[arg-type]


def sample_initial_state(
    simulator: DynamicsProtocol,
    seed_spec: int | list[int],
) -> np.ndarray:
    if isinstance(seed_spec, int):
        rng = np.random.default_rng(seed_spec)
        return simulator.random_initial_state(rng)

    sub_states = []
    sub_simulators = simulator.simulators
    if len(seed_spec) != len(sub_simulators):
        raise ValueError(
            "Per-robot seed specification length must match robot count. "
            f"Got {len(seed_spec)} seeds for {len(sub_simulators)} robots."
        )

    for robot_seed, sub_sim in zip(seed_spec, sub_simulators):
        rng = np.random.default_rng(int(robot_seed))
        sub_states.append(sub_sim.random_initial_state(rng))

    return np.concatenate(sub_states)


def get_inference_device():
    """
    returns the best available device for policy inference
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_policy_input(
    simulator: DynamicsProtocol,
    observation: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Dynamically slices the flat observation array based on the
    feature shapes defined by the simulator.
    """
    policy_input = {}
    features = simulator.get_dataset_features()

    # Reuse the simulator's dataset packing logic to avoid layout drift between
    # runtime observations and the tensors seen during training.
    dummy_action = np.zeros(int(simulator.nu), dtype=np.float32)
    packed_frame = simulator.format_dataset_frame(observation, dummy_action)

    for feature_name in features.keys():
        if is_observation_feature(feature_name):
            if feature_name not in packed_frame:
                raise KeyError(
                    f"Missing observation feature '{feature_name}' in packed frame. "
                    "Check simulator.format_dataset_frame() and dataset schema alignment."
                )
            policy_input[feature_name] = torch.as_tensor(
                packed_frame[feature_name],
                dtype=torch.float32,
                device=device,
            ).view(1, -1)

    return policy_input


def apply_execution_noise(
    simulator: DynamicsProtocol,
    action: np.ndarray,
    action_noise_std: float,
) -> np.ndarray:
    if action_noise_std <= 0.0:
        return action

    noise = np.random.normal(
        loc=0.0,
        scale=action_noise_std,
        size=action.shape,
    ).astype(action.dtype, copy=False)
    return np.clip(
        action + noise,
        -simulator.max_action,
        simulator.max_action,
    )


def rollout_planner(
    simulator: DynamicsProtocol,
    planner: PlannerProtocol,
    initial_state: np.ndarray,
    num_steps: int,
    action_noise_std: float = 0.0,
) -> np.ndarray:
    """
    Rolls out the expert planner from a given initial state.
    """
    state = simulator.reset(initial_state)
    planner.reset()
    trajectory = [state.copy()]

    for _ in range(num_steps):
        if simulator.is_done(state):
            break
        
        obs = simulator.observe(state)
        
        try:
            action = planner(obs)
        except PlannerSolveError as exc:
            print(f"Expert planner failed to solve during evaluation: {exc}")
            break

        executed_action = apply_execution_noise(
            simulator=simulator,
            action=action,
            action_noise_std=action_noise_std,
        )
        state = simulator.step(state, executed_action)
        trajectory.append(state.copy())

    return np.asarray(trajectory)


def rollout_policy(
    simulator: DynamicsProtocol,
    policy: DiffusionPolicy | ACTPolicy | MLPPolicy,
    device: torch.device,
    initial_state: np.ndarray,
    num_steps: int,
    action_noise_std: float = 0.0,
) -> tuple[np.ndarray, bool, int]:
    """
    Rolls out the neural policy from a given initial state.
    
    returns:
        trajectory: array containing visited simulator states
        reached_goal: whether simulator reached goal state
        steps_taken: number of executed simulation steps
    """
    state = simulator.reset(initial_state)
    trajectory = [state.copy()]
    policy.reset()

    for step in range(1, num_steps + 1):
        observation = simulator.observe(state)

        policy_input = create_policy_input(
            simulator=simulator,
            observation=observation,
            device=device,
        )

        with torch.inference_mode():
            action_tensor = policy.select_action(policy_input)

        action = action_tensor.squeeze(0).cpu().numpy()
        executed_action = apply_execution_noise(
            simulator=simulator,
            action=action,
            action_noise_std=action_noise_std,
        )

        state = simulator.step(state, executed_action)
        trajectory.append(state.copy())

        if simulator.is_done(state):
            return np.asarray(trajectory), True, step

    return np.asarray(trajectory), False, num_steps


def run_evaluation(
    system: str,
    policy_type: str,
    config: Mapping[str, Any],
    model_dir: str,
    num_steps: int = 150,
    seeds: list[int] | list[list[int]] | None = None,
    action_noise_std: float = 0.0,
    output_path: str | None = None,
):
    validated_config = validate_system_config(system_name=system, raw_config=config)

    simulator = DynamicsFactory.create(system_name=system, config=validated_config)
    seed_specs = normalize_seed_specs(simulator=simulator, seeds=seeds)

    if not os.path.exists(model_dir):
        print(f"assuming '{model_dir}' is a Hugging Face Hub ID")

    device = get_inference_device()
    print(f"running inference on {device}")

    # Dynamically load the requested policy
    if policy_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(model_dir)
        policy_display_name = "Diffusion"
    elif policy_type == "act":
        policy = ACTPolicy.from_pretrained(model_dir)
        policy_display_name = "ACT"
    elif policy_type == "mlp":
        state_dim = sum(
            int(feature_info["shape"][0])
            for feature_name, feature_info in simulator.get_dataset_features().items()
            if is_observation_feature(feature_name)
        )
        action_dim = int(simulator.nu)

        policy = MLPPolicy(state_dim=state_dim, action_dim=action_dim)

        checkpoint = torch.load(model_dir, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["model_state_dict"])
        else:
            policy.load_state_dict(checkpoint)

        policy_display_name = "MLP"
    else:
        raise ValueError("'policy_type' must be one of {'diffusion', 'act', 'mlp'}.")

    policy.eval()
    policy.to(device)

    print(f"evaluating {len(seed_specs)} seeded trajectories")

    # Instantiate expert planner once and reset it for each rollout.
    expert_planner = PlannerFactory.create(planner_name="casadi", simulator=simulator, config=validated_config)
    goal_state = simulator.goal_state

    expert_trajectories: list[np.ndarray] = []
    policy_trajectories: list[np.ndarray] = []
    per_seed_metrics: list[dict[str, Any]] = []

    for seed_spec in seed_specs:
        torch_seed = int(seed_spec) if isinstance(seed_spec, int) else int(seed_spec[0])
        torch.manual_seed(torch_seed)

        initial_state = sample_initial_state(simulator=simulator, seed_spec=seed_spec)

        expert_trajectory = rollout_planner(
            simulator=simulator,
            planner=expert_planner,
            initial_state=initial_state,
            num_steps=num_steps,
            action_noise_std=action_noise_std,
        )

        policy_trajectory, reached_goal, steps_taken = rollout_policy(
            simulator=simulator,
            policy=policy,
            device=device,
            initial_state=initial_state,
            num_steps=num_steps,
            action_noise_std=action_noise_std,
        )

        policy_final_state = policy_trajectory[-1]
        expert_final_state = expert_trajectory[-1]
        policy_goal_error = float(np.linalg.norm(policy_final_state - goal_state))
        expert_goal_error = float(np.linalg.norm(expert_final_state - goal_state))

        expert_trajectories.append(expert_trajectory)
        policy_trajectories.append(policy_trajectory)
        per_seed_metrics.append(
            {
                "seed": seed_spec,
                "initial_state": initial_state,
                "policy_reached_goal": reached_goal,
                "policy_steps": max(len(policy_trajectory) - 1, 0),
                "expert_steps": max(len(expert_trajectory) - 1, 0),
                "policy_goal_error_l2": policy_goal_error,
                "expert_goal_error_l2": expert_goal_error,
            }
        )

    total_runs = len(per_seed_metrics)
    total_successes = sum(
        1 for metric in per_seed_metrics if metric["policy_reached_goal"]
    )
    success_rate = (total_successes / total_runs) if total_runs > 0 else 0.0
    mean_policy_error = float(
        np.mean([metric["policy_goal_error_l2"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0
    mean_expert_error = float(
        np.mean([metric["expert_goal_error_l2"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0
    mean_policy_steps = float(
        np.mean([metric["policy_steps"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0
    mean_expert_steps = float(
        np.mean([metric["expert_steps"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0

    print("\n--- Evaluation Summary ---")
    print(f"system: {system}")
    print(f"policy_type: {policy_type}")
    print(f"seeds: {seed_specs}")
    print(f"device: {device}")
    print(f"action_noise_std: {action_noise_std:.6f}")
    print(f"goal_state: {np.array2string(goal_state, precision=4)}")
    print(f"num_trajectories: {total_runs}")
    print(f"policy_successes: {total_successes}/{total_runs}")
    print(f"success_rate: {success_rate:.4f}")
    print(f"mean_policy_steps: {mean_policy_steps:.3f}")
    print(f"mean_expert_steps: {mean_expert_steps:.3f}")
    print(f"mean_policy_goal_error_l2: {mean_policy_error:.6f}")
    print(f"mean_expert_goal_error_l2: {mean_expert_error:.6f}")

    # Dynamically set output names
    output_path = output_path or os.path.join(
        os.path.dirname(__file__),
        f"{system}_{policy_type}_policy_path.pdf",
    )

    system_title = system.replace("_", " ").title()

    all_trajectories = expert_trajectories + policy_trajectories
    num_expert = len(expert_trajectories)
    num_policy = len(policy_trajectories)
    path_labels = [None] * (num_expert + num_policy)
    if num_expert > 0:
        path_labels[0] = "Expert"
    if num_policy > 0:
        path_labels[num_expert] = f"{policy_display_name} Policy"
    trajectory_colors = ["tab:blue"] * num_expert + ["tab:orange"] * num_policy
    trajectory_line_styles = ["--"] * num_expert + ["-"] * num_policy

    show_heading = simulator.has_heading

    plot_xy_trajectories(
        simulator=simulator,
        trajectories=all_trajectories,
        path_to_output=output_path,
        title=f"{system_title} {policy_display_name} vs Expert",
        path_labels=path_labels,
        show_heading=show_heading,
        marker="o",
        trajectory_colors=trajectory_colors,
        trajectory_line_styles=trajectory_line_styles,
    )
    print(f"Plot saved to {output_path}")

    video_path = save_xy_rollout_video(
        simulator=simulator,
        trajectories=all_trajectories,
        path_to_output=output_path,
        title=f"{system_title} {policy_display_name} rollout vs Expert",
        show_heading=show_heading,
        fps=12,
        path_labels=path_labels,
        trajectory_colors=trajectory_colors,
        trajectory_line_styles=trajectory_line_styles,
    )
    if video_path is not None:
        print(f"Video saved to {video_path}")

    metrics = {
        "system": system,
        "policy_type": policy_type,
        "seeds": seed_specs,
        "device": str(device),
        "action_noise_std": action_noise_std,
        "goal_state": goal_state,
        "num_trajectories": total_runs,
        "policy_successes": total_successes,
        "success_rate": success_rate,
        "mean_policy_steps": mean_policy_steps,
        "mean_expert_steps": mean_expert_steps,
        "mean_policy_goal_error_l2": mean_policy_error,
        "mean_expert_goal_error_l2": mean_expert_error,
        "per_seed": per_seed_metrics,
        "plot_path": output_path,
        "video_path": video_path,
    }

    return metrics


def main():
    """
    evaluates a trained policy for a selected dynamics system
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        type=str.lower,
        choices=DynamicsFactory.names(),
        required=True,
        help="name of system class, e.g. single_integrator, unicycle2, ...",
    )
    parser.add_argument(
        "--policy-type",
        type=str.lower,
        choices=["diffusion", "act", "mlp"],
        required=True,
        help="the type of policy architecture to evaluate",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to yaml config file for experiment",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="path to a local checkpoint or Hugging Face Hub model ID",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=150,
        help="maximum number of simulation steps",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help=(
            "seed specification: int ('42'), list ('[42, 7]'), or nested per-robot lists "
            "('[[10, 4, 2], [21, 0, 9]]')."
        ),
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.0,
        help=(
            "std-dev of Gaussian action noise applied during expert and policy rollout execution; "
            "default 0.0 keeps evaluation deterministic and clean"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="path to generated PDF plot",
    )

    args = parser.parse_args()
    config = load_and_validate_system_config(system_name=args.system, config_path=args.config)

    run_evaluation(
        system=args.system,
        policy_type=args.policy_type,
        config=config,
        model_dir=args.model_dir,
        num_steps=args.num_steps,
        seeds=parse_seed_argument(args.seeds),
        action_noise_std=args.action_noise_std,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()