import os
import sys
import torch
import argparse
import numpy as np
from typing import Any, Mapping

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config, validate_system_config
from core.factory import DynamicsFactory, SE2_SYSTEMS, PlannerFactory
from planning.casadi_planner import PlannerSolveError
from learning.models.mlp import MLPPolicy
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol

# Import both policy types
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy

from utils import plot_xy_trajectories


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

    current_idx = 0
    for feature_name, feature_info in features.items():
        if feature_name.startswith("observation."):
            # Get the dimension of this specific observation feature
            dim = feature_info["shape"][0]

            # Slice the exact chunk of the observation array
            sliced_obs = observation[current_idx: current_idx + dim]

            # Convert to tensor, add batch dimension, and send to device
            policy_input[feature_name] = torch.as_tensor(
                sliced_obs,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            # Move the pointer forward for the next feature
            current_idx += dim

    return policy_input


def rollout_planner(
    simulator: DynamicsProtocol,
    planner: PlannerProtocol,
    initial_state: np.ndarray,
    num_steps: int,
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
            
        state = simulator.step(state, action)
        trajectory.append(state.copy())

    return np.asarray(trajectory)


def rollout_policy(
    simulator: DynamicsProtocol,
    policy: DiffusionPolicy | ACTPolicy | MLPPolicy,
    device: torch.device,
    initial_state: np.ndarray,
    num_steps: int,
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

        state = simulator.step(state, action)
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
    seeds: list[int] | None = None,
    output_path: str | None = None,
):
    validated_config = validate_system_config(system_name=system, raw_config=config)
    seeds = seeds or [42, 123, 13, 11, 40]

    simulator = DynamicsFactory.create(system_name=system, config=validated_config)

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
            if feature_name.startswith("observation.")
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

    print(f"evaluating {len(seeds)} seeded trajectories")

    # Instantiate expert planner once and reset it for each rollout.
    expert_planner = PlannerFactory.create(planner_name="casadi", simulator=simulator, config=validated_config)
    goal_state = simulator.goal_state

    expert_trajectories: list[np.ndarray] = []
    policy_trajectories: list[np.ndarray] = []
    per_seed_metrics: list[dict[str, Any]] = []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        initial_state = simulator.reset_random()

        expert_trajectory = rollout_planner(
            simulator=simulator,
            planner=expert_planner,
            initial_state=initial_state,
            num_steps=num_steps,
        )

        policy_trajectory, reached_goal, steps_taken = rollout_policy(
            simulator=simulator,
            policy=policy,
            device=device,
            initial_state=initial_state,
            num_steps=num_steps,
        )

        policy_final_state = policy_trajectory[-1]
        expert_final_state = expert_trajectory[-1]
        policy_goal_error = float(np.linalg.norm(policy_final_state - goal_state))
        expert_goal_error = float(np.linalg.norm(expert_final_state - goal_state))

        expert_trajectories.append(expert_trajectory)
        policy_trajectories.append(policy_trajectory)
        per_seed_metrics.append(
            {
                "seed": seed,
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
    print(f"seeds: {seeds}")
    print(f"device: {device}")
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

    plot_xy_trajectories(
        simulator=simulator,
        trajectories=all_trajectories,
        path_to_output=output_path,
        title=f"{system_title} {policy_display_name} vs Expert",
        path_labels=path_labels,
        show_heading=system in SE2_SYSTEMS,
        marker="o",
        trajectory_colors=trajectory_colors,
    )
    print(f"Plot saved to {output_path}")

    metrics = {
        "system": system,
        "policy_type": policy_type,
        "seeds": seeds,
        "device": str(device),
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
        type=int,
        nargs="+",
        default=[42, 123, 13, 11, 40],
        help="list of seeds for initial state and policy sampling",
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
        seeds=args.seeds,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()