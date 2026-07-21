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
from core.factory import DynamicsFactory, HEADING_SYSTEMS

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


def create_policy_input(simulator, observation, device):
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


def rollout_policy(simulator, policy, device, num_steps):
    """
    runs one rollout of a trained policy

    returns:
        trajectory: array containing visited simulator states
        reached_goal: whether simulator reached goal state
        steps_taken: number of executed simulation steps
    """
    state = simulator.reset_random()
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
    seed: int = 42,
    output_path: str | None = None,
):
    validated_config = validate_system_config(system_name=system, raw_config=config)

    np.random.seed(seed)
    torch.manual_seed(seed)

    simulator = DynamicsFactory.create(system_name=system, config=validated_config)

    if not os.path.exists(model_dir):
        print(f"assuming '{model_dir}' is a Hugging Face Hub ID")

    device = get_inference_device()
    print(f"running inference on {device}")

    if policy_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(model_dir)
        policy_display_name = "Diffusion"
    elif policy_type == "act":
        policy = ACTPolicy.from_pretrained(model_dir)
        policy_display_name = "ACT"
    else:
        raise ValueError("'policy_type' must be either 'diffusion' or 'act'.")

    policy.eval()
    policy.to(device)

    trajectory, reached_goal, steps_taken = rollout_policy(
        simulator=simulator,
        policy=policy,
        device=device,
        num_steps=num_steps,
    )

    if reached_goal:
        print(f"goal reached in {steps_taken} steps")
    else:
        print(f"goal not reached after {steps_taken} steps")

    output_path = output_path or os.path.join(
        os.path.dirname(__file__),
        f"{system}_{policy_type}_policy_path.pdf",
    )

    system_title = system.replace("_", " ").title()

    plot_xy_trajectories(
        simulator=simulator,
        trajectories=[trajectory],
        path_to_output=output_path,
        title=f"{system_title} {policy_display_name} Policy Evaluation",
        path_label=f"{policy_display_name} policy path",
        show_heading=system in HEADING_SYSTEMS,
        marker="o",
    )
    return output_path


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
        choices=["diffusion", "act"],
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
        "--seed",
        type=int,
        default=42,
        help="random seed for initial state and policy sampling",
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
        seed=args.seed,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
