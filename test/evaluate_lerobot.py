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


def rollout_planner(simulator, planner, initial_state, num_steps):
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


def rollout_policy(simulator, policy, device, initial_state, num_steps):
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
    seed: int = 42,
    output_path: str | None = None,
):
    validated_config = validate_system_config(system_name=system, raw_config=config)

    # Set seeds to ensure reproducibility of the random initial state
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    else:
        raise ValueError("'policy_type' must be either 'diffusion' or 'act'.")

    policy.eval()
    policy.to(device)

    # 1. Sample a single initial state to be shared by both rollouts
    initial_state = simulator.reset_random()

    # 2. Instantiate and rollout the expert planner
    print("Rolling out expert planner...")
    expert_planner = PlannerFactory.create(planner_name="casadi", simulator=simulator, config=validated_config)
    expert_trajectory = rollout_planner(
        simulator=simulator,
        planner=expert_planner,
        initial_state=initial_state,
        num_steps=num_steps,
    )

    # 3. Rollout the neural policy from the EXACT same initial state
    print("Rolling out neural policy...")
    policy_trajectory, reached_goal, steps_taken = rollout_policy(
        simulator=simulator,
        policy=policy,
        device=device,
        initial_state=initial_state,
        num_steps=num_steps,
    )

    if reached_goal:
        print(f"Policy reached goal in {steps_taken} steps")
    else:
        print(f"Policy did not reach goal after {steps_taken} steps")

    # Dynamically set output names
    output_path = output_path or os.path.join(
        os.path.dirname(__file__),
        f"{system}_{policy_type}_policy_path.pdf",
    )

    system_title = system.replace("_", " ").title()

    # 4. Plot both trajectories overlaying each other
    plot_xy_trajectories(
        simulator=simulator,
        trajectories=[expert_trajectory, policy_trajectory],
        path_to_output=output_path,
        title=f"{system_title} {policy_display_name} vs Expert",
        path_labels=["Expert", f"{policy_display_name} Policy"],
        show_heading=system in SE2_SYSTEMS,
        marker="o",
    )
    print(f"Plot saved to {output_path}")
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
