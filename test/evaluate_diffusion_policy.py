import os
import sys
import json
import torch
import argparse

import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

# import new systems here, then add them to SIMULATORS (and HEADING_SYSTEMS):
from systems.double_integrator import DoubleIntegrator
from systems.single_integrator import SingleIntegrator
from systems.unicycle1 import Unicycle1
from systems.unicycle2 import Unicycle2
from utils import plot_xy_trajectories

SIMULATORS = {
    "single_integrator": SingleIntegrator,
    "double_integrator": DoubleIntegrator,
    "unicycle1": Unicycle1,
    "unicycle2": Unicycle2,
}

HEADING_SYSTEMS = {
    "unicycle1",
    "unicycle2",
}

def get_inference_device():
    """
    returns the best available device for diffusion-policy inference
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_policy_input(simulator, observation, device):
    """
    converts a simulator observation into dictionary expected by LeRobot

    current single-robot systems use same observation vector for
    ``observation.environment_state`` and ``observation.state``
    """

    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    return {
        feature_name: observation_tensor
        for feature_name in simulator.get_dataset_features()
        if feature_name.startswith("observation.")
    }


def rollout_diffusion_policy(simulator, policy, device, num_steps):
    """
    runs one rollout of a trained diffusion policy

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

def main():
    """
    evaluates a trained diffusion policy for a selected dynamics system
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "system",
        type=str.lower,
        choices=SIMULATORS.keys(),
        help="name of system class, e.g. single_integrator, unicycle2, ...",
    )
    parser.add_argument(
        "path_to_config",
        type=str,
        help="path to json config file for experiment",
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="path to a local checkpoint or Hugging Face Hub model ID",
    )
    parser.add_argument(
        "--num_steps",
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
        "--output_path",
        type=str,
        default=None,
        help="path to generated PDF plot",
    )

    args = parser.parse_args()

    with open(args.path_to_config, "r") as file:
        config = json.load(file)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    simulator = SIMULATORS[args.system](config)

    if not os.path.exists(args.model_dir):
        print(f"assuming '{args.model_dir}' is a Hugging Face Hub ID")

    device = get_inference_device()
    print(f"running inference on {device}")

    policy = DiffusionPolicy.from_pretrained(args.model_dir)
    policy.eval()
    policy.to(device)

    trajectory, reached_goal, steps_taken = rollout_diffusion_policy(
        simulator=simulator,
        policy=policy,
        device=device,
        num_steps=args.num_steps,
    )

    if reached_goal:
        print(f"goal reached in {steps_taken} steps")
    else:
        print(f"goal not reached after {steps_taken} steps")

    output_path = args.output_path or os.path.join(
        os.path.dirname(__file__),
        f"{args.system}_diffusion_policy_path.pdf",
    )

    system_title = args.system.replace("_", " ").title()

    plot_xy_trajectories(
        simulator=simulator,
        trajectories=[trajectory],
        path_to_output=output_path,
        title=f"{system_title} diffusion policy evaluation",
        path_label="diffusion policy path",
        show_heading=args.system in HEADING_SYSTEMS,
        marker="o",
    )

if __name__ == "__main__":
    main()