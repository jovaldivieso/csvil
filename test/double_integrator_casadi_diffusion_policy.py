import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.double_integrator import DoubleIntegrator
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Trained Diffusion Policy")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to local checkpoint or Hugging Face Hub ID")
    parser.add_argument("--goal", type=float, nargs=2, default=[1.0, 1.0],
                        help="Target goal position [x, y]")
    args = parser.parse_args()

    config = {"dt": 0.05, "max_accel": 2.0, "goal": args.goal}
    sim = DoubleIntegrator(config)

    # If the path doesn't exist on the hard drive, try hugging face hub id
    if not os.path.exists(args.model_dir):
        print(f"Assuming '{args.model_dir}' is a Hugging Face Hub ID.")

    policy = DiffusionPolicy.from_pretrained(args.model_dir)
    policy.eval()  # Set network to evaluation mode

    # Explicitly set up the M1 GPU (MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Running inference on: {device}")
    policy.to(device)  # Push the neural network to the GPU

    np.random.seed(42)  # Fixed seed for reproducible testing

    # Spawn the test rollout using the same uniform polar logic
    radius = np.random.uniform(0.5, 3.0)
    angle = np.random.uniform(0, 2 * np.pi)
    offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    start_pos = np.array(sim.goal) + offset

    initial_state = np.array([start_pos[0], start_pos[1], 0.0, 0.0])
    state = sim.reset(initial_state)

    policy.reset()

    trajectory = [state[:2].copy()]  # Store (x, y) for plotting

    for step in range(100):
        obs = sim.observe(state)

        obs_dict = {
            "observation.environment_state":
            torch.from_numpy(obs[0:2]).float().unsqueeze(0).to(device),
            "observation.state":
            torch.from_numpy(obs[2:4]).float().unsqueeze(0).to(device)
        }

        with torch.no_grad():
            action_tensor = policy.select_action(obs_dict)
            action = action_tensor.squeeze(0).cpu().numpy()

        state = sim.step(state, action)
        trajectory.append(state[:2].copy())

        if sim.is_done(state):
            print(f"Goal Reached in {step} steps.")
            break

    trajectory = np.array(trajectory)
    plt.figure(figsize=(8, 8))

    plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', color='blue',
             label='Diffusion Policy Path', markersize=4)
    plt.scatter(trajectory[0, 0], trajectory[0, 1], c='green', marker='o',
                s=150, label='Start Position')
    plt.scatter(config["goal"][0], config["goal"][1], c='red', marker='X',
                s=150, label='Goal Position')

    plt.title("Double Integrator: Trained Diffusion Policy Evaluation")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axis('equal')

    file_name = "double_integrator_casadi_diffusion_policy_plot.pdf"
    output_path = os.path.join(os.path.dirname(__file__), file_name)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
