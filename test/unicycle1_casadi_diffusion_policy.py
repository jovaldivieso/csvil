import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.unicycle1 import Unicycle1
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Trained Diffusion Policy")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to local checkpoint or Hugging Face Hub ID")
    args = parser.parse_args()

    # max_v instead of max_accel
    config = {"dt": 0.05, "max_v": 2.0, "goal": args.goal}
    sim = Unicycle1(config)

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

    state = sim.reset_random()
    config["goal"] = sim.goal.copy()

    policy.reset()

    # Store (x, y, theta) for quiver plotting
    trajectory = [state[:3].copy()]

    for step in range(150):
        obs = sim.observe(state)

        # Unicycle1 dataset hack: pass the 3D vector to both keys
        obs_dict = {
            "observation.environment_state":
            torch.from_numpy(obs[0:3]).float().unsqueeze(0).to(device),
            "observation.state":
            torch.from_numpy(obs[0:3]).float().unsqueeze(0).to(device)
        }

        with torch.no_grad():
            action_tensor = policy.select_action(obs_dict)
            action = action_tensor.squeeze(0).cpu().numpy()

        state = sim.step(state, action)
        trajectory.append(state[:3].copy())

        if sim.is_done(state):
            print(f"Goal Reached in {step} steps.")
            break

    trajectory = np.array(trajectory)
    plt.figure(figsize=(8, 8))

    # Plot the driven path
    line, = plt.plot(trajectory[:, 0], trajectory[:, 1], '-o', color='blue',
                     label='Diffusion Policy Path', markersize=4, alpha=0.6)

    # Show start position and initial heading
    plt.scatter(trajectory[0, 0], trajectory[0, 1], c='green', marker='o',
                s=150, label='Start Position', zorder=4)
    plt.quiver(trajectory[0, 0], trajectory[0, 1],
               np.cos(trajectory[0, 2]), np.sin(trajectory[0, 2]),
               color='green', scale=8, width=0.005, zorder=5)

    # Show goal position and target heading
    gx, gy, gtheta = config["goal"]
    plt.scatter(gx, gy, c='red', marker='*', s=300, label='Goal Position',
                zorder=4)
    plt.quiver(gx, gy, np.cos(gtheta), np.sin(gtheta),
               color='red', scale=8, width=0.005, zorder=5)

    # Titles and file names
    plt.title("Unicycle 1: Trained Diffusion Policy Evaluation")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axis('equal')

    file_name = "unicycle1_casadi_diffusion_policy_plot.pdf"
    output_path = os.path.join(os.path.dirname(__file__), file_name)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved successfully to: {output_path}")


if __name__ == "__main__":
    main()
