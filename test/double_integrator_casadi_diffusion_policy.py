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
    # Set default to [0.0, 0.0]
    parser.add_argument(
        "--goal", type=float, nargs=2, default=[0.0, 0.0],
        help="Specific goal coordinate. Defaults to [0.0, 0.0].")
    args = parser.parse_args()

    # Hardcode randomize_goal to False and inject the fixed goal
    config = {
        "dt": 0.05,
        "max_accel": 2.0,
        "goal": args.goal,
        "randomize_goal": False
    }

    sim = DoubleIntegrator(config)

    if not os.path.exists(args.model_dir):
        print(f"Assuming '{args.model_dir}' is a Hugging Face Hub ID.")

    policy = DiffusionPolicy.from_pretrained(args.model_dir)
    policy.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Running inference on: {device}")
    policy.to(device)

    np.random.seed(42)

    state = sim.reset_random()

    config["goal"] = sim.goal.copy()

    policy.reset()

    trajectory = [state[:2].copy()]

    for step in range(100):
        obs = sim.observe(state)

        obs_dict = {
            "observation.environment_state":
            torch.from_numpy(obs[0:4]).float().unsqueeze(0).to(device),
            "observation.state":
            torch.from_numpy(obs[0:4]).float().unsqueeze(0).to(device)
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
                s=150, label='Start Position', zorder=4)
    plt.scatter(config["goal"][0], config["goal"][1], c='red', marker='X',
                s=150, label='Goal Position', zorder=5)

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
