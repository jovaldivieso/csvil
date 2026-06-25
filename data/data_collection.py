# data/data_collection.py
import numpy as np
from pathlib import Path


class DataCollector:
    def __init__(self, simulator, output_dir="data/trajectories"):
        self.sim = simulator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_trajectories(self, motion_planner, num_trajectories,
                             num_steps=100):
        """Generate dataset by rolling out motion planner"""
        dataset = {"observations": [], "actions": []}

        for traj_id in range(num_trajectories):
            # Random initial state
            initial_state = np.random.randn(4) * 0.5  # Gaussian noise

            # Rollout
            traj = self.sim.rollout(initial_state, motion_planner, num_steps)

            # Collect obs-action pairs
            dataset["observations"].extend(traj["observations"])
            dataset["actions"].extend(traj["actions"])

            if (traj_id + 1) % 10 == 0:
                print(
                    f"Collected {traj_id + 1}/{num_trajectories} \
                    trajectories"
                )

        # Convert to numpy and save
        dataset["observations"] = np.array(dataset["observations"])
        dataset["actions"] = np.array(dataset["actions"])

        np.save(self.output_dir / "observations.npy", dataset["observations"])
        np.save(self.output_dir / "actions.npy", dataset["actions"])

        return dataset
