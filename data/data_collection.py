import numpy as np
import torch
import shutil
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class DataCollector:
    def __init__(
        self,
        simulator,
        repo_id="undefined_expert",
        local_dir="data/lerobot_dataset",
    ):
        self.sim = simulator
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.fps = int(1 / self.sim.dt)

    def collect_trajectories(self, motion_planner, num_trajectories,
                             num_steps=100):
        print(f"Collecting {num_trajectories} trajectories...")

        # Wipe old dataset if it exists to allow easy re-running
        if self.local_dir.exists():
            print(f"Cleaning up existing dataset at {self.local_dir}...")
            shutil.rmtree(self.local_dir)

        # Define the LeRobot features schema
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (4,),
                "names": ["goal_rel_x", "goal_rel_y", "vx", "vy"],
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["vx", "vy"],
            },
        }

        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.local_dir,
            features=features,
        )

        # Rollout and Record
        for traj_id in range(num_trajectories):
            # Random initial state: x, y, vx, vy
            initial_state = np.random.randn(4) * 0.5
            state = self.sim.reset(initial_state)

            for _ in range(num_steps):
                obs = self.sim.observe(state)
                action = motion_planner(obs)

                # Append frame to the current active episode
                dataset.add_frame(
                    {
                        "observation.state": torch.from_numpy(obs).float(),
                        "action": torch.from_numpy(action).float(),
                        "task": "reach the target position",
                    }
                )

                # Step simulation
                state = self.sim.step(state, action)

                # Optional: Early termination if goal is reached
                if self.sim.is_done(state):
                    break

            # Lock in the episode (increments episode index automatically)
            dataset.save_episode()

            if (traj_id + 1) % 10 == 0:
                print(f"Collected {traj_id + 1}/{num_trajectories} \
                trajectories")

        print(f"LeRobot Dataset saved successfully to {self.local_dir}")

        return dataset
