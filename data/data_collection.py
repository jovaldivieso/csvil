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

        if self.local_dir.exists():
            print(f"Cleaning up existing dataset at {self.local_dir}...")
            shutil.rmtree(self.local_dir)

        # Schema: Split state into environment (goal-relative) and robot state
        features = {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["goal_rel_x", "goal_rel_y"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["vx", "vy"],
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["ax", "ay"],  # Adjusted to represent acceleration
            },
        }

        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.local_dir,
            features=features,
        )

        for traj_id in range(num_trajectories):
            # Evenly samples a radius between 0.5m and 3.0m, and an angle
            # between 0 and 360 degrees
            radius = np.random.uniform(0.5, 3.0)
            angle = np.random.uniform(0, 2 * np.pi)

            offset = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            start_pos = np.array(self.sim.goal) + offset

            initial_state = np.array([start_pos[0], start_pos[1], 0.0, 0.0])
            state = self.sim.reset(initial_state)

            done_counter = 0
            for _ in range(num_steps):
                obs = self.sim.observe(state)
                action = motion_planner(obs)

                dataset.add_frame({
                    "observation.environment_state":
                    torch.from_numpy(obs[0:2]).float(),
                    "observation.state": torch.from_numpy(obs[2:4]).float(),
                    "action": torch.from_numpy(action).float(),
                    "task": "reach target"
                })

                state = self.sim.step(state, action)

                # Break so it learns to hold its position and stop
                if self.sim.is_done(state):
                    done_counter += 1
                    if done_counter >= 5:
                        break

            dataset.save_episode()

            if (traj_id + 1) % 10 == 0:
                print(f"Collected {traj_id + 1}/{num_trajectories} \
                trajectories")

        print(f"LeRobot Dataset saved successfully to {self.local_dir}")
        return dataset
