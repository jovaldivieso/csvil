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

        # Ask the simulator for its specific data schema
        features = self.sim.get_dataset_features()

        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.local_dir,
            features=features,
        )

        for traj_id in range(num_trajectories):
            # Ask the simulator to initialize itself in a random, valid way
            state = self.sim.reset_random()
            done_counter = 0

            for _ in range(num_steps):
                obs = self.sim.observe(state)
                action = motion_planner(obs)

                # Ask the simulator to format the current frame
                frame_data = self.sim.format_dataset_frame(obs, action)
                frame_data["task"] = "reach target"

                dataset.add_frame(frame_data)
                state = self.sim.step(state, action)

                # Break so it learns to hold its position and stop
                if self.sim.is_done(state):
                    done_counter += 1
                    if done_counter >= 5:
                        break

            dataset.save_episode()

            if (traj_id + 1) % 10 == 0:
                print(
                    f"Collected {traj_id + 1}/{num_trajectories} trajectories")

        print(f"LeRobot Dataset saved successfully to {self.local_dir}")
        return dataset
