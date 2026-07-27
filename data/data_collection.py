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

    def collect_trajectories(self, motion_planner, num_trajectories, num_steps=100):
        print(f"Collecting {num_trajectories} trajectories...")

        if self.local_dir.exists():
            print(f"Cleaning up existing dataset at {self.local_dir}...")
            shutil.rmtree(self.local_dir)

        # creates dataset using simulator specific features:
        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            root=self.local_dir,
            features=self.sim.get_dataset_features(),
        )

        multi_robot = hasattr(self.sim, "robots")

        traj_id = 0
        while traj_id < num_trajectories:
            state = self.sim.reset_random()
            done_counter = 0

            # resets cached planner state for new episode:
            if hasattr(motion_planner, "reset"):
                motion_planner.reset()

            if multi_robot:
                # stores one separate episode for every robot:
                episode_frames = [[] for _ in self.sim.robots]
            else:
                episode_frames = []

            try:
                for _ in range(num_steps):
                    obs = self.sim.observe(state)
                    
                    # The collector doesn't know HOW the planner gets the action
                    action = motion_planner(obs)

                    if multi_robot:
                        local_observations = self.sim.create_local_observations(state)

                        # Ask the simulator to format the current frame
                        for robot_ind, (local_observation, robot_action) in enumerate(
                            zip(local_observations, action)
                        ):
                            frame_data = self.sim.format_dataset_frame(
                                robot_ind,
                                local_observation,
                                robot_action,
                            )
                            frame_data["task"] = "reach target"

                            episode_frames[robot_ind].append(frame_data)
                    else:
                        # Ask the simulator to format the current frame
                        frame_data = self.sim.format_dataset_frame(obs, action)
                        frame_data["task"] = "reach target"

                        episode_frames.append(frame_data)

                    state = self.sim.step(state, action)

                    # Break so it learns to hold its position and stop
                    if self.sim.is_done(state):
                        done_counter += 1
                        if done_counter >= 5:
                            break

            except RuntimeError as error:
                # discards failed episode and samples a new one:
                print(f"planning failed, resampling episode: {str(error)[-2000:]}")
                continue

            if multi_robot:
                # saves one episode for every robot:
                for robot_frames in episode_frames:
                    for frame_data in robot_frames:
                        dataset.add_frame(frame_data)

                    dataset.save_episode()
            else:
                for frame_data in episode_frames:
                    dataset.add_frame(frame_data)

                dataset.save_episode()

            traj_id += 1

            if traj_id % 10 == 0:
                print(
                    f"Collected {traj_id}/{num_trajectories} trajectories")

        print(f"LeRobot Dataset saved successfully to {self.local_dir}")
        return dataset