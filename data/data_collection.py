import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from planning.casadi_planner import PlannerSolveError
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol


class DataCollector:
    def __init__(
        self,
        simulator: DynamicsProtocol,
        repo_id="undefined_expert",
        local_dir="data/lerobot_dataset",
    ):
        self.sim = simulator
        self.repo_id = repo_id
        self.local_dir = Path(local_dir)
        self.fps = int(1 / self.sim.dt)

    def collect_trajectories(
        self,
        motion_planner: PlannerProtocol,
        num_trajectories: int,
        num_steps: int = 100,
        action_noise_std: float = 0.0,
    ) -> LeRobotDataset:
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

        successful_trajectories = 0
        attempted_trajectories = 0
        max_attempts = max(num_trajectories * 3, num_trajectories)

        while successful_trajectories < num_trajectories:
            attempted_trajectories += 1
            if attempted_trajectories > max_attempts:
                raise RuntimeError(
                    "Too many failed trajectory attempts. "
                    f"Collected {successful_trajectories}/{num_trajectories} successful trajectories."
                )

            # Ask the simulator to initialize itself in a random, valid way
            state = self.sim.reset_random()
            done_counter = 0
            planner_failed = False

            # Tell the planner a new episode is starting!
            if hasattr(motion_planner, "reset"):
                motion_planner.reset()

            for _ in range(num_steps):
                obs = self.sim.observe(state)

                # The collector doesn't know HOW the planner gets the action
                try:
                    action = motion_planner(obs)
                except PlannerSolveError as exc:
                    print(
                        "Skipping trajectory due to planner failure: "
                        f"{exc}"
                    )
                    planner_failed = True
                    break

                if action_noise_std > 0.0:
                    noise = np.random.normal(
                        loc=0.0,
                        scale=action_noise_std,
                        size=action.shape,
                    ).astype(action.dtype, copy=False)
                    executed_action = np.clip(
                        action + noise,
                        -self.sim.max_action,
                        self.sim.max_action,
                    )
                else:
                    executed_action = action

                # Ask the simulator to format the current frame
                frame_data = self.sim.format_dataset_frame(obs, action)
                frame_data["task"] = "reach target"

                dataset.add_frame(frame_data)
                state = self.sim.step(state, executed_action)

                # Break so it learns to hold its position and stop
                if self.sim.is_done(state):
                    done_counter += 1
                    if done_counter >= 5:
                        break

            if planner_failed:
                continue

            dataset.save_episode()
            successful_trajectories += 1

            if successful_trajectories % 10 == 0:
                print(
                    "Collected "
                    f"{successful_trajectories}/{num_trajectories} trajectories"
                )

        print(f"LeRobot Dataset saved successfully to {self.local_dir}")
        return dataset
