import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from planning.casadi_planner import PlannerSolveError
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol
from systems.seed_utils import (
    action_noise_rng_for_rollout,
    default_action_noise_seed_for_config,
)


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
        self.action_noise_seed = default_action_noise_seed_for_config(self.sim.config)

    def collect_trajectories(
        self,
        motion_planner: PlannerProtocol,
        num_trajectories: int,
        num_steps: int = 100,
        action_noise_std: float = 0.0,
        initial_states: list[np.ndarray] | None = None,
    ) -> LeRobotDataset:
        print(f"Collecting {num_trajectories} trajectories...")
        provided_initial_states = initial_states or []

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

            # Use provided initial states first; then fall back to simulator RNG.
            if attempted_trajectories <= len(provided_initial_states):
                state = self.sim.reset(provided_initial_states[attempted_trajectories - 1])
                initial_state_source = "provided"
            else:
                state = self.sim.reset_random()
                initial_state_source = "rng_fallback"
            episode_initial_state = state.copy()
            episode_action_noise_rng = action_noise_rng_for_rollout(
                self.action_noise_seed,
                rollout_index=attempted_trajectories,
            )
            planner_failed = False
            episode_frames = []

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
                        "Skipping trajectory due to planner failure "
                        f"(attempt={attempted_trajectories}, source={initial_state_source}, "
                        f"action_noise_std={action_noise_std:.6f})."
                    )
                    print(
                        "Planner failure context: "
                        f"initial_state={np.array2string(np.asarray(episode_initial_state), precision=6)}, "
                        f"current_state={np.array2string(np.asarray(state), precision=6)}, "
                        f"goal_state={np.array2string(np.asarray(self.sim.goal_state), precision=6)}"
                    )
                    print(f"Underlying solver error: {exc}")
                    planner_failed = True
                    break

                if action_noise_std > 0.0:
                    noise = episode_action_noise_rng.normal(
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

                state = self.sim.step(state, executed_action)

                if self.sim.is_collision(state):
                    print(
                        "Discarding trajectory due to expert collision "
                        f"(attempt={attempted_trajectories}, source={initial_state_source})."
                    )
                    planner_failed = True
                    break

                episode_frames.append(frame_data)

                if self.sim.should_terminate_rollout(state):
                    break

            if planner_failed:
                continue

            for frame_data in episode_frames:
                dataset.add_frame(frame_data)
            dataset.save_episode()
            successful_trajectories += 1

            if successful_trajectories % 10 == 0:
                print(
                    "Collected "
                    f"{successful_trajectories}/{num_trajectories} trajectories"
                )

        print(f"LeRobot Dataset saved successfully to {self.local_dir}")
        return dataset