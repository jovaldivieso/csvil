
# db-LaCAM’s mapping from identifiers to motion-primitives is in src/run_dblacam.cpp

import os
import yaml
import subprocess
import tempfile
import warnings

import numpy as np


from .planner import Planner

class DbLacamPlanner(Planner):
    """
    python wrapper around db-lacam executable

    modes:
        - "open_loop": plans once at beginning of an episode and executes returned actions sequentially
        - "replan": runs db-lacam again for every observation and executes only first returned action

    supported CSVIL systems:
        - SingleIntegrator
        - Unicycle1       
    """

    def __init__(self, simulator, config, algorithm_config):
        self.sim = simulator
        self.config = config.get("db_lacam", config)
        self.algorithm_config = algorithm_config
        self.robot_type = self.sim.db_lacam_robot_type

        self.mode = self.config.get("mode", "open_loop")
        
        # path to db-lacam executable and working directory:
        self.executable = self.config.get("executable", "/opt/db-lacam/buildRelease/run_dblacam")
        if not os.path.isfile(self.executable):
            raise FileNotFoundError(f"db-LaCAM executable not found: {self.executable}")
        self.cwd = self.config.get("cwd", os.path.dirname(self.executable))
        if not os.path.isdir(self.cwd):
            raise FileNotFoundError(f"working directory not found: {self.cwd}")

        # run_dblacam requires time limit: 
        self.time_limit_ms = int(self.config.get("time_limit_ms", 60_000))

        # raises planning error:
        # TODO: probably deal with planning error to not return 0 action 
        self.raise_planning_error = bool(self.config.get("raise_planning_error", True))

        # environment configuration, no obstacles:
        environment = self.config.get("environment", {})
        self.environment_min = environment.get("min", [-6.0, -6.0])
        self.environment_max = environment.get("max", [6.0, 6.0])
        self.obstacles = []

        self.cached_plan = None
        self.step_idx = 0
        
        # dynobench model and db-lacam motion primitives were generated with dt = 0.1:
        if self.sim.dt != 0.1:
            raise ValueError(f"parameter mismatch: simulator dt is {self.sim.dt}, but {self.robot_type} uses dt = 0.1")



    def reset(self):
        """
        clears cached trajectory for new episode
        """
        self.cached_plan = None
        self.step_idx = 0

    def _compute_plan(self, obs):
        
        # converts observation to state:
        initial_state = self.sim.invert_obs(obs)
        
        # defines db-lacam problem:
        problem = {
            "environment": {
                "min": [float(value) for value in self.environment_min],
                "max": [float(value) for value in self.environment_max],
                "obstacles": self.obstacles, # []
            },
            "robots": [
                {
                    "type": self.robot_type,
                    "start": np.asarray(initial_state, dtype=float).tolist(),
                    "goal": np.asarray(self.sim.goal_state, dtype=float).tolist(),
                }
            ],
        }

        with tempfile.TemporaryDirectory(prefix="dblacam_") as temp_dir:
            
            # db-lacam communicates through yaml-files:
            algorithm_yaml_path = os.path.join(temp_dir, "algorithm.yaml")
            problem_yaml_path = os.path.join(temp_dir,"problem.yaml")
            result_yaml_path = os.path.join(temp_dir, "result.yaml")

            with open(algorithm_yaml_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(self.algorithm_config, file, sort_keys=False)
                
            with open(problem_yaml_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(problem, file, sort_keys=False)

            command = [
                str(self.executable),
                "-i", str(problem_yaml_path),
                "-o", str(result_yaml_path),
                "--stats", str(os.path.join(temp_dir,"stats.yaml")),
                "--cfg", str(algorithm_yaml_path),
                "-t", str(self.time_limit_ms),
            ]

            process = subprocess.run(command, cwd=self.cwd, capture_output=True)

            # useful error check cases were generated with ChatGPT:
            if process.returncode != 0:
                raise RuntimeError(f"db-lacam failed:\n{process.stdout=}\n{process.stderr=}")
            if not os.path.isfile(result_yaml_path):
                raise RuntimeError(f"db-lacam did not create an output yaml:\n{process.stdout=}\n{process.stderr=}")

            with open(result_yaml_path, "r", encoding="utf-8") as file:
                result_data = yaml.safe_load(file)

        trajectories = result_data.get("result")
        # may happen if db-lacam times out or finds no solution:
        if not trajectories:
            raise RuntimeError("db-lacam output contains no trajectories")

        trajectory = trajectories[0]
        actions = np.asarray(trajectory.get("actions", []), dtype=float)
        # check added by ChatGPT:
        if actions.size == 0:
            actions = np.empty((0, self.sim.nu), dtype=float)

        self.cached_plan = actions
        # resets to first action: 
        self.step_idx = 0

    def __call__(self, obs):
        
        try:
            # replans or plans once for the first time if 'open_loop' mode is used:
            if self.mode == "replan" or self.cached_plan is None:
                self._compute_plan(obs)

            if self.step_idx >= len(self.cached_plan):
                return np.zeros(self.sim.nu, dtype=float)

            action = self.cached_plan[self.step_idx].copy()
            self.step_idx += 1
            return action

        except (RuntimeError) as error:
            if self.raise_planning_error:
                raise

            warnings.warn(f"db-lacam planning failed: {error}")
            return np.zeros(self.sim.nu, dtype=float)