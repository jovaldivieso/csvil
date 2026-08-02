import os
import yaml
import subprocess
import tempfile
import warnings

import numpy as np

from .planner import Planner

class DbLacamPlanner(Planner):
    """
    python wrapper around db-lacam executable for single- and multi-robot planning,
    supporting heterogeneous robot teams

    modes:
        - "open_loop": plans once and executes returned actions sequentially
        - "replan": computes a new joint plan after replan_freq steps

    currently supported CSVIL systems:
        - SingleIntegrator
        - Unicycle1       
    """
    def __init__(self, simulator, config):
        self.sim = simulator
        self.config = config.get("db_lacam", config)

        self.robots = list(self.sim.simulators)

        # algorithm_config is a separate YAML file containing db-lacam’s internal planning parameters,
        # such as search resolution, heuristic settings, motion primitive counts, etc.
        algorithm_config_path = self.config.get("algorithm_config")
        if not algorithm_config_path:
            raise ValueError("'db_lacam.algorithm_config' is required.")

        with open(algorithm_config_path, "r", encoding="utf-8") as file:
            self.algorithm_config = yaml.safe_load(file)

        if not isinstance(self.algorithm_config, dict):
            raise ValueError("db_lacam algorithm config must contain a YAML mapping")

        self.mode = self.config.get("mode", "replan")
        self.replan_freq = int(self.config.get("replan_freq", 5))

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
        self.raise_planning_error = bool(self.config.get("raise_planning_error", True))

        # environment configuration, no obstacles:
        environment = self.config.get("environment", {})
        self.environment_min = environment.get("min", [-6.0, -6.0])
        self.environment_max = environment.get("max", [6.0, 6.0])
        self.obstacles = []

        self.cached_plans = None
        self.step_idx = 0

        # dynobench model and db-lacam motion primitives were generated with dt = 0.1:
        for robot in self.robots:
            if robot.dt != 0.1:
                raise ValueError(f"parameter mismatch: simulator dt is {self.sim.dt}, but {robot.db_lacam_robot_type} uses dt = 0.1")
     
    def reset(self):
        self.cached_plans = None
        self.step_idx = 0

    def _create_states_list(self, obs):
        global_state = self.sim.invert_obs(obs)
        return [global_state[s] for s in self.sim.robot_state_slices]
    
    def _define_dblacam_problem(self, obs):
        """
        converts csvil observations into one db-lacam problem dictionary for all robot
        """

        states = self._create_states_list(obs)

        robot_entries = []
        for robot, state in zip(self.robots, states):
            robot_entries.append(
                {
                    "type": robot.db_lacam_robot_type,
                    "start": np.asarray(state, dtype=float).tolist(),
                    "goal": np.asarray(robot.goal_state, dtype=float).tolist(),
                }
            )

        # defines db-lacam problem:
        return {
            "environment": {
                "min": [float(value) for value in self.environment_min],
                "max": [float(value) for value in self.environment_max],
                "obstacles": self.obstacles, # []
            },
            "robots": robot_entries,
        }
     
    def _compute_plan(self, obs):
        
        problem = self._define_dblacam_problem(obs)

        with tempfile.TemporaryDirectory(prefix="dblacam_") as temp_dir:
            
            # db-lacam communicates through yaml files:
            algorithm_yaml_path = os.path.join(temp_dir, "algorithm.yaml")
            problem_yaml_path = os.path.join(temp_dir,"problem.yaml")
            result_yaml_path = os.path.join(temp_dir, "result.yaml")
            
            with open(algorithm_yaml_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(self.algorithm_config, file, sort_keys=False)
                
            with open(problem_yaml_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(problem, file, sort_keys=False)

            command = [
                self.executable,
                "-i", problem_yaml_path,
                "-o", result_yaml_path,
                "--stats", os.path.join(temp_dir,"stats.yaml"),
                "--cfg", algorithm_yaml_path,
                "-t", str(self.time_limit_ms),
            ]
            process = subprocess.run(command, cwd=self.cwd, capture_output=True, text=True)

            # useful error check cases were generated with ChatGPT:
            if process.returncode != 0:
                raise RuntimeError(f"db-lacam failed:\n{process.stdout[-2000:]=}\n{process.stderr[-2000:]=}")
            if not os.path.isfile(result_yaml_path):
                raise RuntimeError(f"db-lacam did not create an output yaml:\n{process.stdout[-2000:]=}\n{process.stderr[-2000:]=}")

            with open(result_yaml_path, "r", encoding="utf-8") as file:
                result_data = yaml.safe_load(file)
                
        # extracts one action plan per robot from db-lacam result:
        trajectories = result_data.get("result")

        self.cached_plans = []
        for robot, trajectory in zip(self.robots, trajectories):
            actions = np.asarray(trajectory.get("actions") or [], dtype=float)
            actions = actions.reshape(-1, robot.nu)
            self.cached_plans.append(actions)

        self.step_idx = 0

    def _get_current_actions(self):
        """
        returns one action per robot for current planning step
        """
        
        actions = []
        for robot, plan in zip(self.robots, self.cached_plans):
            if self.step_idx < len(plan):
                action = plan[self.step_idx].copy()
            else:
                # returns a zero action if a robot’s plan is already finished:
                action = np.zeros(robot.nu, dtype=float)

            actions.append(action)

        joint_action = np.concatenate(actions)
        return self.sim.validate_action(joint_action)
    
    def __call__(self, obs):
        
        try:
            # computes plan if no cached plan exists or all cached actions were used 
            # or replan mode is active and replan_freq steps passed:
            if (
                self.cached_plans is None
                or self.step_idx >= max((len(plan) for plan in self.cached_plans), default=0)
                or (
                    self.mode == "replan"
                    and self.step_idx >= self.replan_freq
                )
            ):
                self._compute_plan(obs)
            actions = self._get_current_actions()
            self.step_idx += 1
            return actions

        except RuntimeError as error:
            
            if self.raise_planning_error:
                raise
            warnings.warn(f"db-lacam planning failed: {error}")

            # creates one zero-action vector per robot in case of failure:
            zero_action = np.concatenate([np.zeros(robot.nu, dtype=float) for robot in self.robots])
            return self.sim.validate_action(zero_action)