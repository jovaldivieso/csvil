import numpy as np
import torch


class MultiRobotSimulator:
    """
    simulator wrapper for multi-robot setup, supporting heterogeneous robot teams
    """

    def __init__(self, robots, environment_min=(-6.0, -6.0), environment_max=(6.0, 6.0)):
       
        self.robots = list(robots)
        self.num_robots = len(self.robots)

        self.environment_min = np.asarray(environment_min,dtype=float)
        self.environment_max = np.asarray(environment_max,dtype=float)
        
        self.max_obs_dim = max(
            robot.get_dataset_features()["observation.state"]["shape"][0]
            for robot in self.robots
        )
        self.max_state_dim = max(robot.nx for robot in self.robots)
        self.max_action_dim = max(robot.nu for robot in self.robots)

        self.robot_type_names = list(dict.fromkeys(type(robot).__name__ for robot in self.robots))
        self.robot_type_indices = {name: index for index, name in enumerate(self.robot_type_names)}

        # all robots use same simulation clock:
        dts = [float(robot.dt) for robot in self.robots]

        if not all(np.isclose(dt, dts[0]) for dt in dts):
            raise ValueError("all robots must use same dt")
        
        self.dt = dts[0]
        self.state = None
        self.time = 0
        self.rng = np.random.default_rng()
    
    def reset(self, initial_states):
        
        self.state = [
            robot.reset(np.asarray(initial_state, dtype=float))
            for robot, initial_state in zip(self.robots, initial_states)
        ]
        self.time = 0
        return self.state

    def reset_random(self):
        """
        samples random initial states for all robots
        """

        states = [
            robot.random_initial_state(
                self.rng,
                self.environment_min,
                self.environment_max,
            )
            for robot in self.robots
        ]
        return self.reset(states)

    def observe(self, states):
        """
        creates one observation per robot
        """ 
        return [robot.observe(state) for robot, state in zip(self.robots, states)]
    
    def create_local_observations(self, states):
        """
        creates one robot centric dataset observation per robot
        """

        observations = []
        for robot_idx, (ego_robot, ego_state) in enumerate(zip(self.robots, states)):
            
            # pads ego observation to fixed dimension:
            ego_observation = np.zeros(self.max_obs_dim, dtype=np.float32)
            robot_observation = ego_robot.observe(ego_state)
            ego_observation[:len(robot_observation)] = robot_observation

            # encodes ego dynamics type with OHE:
            ego_type = np.zeros(len(self.robot_type_names), dtype=np.float32)
            ego_type[self.robot_type_indices[type(ego_robot).__name__]] = 1.0

            local_observation = [ego_observation, ego_type]

            # adds relative state of other robots:
            for other_idx, (other_robot, other_state) in enumerate(zip(self.robots, states)):
                if other_idx == robot_idx:
                    continue

                # converts neighbor position to ego relative position:
                relative_state = np.asarray(other_state, dtype=np.float32).copy()
                relative_state[:2] -= ego_state[:2]

                neighbor_state = np.zeros(self.max_state_dim, dtype=np.float32)
                neighbor_state[:len(relative_state)] = relative_state
                
                # encodes neighbor dynamics type with OHE:
                neighbor_type = np.zeros(len(self.robot_type_names), dtype=np.float32)
                neighbor_type[self.robot_type_indices[type(other_robot).__name__]] = 1.0

                local_observation.extend([neighbor_state, neighbor_type])

            observations.append(np.concatenate(local_observation))

        return observations                
                
    def step(self, states, actions):
        
        self.state = []
        for robot, state, action in zip(self.robots, states, actions):
            robot.state = robot.step(state, action)
            robot.time += 1
            
            self.state.append(robot.state)

        self.time += 1
        return self.state

    def is_done(self, states):
        """
        returns true if all robots reached their goals
        """
        return all(robot.is_done(state) for robot, state in zip(self.robots, states))
    
    def get_dataset_features(self):
        """
        creates feature definitions for one robot centric sample
        
        example:
            robots = [SingleIntegrator(), Unicycle1()],
        
            single_integrator_state = [1.0, 2.0]
            unicycle_state = [3.0, 2.5, 1.57]
            
            observation of robot 0:
            [
                goal_rel_x,
                goal_rel_y,
                0.0,       # padded (because single integrator has no heading)
                1.0, 0.0,  # ego is single integrator
                2.0,       # neighbor x - ego x
                0.5,       # neighbor y - ego y
                1.57,      # neighbor heading
                0.0, 1.0,  # neighbor is unicycle
            ]
        """

        obs_names = [f"ego_obs_{index}" for index in range(self.max_obs_dim)]
        obs_names.extend([f"ego_type_{name.lower()}" for name in self.robot_type_names])

        for neighbor_ind in range(self.num_robots - 1):
            obs_names.extend(
                [
                    f"neighbor_{neighbor_ind}_rel_x",
                    f"neighbor_{neighbor_ind}_rel_y",
                ]
            )

            obs_names.extend(
                [
                    f"neighbor_{neighbor_ind}_state_{index}"
                    for index in range(2, self.max_state_dim)
                ]
            )

            obs_names.extend(
                [
                    (f"neighbor_{neighbor_ind}_type_{name.lower()}")
                    for name in self.robot_type_names
                ]
            )

        action_names = [f"action_{index}" for index in range(self.max_action_dim)]

        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (len(obs_names),),
                "names": obs_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(obs_names),),
                "names": obs_names,
            },
            "action": {
                "dtype": "float32",
                "shape": (self.max_action_dim,),
                "names": action_names,
            },
            "robot_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["robot_index"],
            },
            "robot_type": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["robot_type"],
            },
        }

    def format_dataset_frame(self, robot_ind, observation, action):
        """
        creates one lerobot frame for one robot
        """
        observation = torch.tensor(observation, dtype=torch.float32)
        # fixed-size action vector:
        padded_action = np.zeros(self.max_action_dim, dtype=np.float32)
        padded_action[:len(action)] = action
        
        return {
            "observation.environment_state": observation,
            "observation.state": observation,
            "action": torch.tensor(padded_action, dtype=torch.float32),
            "robot_index": torch.tensor([robot_ind], dtype=torch.int64),
            "robot_type": torch.tensor([self.robot_type_indices[type( self.robots[robot_ind]).__name__ ]],
                dtype=torch.int64,
            ),
        }

    @property
    def goal_states(self):
        return [robot.goal_state for robot in self.robots]