import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collection import DataCollector
from planning.single_robot_casadi import SingleRobotCasadiPlanner
from systems.single_integrator import SingleIntegrator

config = {"dt": 0.05, "max_accel": 2.0, "horizon": 20, "goal": [1.0, 1.0]}

simulator = SingleIntegrator(config)
planner = SingleRobotCasadiPlanner(simulator, config)

collector = DataCollector(
    simulator,
    repo_id="local/single_integrator_casadi_expert",
    local_dir="data/lerobot_dataset_single_integrator_casadi"
)

collector.collect_trajectories(planner, num_trajectories=100, num_steps=100)
