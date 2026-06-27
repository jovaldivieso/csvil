import sys
import os

# Modify the Python path so it can find custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collection import DataCollector
from planning.single_robot_casadi import SingleRobotCasadiPlanner
from systems.single_integrator import SingleIntegrator

# System configuration
config = {"dt": 0.05, "max_speed": 1.0, "horizon": 20, "goal": [1.0, 1.0]}

# Initialize modules
simulator = SingleIntegrator(config)
planner = SingleRobotCasadiPlanner(config)
collector = DataCollector(simulator)

# Build demonstration dataset
collector.collect_trajectories(planner, num_trajectories=100, num_steps=100)
