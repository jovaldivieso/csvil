from data.data_collection import DataCollector
from planning.casadi_planner import CasadiPlanner

###############################
####### data collection #######
###############################

def collect_casadi_expert_data(
    simulator_class,
    config,
    repo_id,
    local_dir,
    num_traj,
    num_steps,
):
    """
    generates and saves expert trajectories for a dynamics system

    creates a simulator, CasADi planner and data collector, 
    then stores generated expert trajectories as a local LeRobot dataset

    args:
        simulator_class: dynamics simulator class to instantiate (e.g. Unicycle2)
        config: configuration dictionary for simulator and planner
        repo_id: identifier stored in LeRobot dataset metadata
        local_dir: local directory where the generated dataset is saved
        num_traj: number of expert trajectories to collect
        num_steps: maximum number of simulation steps per trajectory

    returns:
        result of DataCollector.collect_trajectories()
    """
    
    simulator = simulator_class(config)
    planner = CasadiPlanner(simulator, config)

    collector = DataCollector(
        simulator=simulator,
        repo_id=repo_id,
        local_dir=local_dir,
    )

    return collector.collect_trajectories(
        motion_planner=planner,
        num_trajectories=num_traj,
        num_steps=num_steps,
    )
 