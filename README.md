# Controller Synthesis via Imitation Learning

This repository contains a modular pipeline for controller synthesis via imitation learning. The primary goal is to imitate a computationally heavy motion planner using a faster neural policy accelerate inference.

## File Structure

```text
csvil/
├── data/
│   ├── data_collection.py
│   ├── lerobot_dataset_double_integrator_casadi/   # Generated dataset
│   └── lerobot_dataset_single_integrator_casadi/   # Generated dataset
├── learning/
│   ├── double_integrator_casadi_diffusion_policy_config.yaml
│   └── training.py
├── planning/
│   └── single_robot_casadi.py
├── systems/
│   ├── __init__.py
│   ├── double_integrator.py
│   ├── dynamics.py
│   ├── gym_wrapper.py
│   └── single_integrator.py
└── test/
    ├── double_integrator_casadi_data.py
    ├── double_integrator_casadi_diffusion_policy.py
    ├── double_integrator_casadi_plot.py
    ├── gym_double_integrator_casadi_plot.py
    ├── single_integrator_casadi_data.py
    └── single_integrator_casadi_plot.py
```

## Pipeline Features

Follow these steps to generate expert data, train the neural network, and evaluate the cloned policy.

### Generate a motion planning expert dataset
Run a data collection script to simulate the motion planner expert and save the state or observation and action data of the given system in the LeRoboDataset format. Example with double integrator system and CasADi based motion planner:

```bash
python test/double_integrator_casadi_data.py
```
This will automatically save a Hugging Face compatible dataset to `data/lerobot_dataset_double_integrator_casadi`.

### Visualize the dataset
Once the dataset is generated, you can use LeRobot's native CLI tool to launch a local web visualizer and inspect the expert trajectories:

```bash
lerobot-dataset-viz \
  --repo-id local/double_integrator_casadi_expert \
  --root data/lerobot_dataset_double_integrator_casadi \
  --mode local \
  --episode-index 0
```

### Imitation learning with diffusion dolicy
Train the neural network using the pre-configured YAML settings. The script will automatically stop and save the model weights once it reaches the target steps.

```bash
python learning/training.py --config learning/double_integrator_casadi_diffusion_policy_config.yaml
```
Checkpoints will be saved automatically in the auto-generated `outputs/train/` directory.

### Evaluate the cloned policy
Test the trained neural network independently of the training loop. Make sure to pass your latest timestamped output folder or Hugging Face Hub ID to the `--model-dir` argument.

```bash
python test/double_integrator_casadi_diffusion_policy.py --model-dir jovaldivieso/double_integrator_casadi_diffusion_policy
```
This will run an autonomous rollout using the trained policy and output a PDF plot of the trajectory to the workspace.
