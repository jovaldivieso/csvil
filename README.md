# Controller Synthesis via Imitation Learning

This repository contains a modular pipeline for controller synthesis via imitation learning. The primary goal is to imitate a computationally heavy motion planner using a faster neural policy accelerate inference.

## File Structure

```text
csvil/
├── data/
│   ├── data_collection.py
│   ├── lerobot_dataset_double_integrator_casadi/   # Generated dataset
│   └── lerobot_dataset_single_integrator_casadi/
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
    ├── double_integrator_casadi_diffusion_policy.py # Evaluation script
    ├── double_integrator_casadi_example.py          # Data generation
    ├── double_integrator_casadi_plot.py
    ├── gym_double_integrator_casadi_plot.py
    ├── single_integrator_casadi_example.py
    └── single_integrator_casadi_plot.py
```

## Pipeline Execution

Follow these steps to generate expert data, train the neural network, and evaluate the cloned policy.

### 1. Generate the Expert Dataset
Run the data collection script to simulate the expert CasADi planner and save the state-action data. This script uses Uniform Polar Sampling to ensure full 360-degree workspace coverage.

```bash
python test/double_integrator_casadi_example.py
```
*(This will automatically save a Hugging Face compatible dataset to `data/lerobot_dataset_double_integrator_casadi`)*

### 2. Visualize the Dataset (Optional)
Once the dataset is generated, you can use LeRobot's native CLI tool to launch a local web visualizer and inspect the expert trajectories:

```bash
lerobot-dataset-viz \
  --repo-id local/double_integrator_casadi_expert \
  --root data/lerobot_dataset_double_integrator_casadi \
  --mode local \
  --episode-index 0
```

### 3. Train the Diffusion Policy
Train the neural network using the pre-configured YAML settings. The script will automatically stop and save the model weights once it reaches the target steps.

```bash
python learning/training.py --config learning/double_integrator_casadi_diffusion_policy_config.yaml
```
*(Checkpoints will be saved automatically in the auto-generated `outputs/train/` directory)*

### 4. Evaluate the Cloned Policy
Test the trained neural network independently of the training loop. Make sure to pass your latest timestamped output folder to the `--model_dir` argument!

```bash
python test/double_integrator_casadi_diffusion_policy.py \
  --model_dir outputs/train/2026-06-29/[YOUR_TIMESTAMP]/checkpoints/015000/pretrained_model
```
*(This will run an autonomous rollout using the trained policy and output a PDF plot of the trajectory to the workspace)*
