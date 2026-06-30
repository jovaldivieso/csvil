# Controller Synthesis via Imitation Learning

This repository contains a modular pipeline for controller synthesis via imitation learning. The primary goal is to imitate a computationally heavy motion planner using a faster neural policy to accelerate inference.

## Installation & setup

### Prerequisites
This project was tested so far with **Python 3.11** and **3.12**. 

### Clone the repository
Clone the project to your local machine and navigate into the root directory:
```bash
git clone git@github.com/jovaldivieso/csvil.git
cd csvil
```

### Set up a virtual environment
Set up the native Python virtual environment (`venv`) or `conda` if you are following the LeRobot tutorials. Example with Python's `venv`:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies
Once your virtual environment is active, install the required packages:
```bash
pip install -r requirements.txt
```
This will install LeRobot, CasADi, and their required dependencies like PyTorch and NumPy.

### Hugging Face futhentication
Because the pipeline uses LeRobot to manage datasets and model checkpoints via the Hugging Face Hub, you should authenticate your terminal if you want to use this feature.
1. Create an account at [huggingface.co](https://huggingface.co/).
2. Go to **Settings > Access Tokens** and create a new token with **Write** permissions.
3. Run the following command in your terminal and paste your token when prompted:
```bash
hf auth login
```

---

## File Structure

```text
csvil/
├── data/
│   ├── data_collection.py
│   ├── lerobot_dataset_double_integrator_casadi/   # Example generated dataset
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
Run a data collection script to simulate the motion planner expert and save the state or observation and action data of the given system in the LeRobot dataset format. Example with the double integrator system and CasADi based motion planner:

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

### Imitation learning with diffusion policy
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
