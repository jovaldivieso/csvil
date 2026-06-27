# Controller Synthesis via Imitation Learning

This repository contains a modular pipeline for controller synthesis via imitation learning, the idea is to imitate a motion planner to accelerate inference.

## File Structure

```text
csvil/
├── data/
│   ├── data_collection.py
│   ├── lerobot_dataset_double_integrator_casadi/
│   └── lerobot_dataset_single_integrator_casadi/
├── planning/
│   └── single_robot_casadi.py
├── systems/
│   ├── dynamics.py
│   ├── double_integrator.py
│   └── single_integrator.py
└── test/
    ├── double_integrator_casadi_example.py
    ├── double_integrator_casadi_plot.py
    ├── inspect_dataset.py
    ├── single_integrator_casadi_example.py
    └── single_integrator_casadi_plot.py
```

## Generating and Visualizing Data

To generate a dataset and view the expert trajectories, follow these two steps:

**1. Generate the Dataset**

Run one of the example scripts to simulate the expert planner and save the data. For example, to generate the Double Integrator dataset, run:

```bash
python test/double_integrator_casadi_example.py
```

**2. Visualize the Dataset**

Once the dataset is generated, use LeRobot's native CLI tool to launch a local web visualizer and inspect the trajectories (starting at episode 0):

```bash
lerobot-dataset-viz \
  --repo-id double_integrator_expert \
  --root data/lerobot_dataset_double_integrator_casadi \
  --mode local \
  --episode-index 0
```
