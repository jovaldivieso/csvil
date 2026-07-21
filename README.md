# Controller Synthesis via Imitation Learning

This repository contains a modular pipeline for controller synthesis via imitation learning. The goal is to imitate a computationally expensive motion planner with a faster neural policy for online control.

The current expert planner uses CasADi. A separate Docker environment for future work with db-LaCAM is included as well.

Recent refactors added strict runtime shape checks for states/observations/actions,
centralized simulator/planner factories, and validated typed system configuration loading.

## Installation & setup

### Clone the repository
Clone the project to your local machine and navigate into the root directory:
```bash
git clone git@github.com:jovaldivieso/csvil.git
cd csvil
```

### Install Docker
Docker is used so every contributor runs the same dependency stack (this is particularly useful on Intel Macs, since some newer PyTorch versions required by LeRobot are not available as native macOS Intel x86_64 packages).

[Install](https://docs.docker.com/desktop/setup/install/windows-install/) Docker Desktop. On macOS, open it from your Applications folder and wait until Docker Desktop indicates that it is running. Verify the installation:

```bash
docker --version
docker compose version
```

`compose.yaml` defines the runnable services. The `csvil` service is the main Python environment and runs the project in a Linux amd64 container. The `db-lacam` service is a separate environment for db-LaCAM and its C++ dependencies.

### Build the project environment

Build the project environment once:

```bash
docker compose build csvil
```

Run project commands through Docker:

```bash
docker compose run --rm csvil <command>
```

The repository is available as `/workspace` inside the container. The `--rm` option removes the temporary container after the command completes (but does not remove the image, generated files or Hugging Face cache).

The db-LaCAM environment is optional and is not needed for the standard CasADi workflow:

```bash
docker compose build db-lacam
```
The first build may take several minutes. Rebuilding is only necessary when changing the Dockerfile, compose-file or requirements-file.

### Hugging Face authentication
Hugging Face authentication is only needed when uploading datasets or models to the Hub.
1. Create an account at [huggingface.co](https://huggingface.co/).
2. Go to **Settings > Access Tokens** and create a new token with **Write** permissions.
3. Run the following command in your terminal and paste your token when prompted:

```bash
docker compose run --rm csvil hf auth login
```

## Core file structure

```text
csvil/
├── core/
│   ├── config.py            # Typed validation and normalized YAML loading
│   ├── factory.py           # DynamicsFactory + PlannerFactory registries
│   └── types.py             # Shared vector schemas and dimension checks
├── data/
│   ├── data_collection.py
│   └── lerobot_dataset_double_integrator_casadi/  # Example generated dataset
├── learning/
│   ├── config/
│   │   └── double_integrator_casadi_diffusion_policy_config.yaml  # Example training config
│   └── training.py
├── planning/
│   ├── planner.py            # Base class
│   └── casadi_planner.py     # Example subclass
├── systems/
│   ├── dynamics.py           # Base class
│   └── double_integrator.py  # Example subclass
└── test/
    ├── config/
    │   └── double_integrator_casadi_config.yaml  # Example expert config
    ├── collect_casadi_expert_data.py
    ├── evaluate_lerobot.py 
    ├── plot_casadi_trajectories.py
    └── utils.py
```

The available systems are:

- `single_integrator`
- `double_integrator`
- `unicycle1`
- `unicycle2`

Each planner and corresponding dynamics use a yaml configuration file in `test/config/`. These files contain the simulator and planner parameters, including e.g. the time step, MPC horizon, goal state and system-specific limits.

System configs are now validated through `core/config.py` before simulation/evaluation
to catch malformed keys and shape mismatches early. Planner-specific keys include:

- `horizon`
- `mode` (`mpc` or `open_loop`)
- `Q_diag` (must match state dimension `nx`)
- `R_weight`
- `terminal_cost_multiplier`

Simulator and planner creation is centralized through `DynamicsFactory` and
`PlannerFactory` in `core/factory.py`.

## Pipeline features

### Generate a motion planning expert dataset

Generate trajectories from the CasADi expert and save them in the LeRobot dataset format:

```bash
docker compose run --rm csvil \
  python test/collect_casadi_expert_data.py \
  --system double_integrator \
  --config test/config/double_integrator_casadi_config.yaml
```

If the CasADi solver fails for a rollout, that trajectory is skipped and not
written to the dataset. The collector keeps sampling until the requested number
of successful trajectories is reached (or a safety max-attempt threshold is hit).

The Hugging Face compatible dataset is saved locally in a directory similar to:

```text
data/lerobot_dataset_double_integrator_casadi/
```

### Visualize expert trajectories

Create a PDF plot of trajectories generated by the CasADi expert:

```bash
docker compose run --rm csvil \
  python test/plot_casadi_trajectories.py \
  --system double_integrator \
  --config test/config/double_integrator_casadi_config.yaml
```

### Visualize the dataset
Once the dataset is generated, you can use LeRobot's native CLI tool to launch a local web visualizer and inspect the expert trajectories:

```bash
docker compose run --rm csvil \
  lerobot-dataset-viz \
  --repo-id local/double_integrator_casadi_expert \
  --root data/lerobot_dataset_double_integrator_casadi \
  --mode local \
  --episode-index 0
```

### Train a diffusion policy with imitation learning

Train a diffusion policy using the matching YAML configuration:

```bash
docker compose run --rm csvil \
  python learning/training.py \
  --config learning/config/double_integrator_casadi_diffusion_policy_config.yaml
```

The training entrypoint now exposes `run_training(config_path)` for programmatic
use in sweep/automation scripts while preserving the same CLI behavior.

Training outputs and checkpoints are saved in the configured output directory.

### Evaluate the learned policy

Evaluate a trained policy in the selected simulator independently of the training loop. Make sure to pass your latest timestamped output folder or Hugging Face Hub ID to the --model-dir argument:

```bash
docker compose run --rm csvil \
  python test/evaluate_lerobot.py \
  --system double_integrator \
  --policy-type diffusion \
  --config test/config/double_integrator_casadi_config.yaml \
  --model-dir outputs/train/<run-name>/checkpoints/<checkpoint-name>
```

The script runs a rollout using the trained policy and saves a trajectory plot.

The evaluation, plotting, and expert collection scripts also expose importable
execution functions (`run_evaluation`, `run_plotting`, `run_collection`) so they
can be called from Python workflows without shelling out to CLI.
