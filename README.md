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
│   ├── models/
│   │   └── mlp.py             # Custom MLP policy used for BC/DAgger
│   ├── train_dagger.py        # Standalone DAgger training loop for the MLP
│   └── train_lerobot.py       # LeRobot training entrypoint (ACT / Diffusion)
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
  python learning/train_lerobot.py \
  --config learning/config/double_integrator_casadi_diffusion_policy_config.yaml
```

The training entrypoint now exposes `run_training(config_path)` for programmatic
use in sweep/automation scripts while preserving the same CLI behavior.

Training outputs and checkpoints are saved in the configured output directory.

### Train a custom MLP policy with DAgger

Train the custom MLP baseline using a standalone PyTorch implementation of DAgger
(Ross et al., 2011). The script starts from an existing offline expert dataset,
then iteratively rolls out the learner, queries the expert on visited states,
and appends corrective labels to the same LeRobot dataset.

```bash
docker compose run --rm csvil \
  python learning/train_dagger.py \
  --system double_integrator \
  --config test/config/double_integrator_casadi_config.yaml \
  --repo-id local/double_integrator_casadi_expert \
  --dataset-root data/lerobot_dataset_double_integrator_casadi \
  --dagger-iterations 5 \
  --epochs-per-iteration 20 \
  --trajectories-per-iteration 20 \
  --steps-per-trajectory 150
```

The script writes checkpoints to `outputs/train_dagger/` after each DAgger
iteration, including:

- `mlp_dagger_checkpoint.pt` (latest)
- `mlp_dagger_iter_XXX.pt` (per-iteration snapshots)

These checkpoints store a dictionary containing `model_state_dict`, optimizer
state, and metadata (`state_dim`, `action_dim`, feature names, iteration).

If a DAgger run crashes mid-write and later reports parquet footer errors,
recreate the dataset directory before restarting. The trainer now finalizes
LeRobot writer state per iteration to keep appended parquet chunks readable.

### Evaluate the learned policy

Evaluate a trained policy in the selected simulator independently of the
training loop. `--policy-type` supports `diffusion`, `act`, and `mlp`.
For `diffusion`/`act`, pass the checkpoint directory (or Hub model ID).
For `mlp`, pass the `.pt` checkpoint file path.

Diffusion example:

```bash
docker compose run --rm csvil \
  python test/evaluate_lerobot.py \
  --system double_integrator \
  --policy-type diffusion \
  --config test/config/double_integrator_casadi_config.yaml \
  --model-dir outputs/train/<run-name>/checkpoints/<checkpoint-name>
```

MLP (DAgger checkpoint) example:

```bash
docker compose run --rm csvil \
  python test/evaluate_lerobot.py \
  --system double_integrator \
  --policy-type mlp \
  --config test/config/double_integrator_casadi_config.yaml \
  --model-dir outputs/train_dagger/mlp_dagger_checkpoint.pt
```

The script runs a rollout using the trained policy and saves a trajectory plot.

### Quickstart: BC vs DAgger vs ACT/Diffusion

Use this block as a minimal comparison workflow for the same system.

1. Generate an expert dataset once:

```bash
docker compose run --rm csvil \
  python test/collect_casadi_expert_data.py \
  --system double_integrator \
  --config test/config/double_integrator_casadi_config.yaml
```

2. BC-style MLP warm start (offline fit only):
   The current MLP pipeline is implemented through DAgger. Setting
   `--dagger-iterations 1` trains on the pre-existing offline dataset first
   (the parameter-free DAgger initialization stage).

```bash
docker compose run --rm csvil \
  python learning/train_dagger.py \
  --system double_integrator \
  --config test/config/double_integrator_casadi_config.yaml \
  --repo-id local/double_integrator_casadi_expert \
  --dataset-root data/lerobot_dataset_double_integrator_casadi \
  --dagger-iterations 1 \
  --epochs-per-iteration 20 \
  --trajectories-per-iteration 1 \
  --steps-per-trajectory 150
```

3. Full MLP + DAgger training (iterative aggregation):

```bash
docker compose run --rm csvil \
  python learning/train_dagger.py \
  --system double_integrator \
  --config test/config/double_integrator_casadi_config.yaml \
  --repo-id local/double_integrator_casadi_expert \
  --dataset-root data/lerobot_dataset_double_integrator_casadi \
  --dagger-iterations 5 \
  --epochs-per-iteration 20 \
  --trajectories-per-iteration 20 \
  --steps-per-trajectory 150
```

4. ACT / Diffusion baselines (LeRobot trainers):

```bash
# Diffusion
docker compose run --rm csvil \
  python learning/train_lerobot.py \
  --config learning/config/double_integrator_casadi_diffusion_policy_config.yaml

# ACT
docker compose run --rm csvil \
  python learning/train_lerobot.py \
  --config learning/config/double_integrator_casadi_act_config.yaml
```

Evaluate commands:

```bash
# MLP
docker compose run --rm csvil \
  python test/evaluate_lerobot.py \
  --system double_integrator \
  --policy-type mlp \
  --config test/config/double_integrator_casadi_config.yaml \
  --model-dir outputs/train_dagger/mlp_dagger_checkpoint.pt

# Diffusion
docker compose run --rm csvil \
  python test/evaluate_lerobot.py \
  --system double_integrator \
  --policy-type diffusion \
  --config test/config/double_integrator_casadi_config.yaml \
  --model-dir outputs/train/<run-name>/checkpoints/<checkpoint-name>

# ACT
docker compose run --rm csvil \
  python test/evaluate_lerobot.py \
  --system double_integrator \
  --policy-type act \
  --config test/config/double_integrator_casadi_config.yaml \
  --model-dir outputs/train/<run-name>/checkpoints/<checkpoint-name>
```

The evaluation, plotting, and expert collection scripts also expose importable
execution functions (`run_evaluation`, `run_plotting`, `run_collection`) so they
can be called from Python workflows without shelling out to CLI.
