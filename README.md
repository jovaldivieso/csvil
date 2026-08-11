# Controller Synthesis via Imitation Learning

This repository contains a modular pipeline for controller synthesis via imitation learning. The goal is to imitate a computationally expensive motion planner with a faster neural policy for online control.

The current expert planner uses CasADi, but the architecture is designed to plug in additional planner backends (for example, OMPL) with minimal code-path changes. A separate Docker environment for work with the db-LaCAM multi-robot global planner is included as well.

Design-wise, the codebase follows a small set of explicit patterns so new systems
and planners can typically be added without changing the overall pipeline shape.

Core design patterns and rationale:

- **Composite Pattern (Fleet-first system abstraction)**
  `MultiRobotSimulator` is the primary simulator boundary: it composes per-robot simulators and exposes one shared state/action/observation interface. A single robot is treated as the fleet-of-1 case (`num_robots = 1`), so data collection, evaluation, and training all reuse the same path without branching on single- vs multi-agent mode.

- **Immutable Value Objects (Math-first geometric abstraction)**
  Manifold-valued states are represented as typed, immutable objects such as `SO2State` and `SE2PoseState` instead of mutable raw arrays. This keeps geometric operations explicit, prevents accidental in-place edits, and confines coordinate-chart conversions to clear API boundaries.

- **Protocol-Based Structural Typing (Duck Typing)**
  `DynamicsProtocol` and `PlannerProtocol` define the required runtime contracts in [systems/dynamics.py](systems/dynamics.py) and [planning/planner.py](planning/planner.py). New components integrate by matching those interfaces, which keeps extension work local and avoids deep inheritance coupling.

- **Factory Pattern (Centralized component instantiation)**
  `DynamicsFactory` and `PlannerFactory` in `core/factory.py` centralize simulator and planner construction. In practice, adding a new backend is usually a registry and config change rather than edits scattered across rollout, evaluation, and training scripts.

- **Defensive Programming and Boundary Contracts**
  Configs are normalized and validated in `core/config.py`, and vector dimensions are checked at simulator, planner, and learning boundaries. That turns YAML mistakes and shape mismatches into early local failures instead of wasted rollouts, training runs, or corrupted datasets.

In practice, this means most features are implemented once at the fleet level,
then reused across single-robot experiments, multi-robot experiments, and
mixed future extensions with minimal system-specific changes.

## Installation & setup

### Clone the repository
Clone the project to your local machine and navigate into the root directory:
```bash
git clone git@github.com:jovaldivieso/csvil.git
cd csvil
```

### Install Docker
Docker is used so every contributor runs the same dependency stack (this is particularly useful on Intel Macs, since some newer PyTorch versions required by LeRobot are not available as native macOS Intel x86_64 packages).

[Install](https://docs.docker.com/desktop/setup/install/) Docker Desktop. On macOS, open it from your Applications folder and wait until Docker Desktop indicates that it is running. Verify the installation:

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
├── README.md                  # End-to-end usage and experiment recipes
├── compose.yaml               # Docker services for csvil and optional db-lacam
├── requirements.txt           # Python dependencies installed in the csvil image
├── docker/
│   ├── Dockerfile             # Main csvil runtime image
│   └── Dockerfile.db-lacam    # Optional db-lacam image
├── core/
│   ├── config.py             # Typed validation and normalized YAML loading
│   ├── factory.py            # DynamicsFactory + PlannerFactory registries
│   └── types.py              # Shared vector schemas and dimension checks
├── data/
│   ├── data_collection.py    # LeRobot writer helpers and rollout recording
│   ├── lerobot_dataset_double_integrator_casadi/  # Example generated dataset
│   └── lerobot_dataset_multi_robot_casadi/        # Example generated multi-robot dataset
├── learning/
│   ├── config/
│   │   ├── double_integrator_casadi_act_config.yaml
│   │   ├── double_integrator_casadi_diffusion_policy_config.yaml
│   │   ├── multi_double_integrator_casadi_act_config.yaml
│   │   ├── multi_double_integrator_casadi_diffusion_policy_config.yaml
│   │   └── multi_unicycle2_casadi_mlp_config.yaml
│   ├── models/
│   │   └── mlp.py             # Custom MLP policy used for BC/DAgger
│   ├── dagger_evaluation.py   # Shared in-loop DAgger evaluation helpers
│   ├── train_dagger.py        # Iterative DAgger for the custom MLP baseline
│   ├── train_lerobot.py       # LeRobot training entrypoint (ACT / Diffusion)
│   └── train_lerobot_dagger.py  # Iterative DAgger wrapper for LeRobot policies
├── planning/
│   ├── planner.py             # Planner protocol and base class
│   ├── casadi_planner.py      # CasADi planner implementation
│   └── dblacam_planner.py     # db-lacam planner implementation
├── systems/
│   ├── dynamics.py            # Base simulator protocol and validation
│   ├── initial_state_utils.py # Shared initial-state parsing/normalization
│   ├── multi_robot.py         # Fleet composition wrapper over per-robot simulators
│   └── seed_utils.py          # Seed defaults and deterministic rollout seeding
├── outputs/
│   ├── plots/                 # Evaluation and expert-rollout plots/videos
│   ├── train/                 # LeRobot training outputs
│   ├── train_dagger/          # Single-robot MLP DAgger checkpoints
│   └── train_dagger_multi_robot/  # Multi-robot MLP DAgger checkpoints
└── test/
    ├── config/
    │   ├── double_integrator_casadi_config.yaml
    │   ├── multi_double_integrator_casadi_config.yaml
    │   ├── multi_unicycle2_casadi_config.yaml
    │   └── multi_robot_dblacam_config.yaml
    ├── collect_expert_data.py    # CLI for expert dataset generation
    ├── evaluate_policy.py        # CLI for rollout/evaluation across policy families
    ├── plot_expert_trajectories.py  # CLI plotting helper for expert rollouts
    ├── plot_trajectories.py      # Legacy single-robot plot helper (CasADi/db-lacam)
    ├── plot_trajectories_multi_robot.py  # Legacy db-lacam multi-robot plot helper
    └── test_simulator_contracts.py  # Fleet-of-1 and schema consistency tests
```

The available systems are:

- `single_integrator`
- `double_integrator`
- `unicycle1`
- `unicycle2`
- `multi_robot` (composite fleet, e.g. multiple double integrators)

These cover different dynamics classes under one pipeline shape: holonomic systems
(`single_integrator`, `double_integrator`) and non-holonomic systems
(`unicycle1`, `unicycle2`). The current built-in dynamics examples span both fully actuated and underactuated systems.

Available planners are:

- `casadi` (single- and multi-robot)
- `dblacam` (currently wired for selected workflows/systems; see db-lacam configs and plotting helpers)

Each planner and corresponding dynamics use a yaml configuration file in `test/config/`. These files contain simulator and planner parameters, for example the time step, MPC horizon, goal state, and system-specific limits.

Initial-state sampling is also configured there and is shared by data collection,
DAgger rollouts, evaluation rollouts, and expert-trajectory plots. The canonical
fields are:

- `initial_position_radius_bounds`: radial workspace around the configured goal
- `initial_position_min_goal_distance`: reject/resample starts too close to the goal
- `initial_state_seed`: reproducible RNG seed for unseeded resets such as collection and DAgger
- `action_noise_seed`: deterministic seed used for rollout action-noise sampling

System configs are now validated through `core/config.py` before simulation/evaluation
to catch malformed keys and shape mismatches early. Planner-specific keys include:

- `horizon`
- `mode` (`mpc` or `open_loop`)
- `Q_diag` (must match state dimension `nx`)
- `R_weight` (scalar fallback)
- `R_diag` (optional full action-space diagonal, length `nu`)
- `R_weight_per_robot` (optional `multi_robot` override; one entry per robot, each scalar or per-action list)
- `terminal_cost_multiplier`
- `collision_slack_penalty_weight` (positive scalar penalty for collision slack in soft pairwise avoidance)

Simulator and planner creation is centralized through `DynamicsFactory` and
`PlannerFactory` in `core/factory.py`.

Canonical multi-robot example:

- `test/config/multi_double_integrator_casadi_config.yaml`
  (2x homogeneous `double_integrator` robots with shared MPC planner settings and global `d_safe`)
- `learning/config/multi_double_integrator_casadi_diffusion_policy_config.yaml`
- `learning/config/multi_double_integrator_casadi_act_config.yaml`

Canonical multi-robot `unicycle2` + MLP DAgger example:

- `test/config/multi_unicycle2_casadi_config.yaml`
  (2x homogeneous `unicycle2` robots with shared MPC planner settings and global `d_safe`)
- `learning/config/multi_unicycle2_casadi_mlp_config.yaml`

## Pipeline features

### Generate a motion planning expert dataset

Generate trajectories from the CasADi expert and save them in the LeRobot dataset format:

```bash
docker compose run --rm csvil \
python test/collect_expert_data.py \
--system double_integrator \
--planner casadi \
--config test/config/double_integrator_casadi_config.yaml \
--action-noise-std 0.0 \
--num-traj 100
```

If the CasADi solver fails for a rollout, that trajectory is skipped and not
written to the dataset. The collector keeps sampling until the requested number
of successful trajectories is reached (or a safety max-attempt threshold is hit).

The Hugging Face compatible dataset is saved locally in a directory similar to:

```text
data/lerobot_dataset_double_integrator_casadi/
```

Example 2-robot double-integrator data collection command with the CasADi expert:

```bash
docker compose run --rm csvil \
python test/collect_expert_data.py \
--system multi_robot \
--planner casadi \
--config test/config/multi_double_integrator_casadi_config.yaml \
--action-noise-std 0.0 \
--num-traj 100
```

Without `--repo-id` / `--local-dir`, the collector uses the default naming
`local/multi_robot_casadi_expert` and `data/lerobot_dataset_multi_robot_casadi`
based only on `--system` and `--planner`.

#### Action-noise injection for robust offline datasets

The expert collector supports execution-time action noise through:

- `--action-noise-std <float>` (default: `0.0`)

This widens the visited state distribution while preserving expert supervision:

- The dataset logs the clean expert action (planner output) for each observed state.
- The simulator executes a noisy action sampled as Gaussian perturbation with std-dev `action_noise_std`.
- The executed action is clipped to simulator limits `[-max_action, max_action]` before stepping dynamics.

This creates trajectories that drift off nominal paths but are still paired with optimal recovery labels for ACT/Diffusion training.

Example with noise enabled:

```bash
docker compose run --rm csvil \
python test/collect_expert_data.py \
--system double_integrator \
--planner casadi \
--config test/config/double_integrator_casadi_config.yaml \
--action-noise-std 0.05
```

#### Collision-avoidance-focused sampling strategy

To intentionally enrich the dataset with near-conflict interaction cases, use
`--initial-states` to front-load specific multi-robot starts where agents must
move around each other to reach their goals. After the provided list is
exhausted, collection automatically falls back to random simulator sampling.

Example:

```bash
docker compose run --rm csvil \
python test/collect_expert_data.py \
--system multi_robot \
--planner casadi \
--config test/config/multi_double_integrator_casadi_config.yaml \
--action-noise-std 0.0 \
--initial-states '[
  [[ 0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
  [[ 0.0, 0.95, 0.0, 0.0], [0.0, -0.95, 0.0, 0.0]],
  [[ 0.0, 0.9, 0.0, 0.0], [0.0, -0.9, 0.0, 0.0]],
  [[ 0.0, 0.85, 0.0, 0.0], [0.0, -0.85, 0.0, 0.0]],
  [[ 0.0, 0.8, 0.0, 0.0], [0.0, -0.8, 0.0, 0.0]],
  [[ 0.0, 0.75, 0.0, 0.0], [0.0, -0.75, 0.0, 0.0]],
  [[ 0.0, 0.7, 0.0, 0.0], [0.0, -0.7, 0.0, 0.0]],
  [[ 0.0, 0.65, 0.0, 0.0], [0.0, -0.65, 0.0, 0.0]],
  [[ 0.0, 0.6, 0.0, 0.0], [0.0, -0.6, 0.0, 0.0]],
  [[ 0.0, 0.55, 0.0, 0.0], [0.0, -0.55, 0.0, 0.0]],
  [[ 0.0, 0.5, 0.0, 0.0], [0.0, -0.5, 0.0, 0.0]]
]'
```

### Visualize expert trajectories

Create a PDF plot and animation of trajectories generated by the CasADi expert:

```bash
docker compose run --rm csvil \
python test/plot_expert_trajectories.py \
--system double_integrator \
--planner casadi \
--config test/config/double_integrator_casadi_config.yaml \
--num-steps 150 \
--action-noise-std 0.03
```

For a multi-robot setup:

```bash
docker compose run --rm csvil \
python test/plot_expert_trajectories.py \
--system multi_robot \
--planner casadi \
--config test/config/multi_double_integrator_casadi_config.yaml \
--num-steps 150 \
--action-noise-std 0.03
```

Collision-avoidance-focused multi-robot example with an explicit start configuration:

```bash
docker compose run --rm csvil \
python test/plot_expert_trajectories.py \
--system multi_robot \
--planner casadi \
--config test/config/multi_double_integrator_casadi_config.yaml \
--num-steps 150 \
--output-path outputs/plots/multi_robot_collision_focus.pdf \
--initial-states '[
  [[ 0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]]
]'
```

The plotter also supports the same execution-time action noise used in data
collection:

- `--action-noise-std <float>` (default: `0.0`)

This leaves deterministic expert plotting unchanged by default, but lets you
visually inspect how noisy execution bends the expert rollouts while the same
planner remains in control.

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

### Train a diffusion or ACT policy with LeRobot

Train a diffusion policy or ACT using the matching YAML configuration:

```bash
docker compose run --rm csvil \
python learning/train_lerobot.py \
--config learning/config/double_integrator_casadi_diffusion_policy_config.yaml
```

Multi-robot diffusion config:

```bash
docker compose run --rm csvil \
python learning/train_lerobot.py \
--config learning/config/multi_double_integrator_casadi_diffusion_policy_config.yaml
```

The training entrypoint now exposes `run_training(config_path)` for programmatic
use in sweep/automation scripts while preserving the same CLI behavior.

Training outputs and checkpoints are saved in the configured output directory.

### Train a custom PyTorch based MLP policy with DAgger

Train the custom MLP baseline with a standalone PyTorch DAgger loop.
It can start from an existing expert dataset or in fresh mode, rolls out the learner, queries the expert on visited states, and appends corrective labels to reduce distribution shift. Expert execution during aggregation can be mixed with the learner via `--expert-mix-beta-start` / `--expert-mix-beta-end`.

```bash
docker compose run --rm csvil \
python learning/train_dagger.py \
--system double_integrator \
--expert-config test/config/double_integrator_casadi_config.yaml \
--repo-id local/double_integrator_casadi_expert \
--dataset-root data/lerobot_dataset_double_integrator_casadi \
--dagger-iterations 1 \
--trajectories-per-iteration 100 \
--steps-per-trajectory 150 \
--target-epochs-per-round 100 \
--action-noise-std 0.0
```

Multi-robot MLP policy with DAgger training for 2 `double_integrator` systems:

```bash
docker compose run --rm csvil \
python learning/train_dagger.py \
--system multi_robot \
--expert-config test/config/multi_double_integrator_casadi_config.yaml \
--mlp-config learning/config/multi_double_integrator_casadi_mlp_config.yaml \
--repo-id local/multi_robot_casadi_expert \
--dataset-root data/lerobot_dataset_multi_robot_casadi \
--dagger-iterations 1 \
--trajectories-per-iteration 100 \
--steps-per-trajectory 150 \
--target-epochs-per-round 100 \
--action-noise-std 0.0
```

Fresh-start MLP DAgger example that skips a separate offline pretraining stage and only starts decaying expert mixing after evaluation success becomes nonzero:

```bash
docker compose run --rm csvil \
python learning/train_dagger.py \
--system double_integrator \
--expert-config test/config/double_integrator_casadi_config.yaml \
--dagger-iterations 5 \
--trajectories-per-iteration 20 \
--steps-per-trajectory 150 \
--target-epochs-per-round 10 \
--action-noise-std 0.0 \
--expert-mix-beta-start 0.5 \
--expert-mix-beta-decay-rate 0.1 \
--expert-mix-decay-after-success-rate 0.0
```

Fresh-start `unicycle2` MLP DAgger example:

```bash
docker compose run --rm csvil \
python learning/train_dagger.py \
--system unicycle2 \
--expert-config test/config/unicycle2_casadi_config.yaml \
--dagger-iterations 10 \
--trajectories-per-iteration 20 \
--steps-per-trajectory 200 \
--target-epochs-per-round 10 \
--action-noise-std 0.0 \
--expert-mix-beta-start 0.5 \
--expert-mix-beta-decay-rate 0.1 \
--expert-mix-decay-after-success-rate 0.0
```

Observed result for the command above:

- `Round 10 evaluation: eval_success_rate=100.0% eval_mean_steps=139.70 eval_min_steps=70 eval_max_steps=169 episodes=10`
- `Saved checkpoints: outputs/train_dagger/mlp_dagger_checkpoint.pt and outputs/train_dagger/mlp_dagger_iter_009.pt`

Single-trajectory evaluation command used to verify the checkpoint:

```bash
docker compose run --rm csvil \
python test/evaluate_policy.py \
--system unicycle2 \
--policy-type mlp \
--config test/config/unicycle2_casadi_config.yaml \
--model-dir outputs/train_dagger/mlp_dagger_checkpoint.pt \
--num-steps 200 \
--action-noise-std 0.0 \
--seeds "[0]" \
--initial-states '[
  [[-0.5, 0.5, 0.78, 0.0, 0.0]]
]'
```

Observed summary for the command above:

- `policy_successes: 1/1`
- `success_rate: 1.0000`
- `mean_policy_steps: 153.000`
- `mean_expert_steps: 131.000`
- `mean_policy_goal_error_l2: 0.034745`
- `mean_expert_goal_error_l2: 0.043021`

`--dagger-iterations` means the number of refinement rounds, where each
round does: aggregate learner rollouts with expert labels, then retrain.

- `--dagger-iterations 0`: pure offline training only (no aggregation)
- `--dagger-iterations 1`: one aggregate + retrain refinement
- `--expert-mix-beta-start` / `--expert-mix-beta-end`: control how often the expert action is executed during aggregation rollouts; set both to `0.0` to recover the old no-mixing behavior
- `--expert-mix-beta-decay-rate`: optional additive per-round schedule `beta_t = max(0, beta_start - rate * t)`; when set, this overrides `--expert-mix-beta-end`
- `--expert-mix-decay-after-success-rate`: optional gate that delays beta decay until evaluation success exceeds a threshold; set to `0.0` for a strict "start decaying only after success is nonzero" gate
- `--mlp-config`: optional YAML file for MLP architecture, e.g. `learning/config/multi_double_integrator_casadi_mlp_config.yaml` with default `model.hidden_dims: [512, 256, 128]`
- Aggregation logs progress every 10 episodes and reports `aggregation_success_rate` and `aggregation_mean_steps`.
- After each retrain, deterministic in-loop evaluation reports `eval_success_rate` and `eval_mean_steps`.
- Evaluation defaults to 10 seeded rollouts; tune with `--eval-episodes`, `--eval-steps`, `--eval-seed-start`, and `--eval-action-noise-std`.
- `--action-noise-std 0.0`: no noise added to action to perturb the states

### Train a diffusion or ACT policy with LeRobot and DAgger

LeRobot policies can also be used with DAgger training:

```bash
docker compose run --rm csvil \
python learning/train_lerobot_dagger.py \
--system double_integrator \
--expert-config test/config/double_integrator_casadi_config.yaml \
--lerobot-train-config learning/config/double_integrator_casadi_act_config.yaml \
--repo-id local/double_integrator_casadi_expert \
--dataset-root data/lerobot_dataset_double_integrator_casadi \
--dagger-iterations 1 \
--trajectories-per-iteration 100 \
--steps-per-trajectory 150 \
--target-epochs-per-round 100 \
--action-noise-std 0.0
```

Recommended incremental DAgger recipe for ACT/Diffusion: skip separate offline pretraining, reuse the current expert dataset, and let DAgger handle the first training pass and later refinements.

```bash
docker compose run --rm csvil \
python learning/train_lerobot_dagger.py \
--system double_integrator \
--expert-config test/config/double_integrator_casadi_config.yaml \
--lerobot-train-config learning/config/double_integrator_casadi_act_config.yaml \
--repo-id local/double_integrator_casadi_expert \
--dataset-root data/lerobot_dataset_double_integrator_casadi \
--dagger-iterations 5 \
--trajectories-per-iteration 20 \
--steps-per-trajectory 150 \
--target-epochs-per-round 10 \
--action-noise-std 0.0 \
--expert-mix-beta-start 0.6
```

or

```bash
docker compose run --rm csvil \
python learning/train_lerobot_dagger.py \
--system multi_robot \
--expert-config test/config/multi_double_integrator_casadi_config.yaml \
--lerobot-train-config learning/config/multi_double_integrator_casadi_act_config.yaml \
--repo-id local/multi_double_integrator_casadi_expert \
--dataset-root data/lerobot_dataset_multi_robot_casadi \
--dagger-iterations 1 \
--trajectories-per-iteration 100 \
--steps-per-trajectory 150 \
--target-epochs-per-round 100 \
--action-noise-std 0.0
```

How this loop works:

- Initial pass trains from scratch unless `--initial-pretrained-path` is provided.
- Offline ACT/Diffusion pretraining is optional; if you already have an expert dataset, you can skip a separate `train_lerobot.py` warm-start and let DAgger perform the first training pass directly.
- Round 0 uses the fixed `steps` value from `--lerobot-train-config` (for example `steps: 12000` for ACT).
- Refinement rounds (1+) aggregate expert labels, then retrain with `policy.pretrained_path` set to the latest checkpoint.
- Refinement retraining steps are sized with `--target-epochs-per-round`; `--max-train-steps` applies as a hard cap.
- Aggregation logs progress every 10 episodes and reports `aggregation_success_rate` / `aggregation_mean_steps`.
- After each retrain, deterministic in-loop evaluation reports `eval_success_rate` / `eval_mean_steps`.
- Evaluation defaults to 10 seeded rollouts; tune with `--eval-episodes`, `--eval-steps`, `--eval-seed-start`, and `--eval-action-noise-std`.
- `--expert-mix-beta-start` / `--expert-mix-beta-end`: control expert execution during aggregation rollouts; set both to `0.0` to disable mixing, or use a positive start value and decay to `0.0` for incremental DAgger.
- `--expert-mix-beta-decay-rate`: optional additive per-round schedule `beta_t = max(0, beta_start - rate * t)`; when set, this overrides `--expert-mix-beta-end`.
- `--expert-mix-decay-after-success-rate`: optional gate that delays beta decay until evaluation success exceeds a threshold; set to `0.0` if you want decay to remain off until eval success becomes nonzero.
- During aggregation, the learner action is executed in simulation while the dataset stores expert (CasADi) corrective labels.
- Dataset appends use `LeRobotDataset.resume(...)` and call `finalize()` each iteration to keep parquet chunks readable.

Optional multi-robot visibility gating can be set in the simulator config:

- `inter_robot_visibility_radius`: either one shared scalar radius, or a
  per-robot list of radii (broadcast-style API)

When another robot is outside the observing robot's visibility radius, its
relative x/y term is zeroed in the observation.

By default, push-to-hub is disabled in this iterative loop to avoid uploading
partial per-iteration checkpoints. Use `--allow-push-to-hub` to keep the
original training-config upload behavior.

LeRobot DAgger checkpoints are discovered under `--train-output-root`
(default `outputs/train`) using the LeRobot run layout
`<run>/checkpoints/last/pretrained_model`.

If a DAgger run crashes mid-write and later reports parquet footer errors,
recreate the dataset directory before restarting. The trainer now finalizes
LeRobot writer state per iteration to keep appended parquet chunks readable.

### Evaluate the learned policy

Evaluate a trained policy independently of the training loop.
`--policy-type` supports `diffusion`, `act`, and `mlp`.
For `diffusion`/`act`, use either a local `pretrained_model` checkpoint directory (for example `outputs/train/<run-name>/checkpoints/last/pretrained_model`) or a Hub model ID (for example `jovaldivieso/double_integrator_casadi_act`).
For `mlp`, use a local `.pt` checkpoint.

The evaluator also supports optional execution-time action noise for robustness
benchmarking:

- `--action-noise-std <float>` (default: `0.0`)

When enabled, the same Gaussian action perturbation rule is applied to both the
expert and policy rollouts before stepping the simulator, so clean evaluation
remains the default while disturbance benchmarking stays comparable.

Evaluation example for a `multi_robot` system with 2 double integrators and `diffusion` policy type for a specific start state not included in the dataset:

```bash
docker compose run --rm csvil \
python test/evaluate_policy.py \
--system multi_robot \
--policy-type diffusion \
--config <test-config.yaml> \
--model-dir <checkpoint-or-hub-id> \
--num-steps 150 \
--action-noise-std 0.03 \
--seeds "[0]" \
--initial-states '[
  [[ 0.0, 0.5, 0.0, 0.0], [0.0, -0.5, 0.0, 0.0]]
]'
```

More examples:

```bash
# ACT / Diffusion (local checkpoint dir)
docker compose run --rm csvil \
python test/evaluate_policy.py \
--system multi_robot \
--policy-type act \
--config test/config/multi_double_integrator_casadi_config.yaml \
--model-dir outputs/train/<run-name>/checkpoints/last/pretrained_model \
--seeds "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"

# ACT (Hub model ID)
docker compose run --rm csvil \
python test/evaluate_policy.py \
--system double_integrator \
--policy-type act \
--config test/config/double_integrator_casadi_config.yaml \
--model-dir jovaldivieso/double_integrator_casadi_act

# MLP (local .pt checkpoint)
docker compose run --rm csvil \
python test/evaluate_policy.py \
--system multi_robot \
--policy-type mlp \
--config test/config/multi_double_integrator_casadi_config.yaml \
--model-dir outputs/train_dagger_multi_robot/mlp_dagger_checkpoint.pt \
--seeds "[[42], [21]]"

# Robustness benchmark (same disturbance model for expert and policy)
docker compose run --rm csvil \
python test/evaluate_policy.py \
--system multi_robot \
--policy-type act \
--config test/config/multi_double_integrator_casadi_config.yaml \
--model-dir outputs/train/<run-name>/checkpoints/last/pretrained_model \
--seeds "[[0], [100]]" \
--action-noise-std 0.03
```

The script runs rollouts and saves trajectory plots.

### Multi-Robot Workflows

Multi-robot systems are first-class global systems: run with `--system multi_robot`
plus a multi-robot YAML config, and the simulator exposes unified LeRobot keys
(`observation.environment_state`, `observation.state`, `action`) with concatenated
dimensions across robots.

Seed format quick reference:

- Flat list (for independent repeated rollouts): `--seeds "[0, 1, 2, 3]"`
- Nested list (for paired per-robot seeds in one joint rollout): `--seeds "[[0, 1, 2, 3], [100, 101, 102, 103]]"`

### Quickstart

Use this compact sequence for most experiments:

1. Collect expert data: see `Generate a motion planning expert dataset`.
2. Train one policy family:
   - MLP + DAgger: `learning/train_dagger.py`
   - ACT/Diffusion: `learning/train_lerobot.py`
   - ACT/Diffusion + DAgger: `learning/train_lerobot_dagger.py`
3. Evaluate with `test/evaluate_policy.py` using the canonical template above.

For robust model selection, prefer multi-seed evaluation (for example 30 seeds)
and compare success rate plus terminal error/cost, not only visual trajectory quality.

The evaluation, plotting, and expert collection scripts also expose importable
execution functions (`run_evaluation`, `run_plotting`, `run_collection`) so they can be called from Python workflows without shelling out to CLI.

### CLI argument quick reference

Use this as a compact quick reference for current entrypoint flags.

- `test/collect_expert_data.py`
  - required workflow args: `--system`, `--planner`, `--config`
  - optional rollout args: `--num-traj`, `--num-steps`, `--action-noise-std`, `--initial-states`
  - optional dataset args: `--repo-id`, `--local-dir`
- `test/plot_expert_trajectories.py`
  - required workflow args: `--system`, `--planner`, `--config`
  - optional rollout args: `--seeds`, `--initial-states`, `--num-steps`, `--action-noise-std`, `--output-path`
- `test/evaluate_policy.py`
  - required workflow args: `--system`, `--policy-type`, `--config`, `--model-dir`
  - optional rollout args: `--num-steps`, `--seeds`, `--initial-states`, `--action-noise-std`, `--output-path`
- `learning/train_lerobot.py`
  - required args: `--config`
- `learning/train_dagger.py`
  - required args: `--system`, `--expert-config`
  - optional dataset args: `--repo-id`, `--dataset-root` (omit both for fresh DAgger mode without offline dataset pretraining)
  - optional DAgger args: `--planner`, `--dagger-iterations`, `--trajectories-per-iteration`, `--steps-per-trajectory`, `--action-noise-std`, `--expert-mix-beta-start`, `--expert-mix-beta-end`, `--expert-mix-beta-decay-rate`, `--expert-mix-decay-after-success-rate`
  - optional training/eval args: `--target-epochs-per-round`, `--eval-episodes`, `--eval-steps`, `--eval-seed-start`, `--eval-action-noise-std`, `--batch-size`, `--learning-rate`, `--checkpoint-dir`, `--seed`, `--max-train-steps`
- `learning/train_lerobot_dagger.py`
  - required args: `--system`, `--expert-config`, `--lerobot-train-config`
  - optional dataset args: `--repo-id`, `--dataset-root` (omit both for fresh DAgger mode)
  - optional DAgger args: `--planner`, `--policy-type`, `--dagger-iterations`, `--trajectories-per-iteration`, `--steps-per-trajectory`, `--action-noise-std`, `--expert-mix-beta-start`, `--expert-mix-beta-end`, `--expert-mix-beta-decay-rate`, `--expert-mix-decay-after-success-rate`
  - optional training/eval args: `--train-output-root`, `--initial-pretrained-path`, `--seed`, `--target-epochs-per-round`, `--eval-episodes`, `--eval-steps`, `--eval-seed-start`, `--eval-action-noise-std`, `--max-train-steps`, `--allow-push-to-hub`

## TODO / Roadmap / Brainstorming
- Open: Observation packing still uses fixed-size concatenation and currently overloads zeros for "robot at same position" vs "robot not observable" in visibility-gated relative slots; transition to variable-neighbor representations with explicit masks (for example Deep Sets / GNN-friendly schema).
- Open: Add permutation-invariant multi-agent policy models in `learning/models` (Deep Sets and/or GNN, optionally CNN for grid observations) and integrate them into `train_dagger.py` so policies can generalize across variable neighbor counts and neighbor-order permutations instead of relying on fixed concatenation-based MLP inputs.
- Open: Extend the current coordinate-wise Euclidean/$S^1$ planner residual abstraction to support intrinsic `SO(3)` and `SE(3)` residuals once 3D rigid-body systems and richer manifold-valued states become first-class targets.
- Open: Extend protocol-level simulator metadata beyond the already-promoted `is_euclidean` field (for example, plotting metadata and coordinate semantics) to remove remaining script-local heuristics.
- Open: Complete the current refactor toward Lie-group-first internal state handling. For systems with manifold-valued state, represent internal orientation/pose math with SO(2), SE(2), and their Lie algebras/operators wherever we control the code, and confine coordinate-chart conversions (`theta`, flattened pose vectors, concatenated dataset tensors) to explicit boundary adapters for CasADi, LeRobot, OMPL, YAML parsing, and other external APIs.
- Open: Add optional per-robot state-cost blocks (`Q_diag_per_robot`) with the same normalization rules as `R_weight_per_robot` for heterogeneous fleets.
- Open: Add structured benchmark suites that report success rate, terminal error, trajectory cost, safety-margin statistics, and solver wall-time across systems and policies.
- Open: Add repeatable experiment manifests (seed bundles, config snapshots, artifact indexing) for reproducible BC/DAgger/ACT/Diffusion comparisons.
- Open: Add stress tests for edge-case fleet layouts (high robot count, mixed dynamics, tight `d_safe`) to validate solver conditioning and feasibility behavior.
- Open: Plan the transition from 2D validation systems to 3D robotics workflows by introducing a 3D double-integrator baseline and 3D trajectory visualization/evaluation; separate `double_integrator_2d.py` and `double_integrator_3d.py` and removing current `double_integrator.py` (renamed to `double_integrator_2d.py`) and keep factory/config/test coverage consistent across both.
- Brainstorming: Evaluate a collision-checking abstraction for FCL support (for example, a base `collision_checker.py` analogous to `planning/planner.py`, plus an `fcl_collision_checker.py` implementation), then define how it composes with planners and system configs.
- Brainstorming: Add composable observer/noise models so training and evaluation can sweep partial observability and sensor corruption systematically (beyond current execution-time action-noise injection).
- Brainstorming: Build Lie-group-aware inductive bias into the simple MLP baselines, using invariance/equivariance that matches each system's symmetry. For example, SE(2)-equivariant policy structure is a natural target for unicycle systems such as `unicycle2`, instead of treating pose coordinates as generic concatenated scalars.
- Brainstorming: Extend the planner stack to support OMPL as an additional backend. The expected integration path is straightforward: add an OMPL planner implementation that inherits from `planning/planner.py` and register it through `PlannerFactory`.
- Brainstorming: Upgrade the CasADi planner formulation to be Lie-group-aware wherever practical. Even if the NLP still uses Euclidean decision variables, the model can move toward SO(2)/SE(2)-consistent residuals, compose/inverse operations, and, where useful, explicit manifold-style constraints for rotation representations rather than relying on raw angle subtraction alone.
- Brainstorming: Add a safety module similar to GLAS-style barrier-function shielding.

