# Controller Synthesis via Imitation Learning

This repository distils a computationally expensive motion planner into a faster neural policy for online control. It supports single robots and homogeneous fleets through the same fleet-first simulator interface, with CasADi as the current expert planner and optional db-LaCAM and future planner backends.

Simulator and planner protocols, factories, and validated configuration keep extensions local while catching schema and dimension errors at their boundaries. The primary workflow is decentralized multi-robot DAgger with masked neighbor-observation encoders.

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
│   ├── lerobot_dataset_double_integrator_casadi/  # Example generated dataset
│   └── lerobot_dataset_multi_robot_casadi/        # Example generated multi-robot dataset
├── learning/
│   ├── config_loaders.py     # YAML and policy/encoder configuration helpers
│   ├── config/
│   │   └── multi_unicycle2_casadi_mlp_config.yaml
│   ├── models/
│   │   ├── deepset_encoder.py  # Permutation-invariant neighbor-set encoder
│   │   ├── flow_policy.py      # Conditional flow-matching action policy used for BC/DAgger
│   │   ├── encoder.py          # Shared interface and factory
│   │   ├── mlp_policy.py       # MLP action policy
│   │   └── policy.py           # Shared interface and factory
│   ├── data_utils.py           # Stateless policy batch formatting and action chunks
│   ├── dagger/
│   │   ├── __init__.py        # Public DAgger helper API
│   │   ├── beta_controller.py # Expert-mixing schedules and adaptive controller
│   │   ├── feature_cache.py   # Observation schema and feature packing utilities
│   │   ├── metrics.py         # DAgger evaluation metrics
│   │   ├── rollouts.py        # Collection, evaluation, and action execution
│   │   └── utils.py           # Seeding, step resolution, and metric logging
│   ├── train_dagger.py        # Object-oriented MLP / flow DAgger trainer and CLI
├── planning/
│   ├── planner.py             # Planner protocol and base class
│   ├── casadi_planner.py      # CasADi planner implementation
│   └── dblacam_planner.py     # db-LaCAM planner implementation
├── systems/
│   ├── dynamics.py            # Base simulator protocol and validation
│   ├── double_integrator.py   # Example simulator subclass (holonomic)
│   ├── initial_state_utils.py # Shared initial-state parsing/normalization
│   ├── multi_robot.py         # Fleet composition wrapper over per-robot simulators
│   ├── unicycle2.py           # Example simulator subclass (non-holonomic)
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
    ├── evaluate_policy.py        # CLI for rollout/evaluation across policy families
    ├── plot_expert_trajectories.py  # Canonical single/multi-robot expert analysis CLI (plots + optional MP4)
    └── test_simulator_contracts.py  # Schema consistency tests
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

Unicycle2 uses explicit per-dimension tolerances: `pos_tol`, `theta_tol`, `vel_tol`, and `omega_tol`.
`error_tolerance` is intentionally rejected for `unicycle2` configs.

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
Canonical multi-robot `unicycle2` + MLP DAgger example:

- `test/config/multi_unicycle2_casadi_config.yaml`
  (2x homogeneous `unicycle2` robots with shared MPC planner settings and global `d_safe`)
- `learning/config/multi_unicycle2_casadi_mlp_config.yaml`

## Pipeline Tutorial

The primary workflow is decentralized DAgger for a homogeneous multi-robot
`unicycle2` fleet. Each robot runs the shared policy from its ego observation
and masked neighbor observations; actions are combined only when stepping the
simulator. Fresh-start DAgger collects its own expert-labelled rollouts, so no
offline dataset is required.

### MLP DAgger

Train the canonical MLP policy:

```bash
python learning/train_dagger.py \
--system multi_robot \
--expert-config test/config/multi_unicycle2_casadi_config.yaml \
--policy-config learning/config/multi_unicycle2_casadi_mlp_config.yaml \
--dagger-iterations 5 \
--trajectories-per-iteration 50 \
--steps-per-trajectory 200 \
--target-epochs-per-round 10 \
--action-noise-std 0.03 \
--expert-mix-beta-start 0.5 \
--expert-mix-beta-decay-rate 0.25 \
--expert-mix-decay-after-eval-success 0.0 \
--eval-episodes 100
```

Evaluate its checkpoint:

```bash
python test/evaluate_policy.py \
--system multi_robot \
--policy-type mlp \
--config test/config/eval_multi_unicycle2_casadi_config.yaml \
--model-dir outputs/train_dagger_multi_robot/mlp_dagger_checkpoint.pt \
--num-steps 200 \
--action-noise-std 0.03 \
--seeds 4
```

### Flow DAgger

Train the flow-matching policy with the same rollout schedule:

```bash
python learning/train_dagger.py \
--system multi_robot \
--expert-config test/config/multi_unicycle2_casadi_config.yaml \
--policy-config learning/config/multi_unicycle2_casadi_flow_config.yaml \
--dagger-iterations 5 \
--trajectories-per-iteration 50 \
--steps-per-trajectory 200 \
--target-epochs-per-round 10 \
--action-noise-std 0.03 \
--expert-mix-beta-start 0.5 \
--expert-mix-beta-decay-rate 0.25 \
--expert-mix-decay-after-eval-success 0.0 \
--eval-episodes 100
```

Evaluate its checkpoint:

```bash
python test/evaluate_policy.py \
--system multi_robot \
--policy-type flow \
--config test/config/eval_multi_unicycle2_casadi_config.yaml \
--model-dir outputs/train_dagger_multi_robot/flow_dagger_checkpoint.pt \
--num-steps 200 \
--action-noise-std 0.03 \
--seeds 4
```

For Docker, prefix any command above with `docker compose run --rm csvil`.

The MLP and flow configurations use the same permutation-invariant DeepSet
neighbor encoder. `--dagger-iterations` is the number of aggregate-and-retrain
rounds. The expert-mixing flags begin rollouts at 50% expert actions and decay
that fraction by 0.25 per round after evaluation records nonzero success.

`--dagger-iterations` means the number of refinement rounds, where each
round does: aggregate learner rollouts with expert labels, then retrain.

- `--dagger-iterations 0`: pure offline training only (no aggregation)
- `--dagger-iterations 1`: one aggregate + retrain refinement
- `--expert-mix-beta-start` / `--expert-mix-beta-end`: control how often the expert action is executed during aggregation rollouts; set both to `0.0` to recover the old no-mixing behavior
- `--expert-mix-beta-decay-rate`: optional additive per-round schedule `beta_t = max(0, beta_start - rate * t)`; when set, this overrides `--expert-mix-beta-end`
- `--expert-mix-decay-after-eval-success`: optional gate that delays beta decay until evaluation success exceeds a threshold; set to `0.0` for a strict "start decaying only after success is nonzero" gate
- `--config-goal-after-eval-success`: optional threshold that starts training on randomized goals and switches to the config goal once evaluation success reaches the configured percentage
- `--adaptive-beta-recovery`: optional (default `true`); when enabled, beta increases by one schedule step after an eval-success regression; when disabled, beta follows monotonic decay
- `--policy-config`: optional YAML file for the policy architecture (MLP or flow), e.g. `learning/config/multi_double_integrator_casadi_mlp_config.yaml` with default `model.hidden_dims: [256, 256, 128]`, or `learning/config/multi_double_integrator_casadi_flow_config.yaml` with `model.policy_type: flow`
- Aggregation logs progress every 10 episodes and reports `aggregation_success_rate` and `aggregation_mean_steps`.
- After each retrain, deterministic in-loop evaluation reports `eval_success_rate` and `eval_mean_steps`.
- Evaluation defaults to 10 seeded rollouts; tune with `--eval-episodes`, `--eval-steps`, `--eval-seed-start`, and `--eval-action-noise-std`.
- `--action-noise-std 0.0`: no noise added to action to perturb the states

For expert-only offline training, use `--dagger-iterations 0` with a
precollected expert dataset:

- `learning/train_dagger.py` for the custom MLP or flow baselines (`model.policy_type` in `--policy-config`)

To keep DAgger aggregation expert-only, set both
`--expert-mix-beta-start` and `--expert-mix-beta-end` to `1.0`.

### Evaluate the learned policy

Evaluate a trained policy independently of the training loop.
`--policy-type` supports only `mlp` and `flow`.
Use a local metadata `.pt` checkpoint produced by `learning/train_dagger.py`.

The evaluator also supports optional execution-time action noise for robustness
benchmarking:

- `--action-noise-std <float>` (default: `0.0`)

When enabled, the same Gaussian action perturbation rule is applied to both the
expert and policy rollouts before stepping the simulator, so clean evaluation
remains the default while disturbance benchmarking stays comparable.

Use the MLP and flow evaluation commands in the tutorial above as the standard
multi-robot benchmark. Replace `--policy-type` and `--model-dir` with the
relevant MLP or Flow checkpoint; the evaluator saves trajectory plots for each
rollout.

### Multi-Robot Workflows

Multi-robot systems are first-class global systems: run with `--system multi_robot`
plus a multi-robot YAML config. The simulator exposes the unified observation
keys (`observation.environment_state`, `observation.state`,
`observation.neighbor_state`, `observation.neighbor_mask`) and per-robot action
frames used by the custom MLP and Flow policies.

Seed format quick reference:

- Flat list (for independent repeated rollouts): `--seeds "[0, 1, 2, 3]"`
- Nested list (for paired per-robot seeds in one joint rollout): `--seeds "[[0, 1, 2, 3], [100, 101, 102, 103]]"`

### Quickstart

Use this compact sequence for most experiments:

1. Train one policy with `learning/train_dagger.py`.
2. Evaluate with `test/evaluate_policy.py` using the canonical template above.

For robust model selection, prefer multi-seed evaluation (for example 30 seeds)
and compare success rate plus terminal error/cost, not only visual trajectory quality.

The evaluation and plotting scripts also expose importable execution functions
so they can be called from Python workflows without shelling out to the CLI.

### CLI argument quick reference

Use this as a compact quick reference for current entrypoint flags.

- `test/plot_expert_trajectories.py`
  - required workflow args: `--system`, `--planner`, `--config`
  - optional rollout args: `--seeds`, `--initial-states`, `--num-steps`, `--action-noise-std`, `--output-path`
- `test/evaluate_policy.py`
  - required workflow args: `--system`, `--policy-type`, `--config`, `--model-dir`
  - optional rollout args: `--num-steps`, `--seeds`, `--initial-states`, `--action-noise-std`, `--output-path`
- `learning/train_dagger.py`
  - required args: `--system`, `--expert-config`
  - optional dataset args: `--repo-id`, `--dataset-root` (omit both for fresh DAgger mode without offline dataset pretraining)
  - optional DAgger args: `--planner`, `--dagger-iterations`, `--trajectories-per-iteration`, `--steps-per-trajectory`, `--action-noise-std`, `--expert-mix-beta-start`, `--expert-mix-beta-end`, `--expert-mix-beta-decay-rate`, `--expert-mix-decay-after-eval-success`, `--config-goal-after-eval-success`, `--adaptive-beta-recovery`
  - optional training/eval args: `--target-epochs-per-round`, `--eval-episodes`, `--eval-steps`, `--eval-seed-start`, `--eval-action-noise-std`, `--batch-size`, `--learning-rate`, `--checkpoint-dir`, `--seed`, `--max-train-steps`
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

### Optional: Visualize expert trajectories

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

### Optional: Visualize the dataset
Once the dataset is generated, you can use LeRobot's native CLI tool to launch a local web visualizer and inspect the expert trajectories:

```bash
docker compose run --rm csvil \
lerobot-dataset-viz \
--repo-id local/double_integrator_casadi_expert \
--root data/lerobot_dataset_double_integrator_casadi \
--mode local \
--episode-index 0
```


## TODO / Roadmap / Brainstorming
- Open: Extend protocol-level simulator metadata beyond the already-promoted `is_euclidean` field (for example, plotting metadata and coordinate semantics) to remove remaining script-local heuristics.
- Open: Add structured benchmark suites that report success rate, terminal error, trajectory cost, safety-margin statistics, and solver wall-time across systems and policies.
- Open: Add repeatable experiment manifests (seed bundles, config snapshots, artifact indexing) for reproducible BC/DAgger comparisons.
- Open: Add stress tests for edge-case fleet layouts (high robot count, mixed dynamics, tight `d_safe`) to validate solver conditioning and feasibility behavior.
- Open: Plan the transition from 2D validation systems to 3D robotics workflows by introducing a 3D double-integrator baseline and 3D trajectory visualization/evaluation; separate `double_integrator_2d.py` and `double_integrator_3d.py` and removing current `double_integrator.py` (renamed to `double_integrator_2d.py`) and keep factory/config/test coverage consistent across both.
- Brainstorming: Evaluate a collision-checking abstraction for FCL support (for example, a base `collision_checker.py` analogous to `planning/planner.py`, plus an `fcl_collision_checker.py` implementation), then define how it composes with planners and system configs.
- Brainstorming: Add composable observer/noise models so training and evaluation can sweep partial observability and sensor corruption systematically (beyond current execution-time action-noise injection).
- Brainstorming: Extend the planner stack to support OMPL as an additional backend. The expected integration path is straightforward: add an OMPL planner implementation that inherits from `planning/planner.py` and register it through `PlannerFactory`.
- Brainstorming: Add a safety module similar to GLAS-style barrier-function shielding.