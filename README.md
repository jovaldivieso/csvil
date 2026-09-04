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
├── learning/
│   ├── config_loaders.py     # YAML and policy/encoder configuration helpers
│   ├── config/
│   │   ├── default_policy_config.yaml           # Used when --policy-config is omitted (MLP, no DAgger schedule)
│   │   ├── multi_unicycle2_casadi_flow_config.yaml
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
│   │   └── utils.py           # Seeding, step resolution, config overrides, and metric logging
│   ├── train_dagger.py        # Object-oriented MLP / flow DAgger trainer and CLI
├── planning/
│   ├── planner.py             # Planner protocol and base class
│   ├── casadi_planner.py      # CasADi planner implementation
│   └── dblacam_planner.py     # db-LaCAM planner implementation
├── systems/
│   ├── dynamics.py            # Base simulator protocol and validation
│   ├── double_integrator.py   # Example simulator subclass (holonomic)
│   ├── initial_state_utils.py # Shared initial/goal-state parsing and normalization
│   ├── multi_robot.py         # Fleet composition wrapper over per-robot simulators
│   ├── unicycle2.py           # Example simulator subclass (non-holonomic)
│   └── seed_utils.py          # Seed defaults and deterministic rollout seeding
├── outputs/
│   ├── plots/                 # Evaluation and expert-rollout plots/videos
│   ├── train/                 # LeRobot training outputs
│   ├── train_dagger/          # Single-robot DAgger checkpoints (MLP or flow)
│   └── train_dagger_multi_robot/  # Multi-robot DAgger checkpoints (MLP or flow)
└── test/
    ├── config/
    │   ├── multi_unicycle2_casadi_config.yaml       # Canonical example (used throughout this README)
    │   ├── multi_double_integrator_casadi_config.yaml
    │   └── multi_robot_dblacam_config.yaml          # Long-form robots: list example (distinct per-robot `start` states)
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

### Expert configs (`test/config/`)

Each planner and corresponding dynamics use a YAML configuration file in
`test/config/`. These "expert configs" hold only solver/dynamics
parameters — time step, MPC horizon, cost weights, collision radii, and
system-specific limits (e.g. `max_accel`, `max_omega`). They intentionally do
**not** define a goal, random initial-state sampling bounds, or convergence
tolerances any more: those are experiment-specific and now live in the policy
config's `training:` block (see below), so the same expert config is reused,
unmodified, across every experiment run against that system.

Multi-robot fleets are described under a `robots:` key. For a homogeneous fleet
(all robots share the same system and config), use the shorthand:

```yaml
robots:
  num_robots: 2
  system: unicycle2
  config:
    dt: 0.05
    max_accel: 2.0
    max_omega: 2.0
    max_speed: 2.0
```

This expands into `num_robots` identical entries before validation, so nothing
downstream needs to know which form was used. Use the long list form —
`robots: [{system: ..., config: {...}}, ...]`, one entry per robot — when
otherwise homogeneous robots need distinct per-robot configurations or `start`
states. Mixed simulator types are not currently supported because
`MultiRobotSimulator` requires strictly homogeneous fleets.

System configs are validated through `core/config.py` before simulation/evaluation
to catch malformed keys and shape mismatches early. Planner-specific keys include:

- `horizon`
- `mode` (`mpc` or `open_loop`)
- `Q_diag` (must match state dimension `nx`)
- `R_weight` (scalar fallback)
- `R_diag` (optional full action-space diagonal, length `nu`)
- `R_weight_per_robot` (optional `multi_robot` override; one entry per robot, each scalar or per-action list)
- `terminal_cost_multiplier`
- `collision_slack_penalty_weight` (positive scalar penalty for collision slack in soft pairwise avoidance)

Every system still *accepts* `goal`, `randomize_goal`, initial-position sampling
bounds (`initial_position_radius_bounds`, `initial_position_min_goal_distance`),
and its own convergence tolerances (`error_tolerance`, or for `unicycle2`:
`pos_tol`, `theta_tol`, `vel_tol`, `omega_tol`; `error_tolerance` is rejected for
`unicycle2`) directly in the expert config, defaulting sensibly when omitted —
but the recommended pattern is to leave them out and set them per-experiment
instead, in the policy config's `training:` block, described next.

Simulator and planner creation is centralized through `DynamicsFactory` and
`PlannerFactory` in `core/factory.py`.

### Training and curriculum config (`learning/config/`)

`learning/train_dagger.py`'s `--policy-config` (default:
`learning/config/default_policy_config.yaml`, a minimal MLP with no DAgger
schedule) points at a YAML file with a `model:` block (architecture) and an
optional `training:` block. Everything experiment-specific — the DAgger
schedule, expert-mixing, evaluation cadence, and the goal/initial-state
curriculum and sampling/tolerance tuning — lives here instead of on the command
line, so one file plus `--experiment-name` reproduces a full run:

```yaml
training:
  dagger_iterations: 2
  trajectories_per_iteration: [100, 100]
  steps_per_trajectory: 200
  target_epochs_per_round: [50, 50]
  action_noise_std: 0.03
  expert_mix_beta_start: 0.5
  expert_mix_beta_decay_rate: 0.25
  eval_episodes: 100

  # Optional: override the expert config's random initial-state sampling bounds
  # and convergence tolerances for this experiment. Applies to both DAgger data
  # collection every round and in-loop evaluation; never to the supervised
  # training step itself, which has no simulator in the loop.
  initial_position_min_goal_distance: 0.05
  initial_position_radius_bounds: [0.05, 3.0]
  tolerance_overrides:
    pos_tol: 0.2
    theta_tol: 1.1
    vel_tol: 0.1
    omega_tol: 0.1

  # Round-based goal curriculum: one 'random' or 'config' entry per DAgger round.
  # 'random' rounds sample random goals/states; 'config' rounds draw from the
  # initial_states/goal_states pairs below, index-for-index, falling back to
  # random for the rest of that round once either list is exhausted.
  training_curriculum: [random, config]
  initial_states:
    - [[-2.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 3.14, 0.0, 0.0]]
    - [[0.0, -2.0, 1.57, 0.0, 0.0], [0.0, 2.0, -1.57, 0.0, 0.0]]
  goal_states:
    - [[2.0, 0.0, 0.0], [-2.0, 0.0, 3.14]]
    - [[0.0, 2.0, 1.57], [0.0, -2.0, -1.57]]
```

This is the reproducible way to target hard cases such as head-on collision
courses: define the initial/goal state pairs once, commit the file, and both
training and evaluation read the same scenarios, instead of hand-tuning a
one-off `--initial-states` JSON string on the command line every time.

Every `training:` key has a matching `--flag` on `train_dagger.py` that
overrides the YAML value for a single run (see the CLI quick reference below).
`--initial-states`/`--goal-states`/`--tolerance-overrides` are also available on
`evaluate_policy.py` and `plot_expert_trajectories.py`, for ad hoc inspection of
a specific scenario without touching any config file.

## Pipeline Tutorial

The primary workflow is decentralized DAgger for a homogeneous multi-robot
`unicycle2` fleet. Each robot runs the shared policy from its ego observation
and masked neighbor observations; actions are combined only when stepping the
simulator. Fresh-start DAgger collects its own expert-labelled rollouts, so no
offline dataset is required.

### Train

```bash
python learning/train_dagger.py \
--system multi_robot \
--expert-config test/config/multi_unicycle2_casadi_config.yaml \
--policy-config learning/config/multi_unicycle2_casadi_flow_config.yaml \
--experiment-name 2_unicycle2_casadi_deepset_flow
```

That's the whole command — the DAgger schedule, expert-mixing, curriculum, and
evaluation cadence all come from `multi_unicycle2_casadi_flow_config.yaml`'s
`training:` block (see above). To train the MLP baseline instead, swap
`--policy-config` for `learning/config/multi_unicycle2_casadi_mlp_config.yaml`;
the command is otherwise identical, since both policies share the same
DeepSet neighbor encoder and DAgger loop.

Checkpoints land under `--checkpoint-dir` (default
`outputs/train_dagger_multi_robot/<experiment-name>/`) as
`<policy_type>_dagger_checkpoint.pt` (latest) and one
`<policy_type>_dagger_iter_NNN.pt` per completed round.

### Evaluate

```bash
python test/evaluate_policy.py \
  --system multi_robot \
  --policy-type flow \
  --config test/config/multi_unicycle2_casadi_config.yaml \
  --model-dir outputs/train_dagger_multi_robot/2_unicycle2_casadi_deepset_flow/flow_dagger_iter_001.pt \
  --num-steps 200 \
  --action-noise-std 0.03 \
  --initial-states '[[[-2.1, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 3.14, 0.0, 0.0]]]' \
  --goal-states '[[[2.0, 0.0, 0.0], [-2.0, 0.0, 3.14]]]' \
  --tolerance-overrides '{"pos_tol": 0.2, "theta_tol": 1.1, "vel_tol": 0.05, "omega_tol": 0.05}'
```

`--policy-type` supports `mlp` and `flow`. `--initial-states`/`--goal-states` are
optional (omit them for randomly seeded rollouts); pass them to check
performance on a specific scenario, e.g. the same swap/crossing cases used
during training. `--tolerance-overrides` reproduces whatever convergence
tolerances the policy was trained under, since the expert config no longer
bakes in a fixed value. The evaluator also applies the same execution-time
Gaussian action noise (`--action-noise-std`, default `0.0`) to both the expert
and policy rollouts, so clean evaluation stays the default while disturbance
benchmarking remains comparable. For robust model selection, prefer evaluating
many seeds (`--seeds "[0, 1, ..., 29]"`) and comparing success rate plus
terminal error, not only visual trajectory quality.

### Generalizing to a Different Fleet Size

Because the decentralized policy only ever sees an ego observation plus
masked neighbor slots, a checkpoint trained on a 2-robot fleet can be
evaluated directly on a larger fleet by pointing `--config` at a config with
a different `robots.num_robots` — no retraining required:

```bash
python test/evaluate_policy.py \
  --system multi_robot \
  --policy-type flow \
  --config test/config/eval_multi_unicycle2_casadi_config.yaml \
  --model-dir outputs/train_dagger_multi_robot/2_unicycle2_casadi_deepset_flow/flow_dagger_checkpoint.pt \
  --num-steps 200 \
  --action-noise-std 0.03 \
  --initial-states '[[[-2.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 3.14, 0.0, 0.0], [0.0, -2.0, 1.57, 0.0, 0.0], [0.0, 2.0, -1.57, 0.0, 0.0]]]' \
  --goal-states '[[[2.0, 0.0, 0.0], [-2.0, 0.0, 3.14], [0.0, 2.0, 1.57], [0.0, -2.0, -1.57]]]' \
  --tolerance-overrides '{"pos_tol": 0.2, "theta_tol": 1.1, "vel_tol": 0.5, "omega_tol": 0.5}'
```

Here `eval_multi_unicycle2_casadi_config.yaml` sets `robots.num_robots: 4` (a
4-way rotational swap), while the checkpoint was trained on 2 robots — the
encoder's per-neighbor schema and observation horizon must still match, but
`neighbor_slots` adapts automatically to the runtime fleet size.

For Docker, prefix either command with `docker compose run --rm csvil`.

A few flags worth knowing about beyond what's in the policy config:

- `--dagger-iterations` (CLI) / `dagger_iterations` (YAML): number of aggregate-and-retrain rounds. `0` means pure offline training on a precollected dataset only (requires `--repo-id`/`--dataset-root`); omitting both instead starts fresh-start DAgger, collecting its own rollouts from round 0.
- `--expert-mix-beta-start` / `--expert-mix-beta-end`: how often the expert action executes during aggregation rollouts; set both to `1.0` to keep aggregation strictly expert-only, or both to `0.0` to recover the old no-mixing behavior.
- `--expert-mix-beta-decay-rate`: optional per-round schedule `beta_t = max(0, beta_start - rate * t)`; overrides `--expert-mix-beta-end` when set.
- `--expert-mix-decay-after-eval-success`: optional gate that delays beta decay until evaluation success exceeds a threshold.
- `--adaptive-beta-recovery` (default `false`): beta increases one schedule step after an eval-success regression; enable for adaptive recovery, or leave disabled for monotonic decay.
- Aggregation logs progress every 10 episodes and reports `aggregation_success_rate`/`aggregation_mean_steps`; each retrain's in-loop evaluation reports `eval_success_rate`/`eval_mean_steps`. Evaluation defaults to 10 seeded rollouts; tune with `--eval-episodes`, `--eval-steps`, `--eval-seed-start`, `--eval-action-noise-std`.

### Multi-Robot Workflows

Multi-robot systems are first-class global systems: run with `--system multi_robot`
plus a multi-robot YAML config (see the homogeneous-fleet shorthand above). The
simulator exposes the unified observation keys (`observation.environment_state`,
`observation.state`, `observation.neighbor_state`, `observation.neighbor_mask`)
and per-robot action frames used by the custom MLP and Flow policies.

Optional multi-robot visibility gating can be set in the simulator config:

- `inter_robot_visibility_radius`: either one shared scalar radius, or a
  per-robot list of radii (broadcast-style API)

When another robot is outside the observing robot's visibility radius, its
relative-pose features (position and periodic heading terms) are zeroed in the observation.

Seed format quick reference:

- Flat list (for independent repeated rollouts): `--seeds "[0, 1, 2, 3]"`
- Nested list (for paired per-robot seeds in one joint rollout): `--seeds "[[0, 1, 2, 3], [100, 101, 102, 103]]"`

State/goal-spec format quick reference (`--initial-states`, `--goal-states`, and
the matching `training:` YAML lists) — same shape rules for both:

- One global rollout vector: `'[0.5, -0.1, 0.0, 0.0]'`
- Multiple global rollouts: `'[[0.5, -0.1, 0.0, 0.0], [0.2, 0.3, 0.0, 0.0]]'`
- Multi-robot rollouts, nested per robot: `'[[[robot1...], [robot2...]], ...]'`

Initial-state vectors use the system's full state dimension (`nx`); goal vectors
use whatever dimension that system's own `goal` uses (e.g. 3 for `unicycle2`:
`x, y, theta`).

By default, push-to-hub is disabled in this iterative loop to avoid uploading
partial per-iteration checkpoints. Use `--allow-push-to-hub` to keep the
original training-config upload behavior.

LeRobot DAgger checkpoints are discovered under `--train-output-root`
(default `outputs/train`) using the LeRobot run layout
`<run>/checkpoints/last/pretrained_model`.

If a DAgger run crashes mid-write and later reports parquet footer errors,
recreate the dataset directory before restarting. The trainer now finalizes
LeRobot writer state per iteration to keep appended parquet chunks readable.

### CLI argument quick reference

Use this as a compact quick reference for current entrypoint flags.

- `test/plot_expert_trajectories.py`
  - required workflow args: `--system`, `--planner`, `--config`
  - optional rollout args: `--seeds`, `--initial-states`, `--goal-states`, `--tolerance-overrides`, `--num-traj`, `--num-steps`, `--use-config-start`, `--action-noise-std`, `--video`/`--no-video`, `--video-fps`, `--output-path`
- `test/evaluate_policy.py`
  - required workflow args: `--system`, `--policy-type`, `--config`, `--model-dir`
  - optional rollout args: `--num-steps`, `--seeds`, `--initial-states`, `--goal-states`, `--tolerance-overrides`, `--action-noise-std`, `--output-path`
- `learning/train_dagger.py`
  - required args: `--experiment-name`, `--system`, `--expert-config`
  - optional dataset args: `--repo-id`, `--dataset-root` (omit both for fresh DAgger mode without offline dataset pretraining)
  - optional DAgger args: `--planner`, `--dagger-iterations`, `--trajectories-per-iteration`, `--steps-per-trajectory`, `--action-noise-std`, `--training-curriculum`, `--initial-states`, `--goal-states`, `--initial-position-min-goal-distance`, `--initial-position-radius-bounds`, `--tolerance-overrides`, `--expert-mix-beta-start`, `--expert-mix-beta-end`, `--expert-mix-beta-decay-rate`, `--expert-mix-decay-after-eval-success`, `--adaptive-beta-recovery`/`--no-adaptive-beta-recovery`
  - optional training/eval args: `--target-epochs-per-round`, `--eval-episodes`, `--eval-steps`, `--eval-seed-start`, `--eval-action-noise-std`, `--batch-size`, `--learning-rate`, `--policy-config`, `--checkpoint-dir`, `--seed`, `--max-train-steps`

Every flag above also accepts `--help` for its full description, e.g.
`python learning/train_dagger.py --help`.

### Optional: Visualize expert trajectories

Create a PDF plot and animation of trajectories generated by the CasADi expert,
using the same canonical fleet as training:

```bash
docker compose run --rm csvil \
python test/plot_expert_trajectories.py \
--system multi_robot \
--planner casadi \
--config test/config/multi_unicycle2_casadi_config.yaml \
--num-steps 200 \
--action-noise-std 0.03
```

A single-robot system uses the same command with a single-system config, e.g.
`--system unicycle2 --config test/config/unicycle2_casadi_config.yaml`.

To inspect a specific scenario (e.g. a head-on collision course) rather than
random rollouts, pass `--initial-states`/`--goal-states` the same way the
`training:` config does — this is the same mechanism used to define hard cases
for DAgger training, so a scenario you like here can be copied straight into a
policy config's `initial_states`/`goal_states` lists for reproducible training:

```bash
docker compose run --rm csvil \
python test/plot_expert_trajectories.py \
--system multi_robot \
--planner casadi \
--config test/config/multi_unicycle2_casadi_config.yaml \
--num-steps 200 \
--initial-states '[[[-2.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 3.14, 0.0, 0.0]]]' \
--goal-states '[[[2.0, 0.0, 0.0], [-2.0, 0.0, 3.14]]]' \
--output-path outputs/plots/multi_robot_head_on.pdf
```

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
