# Agent context

Orientation for a coding agent picking up this repository. It complements, and does
not repeat:

- [README.md](../README.md) — installation and the general pipeline.
- [evaluation.md](evaluation.md) — how to train, evaluate and plot both studies.
- [experiment_plan.md](experiment_plan.md) — what the studies ask, their design,
  decisions and results.
- [study2_encoders.md](study2_encoders.md) — study 2 in full: training setup, evaluation
  scenarios, limits.

Read those for the "what" and "how to run". This file is the "where things are, what
must stay true, and what has already bitten us".

## The project

Decentralized DAgger imitation learning for homogeneous `unicycle2` robot fleets. A
CasADi MPC expert (`planning/casadi_planner.py`) labels states; a shared per-robot
policy learns from ego observations plus masked neighbour observations. Policy heads:
`mlp`, `flow` (flow matching), `safeflow`. Neighbour encoders: `deepset`,
`transformer`, `gnn`. Trained policies run at any fleet size.

## Repository map

| Path | Role |
|---|---|
| `core/config.py` | Config schema and validation. **Rejects unknown keys.** |
| `systems/unicycle2.py`, `systems/multi_robot.py` | Dynamics, sampling, observations, collision checks |
| `planning/casadi_planner.py` | The MPC expert |
| `learning/train_dagger.py` | Training CLI; merges CLI flags with the policy config's `training:` section |
| `learning/config_loaders.py` | `DEFAULT_DAGGER_TRAINING_CONFIG` — every key the `training:` section accepts |
| `learning/dagger/dagger_config.py`, `dagger_trainer.py`, `rollouts.py` | Validation, the DAgger loop, collection and eval |
| `test/config/multi_unicycle2_casadi_config.yaml` | Template every study scenario config is derived from |
| `test/config/generate_fleet_configs.py` | → `test/config/study/unicycle2_fleet_NN.yaml` (the training/expert scenario) |
| `test/config/generate_circle_configs.py` | → `test/config/study/circle/unicycle2_circle_NN.yaml` (antipodal ring) |
| `test/config/generate_eval_configs.py` | → `test/config/study/arena/`, `fleet/`, `density/` (the three evaluation axes) |
| `test/plot_scenarios.py` | Scenario layouts (starts, goals, d_collision, visibility) as PNG |
| `test/make_mock_checkpoints.py` | Untrained checkpoints, for testing the evaluation pipeline without training |
| `learning/config/study/generate_study_policy_configs.py` | → study 2 policy configs (see below) |
| `train.sh` | Parallel Docker training: `train.sh <experiment> <policy_dir> <expert_dir>`, every policy × expert × seed |
| `eval.sh` | Parallel Docker evaluation: `eval.sh <experiment> [random|density|circle|all]` |
| `eval_study1.sh` | Study 1 evaluation, runs locally without Docker |
| `test/evaluate_scaling.py` | Policy-only rollouts at many fleet sizes → one CSV row per checkpoint × fleet size |
| `test/evaluate_policy.py` | Expert **and** policy from the same start → PDF + MP4. Practical only up to ~8 robots |
| `test/plot_study_results.py`, `test/plot_study1_results.py` | Study 2 (`--axis fleet|density|auto`) / study 1 figures |
| `outputs/`, `data/`, `logs/` | Gitignored: models, eval CSVs, plots, LeRobot datasets, run logs |

Generated study 2 policy configs: `learning/config/study/n{02,04,06,08}/<encoder>_{mlp,flow}.yaml`,
from the templates `learning/config/study/<encoder>_{mlp,flow}_config.yaml`. Schedule:
`training_schedule()`, the same for every fleet size. Each config carries its full DAgger
schedule and per-robot ring layouts in its `training:` section, so a `nNN/` directory
only fits expert configs with NN robots. The generator deletes and rewrites each `nNN/`
directory, because `train.sh` trains every YAML it finds there.

## Current state

- Branch `study-encoder`. Work now focuses on study 2 (encoders). Training goes through
  `train.sh`, evaluation through `eval.sh`; `run_study.sh` and the study 1 retrain configs
  are gone.
- **Evaluation runs on two axes**, each changing one quantity: `random` (fleet size 2-32
  at the training density) and `density` (0.25x-3x the training density at N=4 and N=8),
  plus `circle` as the stress case. Training keeps one density for every fleet size, so
  the training fleet size is the only thing that differs between training runs.
- **Checkpoints from before the observation layout changed cannot be loaded** (`state_dim`
  16 instead of 20 at N=2, i.e. everything under `outputs/study2/models/` and most of
  `outputs/train_dagger_multi_robot/`). They fail with "Canonical ego features must
  concatenate to shape (B, 6)". The study-1 retrain runs (`deepset_*_h1/h10_n04_s*`) still
  load.
- Retrained `deepset_flow_h1_n04_s*` shares its name with an original study 1 run.
  Keep retrained runs out of `outputs/study1/models/`; evaluate with
  `MODELS=<dir> ./eval_study1.sh`. Runs started before the relabelling are named
  `deepset_mlp_n04_s*` instead of `deepset_mlp_h1_n04_s*`; both plot as MLP h=1.
- The first study 2 grid is done; its results are in `outputs/study2/`.

## Common tasks

```bash
# Tests are unittest; pytest is not installed.
python3 -m unittest discover -s test -t .
python3 test/test_config_loaders.py            # a single file

# Regenerate configs -- in this order, the policy generator reads the fleet YAMLs.
python3 test/config/generate_fleet_configs.py
python3 test/config/generate_circle_configs.py
python3 learning/config/study/generate_study_policy_configs.py
```

To check that a policy config trains without training anything, run
`learning.train_dagger.main()` with `sys.argv` set to the real launch arguments and
`learning.dagger.dagger_trainer.DaggerTrainer.run` replaced by a stub that captures
`self.cfg`. This goes through the real CLI merge and `DaggerConfig` validation. Call
`self.setup()` inside the stub to also build the policy and simulator.

To validate a scenario config: `core.config.load_and_validate_system_config("multi_robot", path)`.

## Invariants — keep these true

- **Generated YAMLs are never edited by hand.** Change the template or generator, then
  regenerate. Every generator validates its output and fails loudly.
- **The schedule has one source: the generator.** `train.sh` passes
  `train_dagger.py` only `--experiment-name/--system/--expert-config/--policy-config/
  --seed/--checkpoint-dir`. Command-line flags override the config's `training:` section, so never add
  a schedule flag to the runner.
- **Evaluation ignores policy-config overrides.** `evaluate_scaling.py` scores against
  the scenario config only. A policy config's `workspace_bounds` or tolerance overrides
  change training and in-loop eval, not the final numbers. Study 1's retrain does
  set them (to match the example config), so its training and final criteria differ.
- **Checkpoint file is `<policy_type>_dagger_checkpoint.pt`**, and a variant label is
  not a policy type (`mlp_h10` is an `mlp`). `train.sh` runs are
  `outputs/<experiment>/models/<policy>_<expert>_s<seed>/`. The old study runs are
  `<encoder>_<variant>_n<NN>_s<seed>`, which is the only layout the eval scripts read.
- **`evaluate_scaling.py` appends to its output CSV.** Both runners delete the
  per-policy CSV first; do the same when calling it by hand.

- **Step budgets are derived per config** (`step_budget` in `test/evaluate_scaling.py`):
  `3 * longest distance / (max_linear_vel * dt)`. A fixed budget would fail sparse configs
  by timeout just for having a larger workspace. The expert alone needs ~1.7x the
  straight-line time on the ring.

## Gotchas already hit

**Config schema**
- The `unicycle2` keys were renamed upstream: `max_linear_vel`, `max_angular_vel`,
  `max_linear_accel`, `max_angular_accel`. `goal_position_bounds` and
  `initial_position_*` no longer exist and fail validation.
- `workspace_bounds` bounds **both** random starts and random goals, absolutely. It
  defaults to `[-1, 1]`, so a config that omits it silently samples a tiny arena.
- Explicit `initial_states` / `start` are not checked against `workspace_bounds`.
- `validate_system_config` moves a robot's `start` from `config.start` to the robot
  entry's top level. Read both places.
- A policy config's `training_curriculum` needs exactly one entry per DAgger round. A
  quick test that overrides `--dagger-iterations 1` must also pass
  `--training-curriculum config` and matching per-round lists.

**Python / tooling**
- `test/` shadows the standard library's `test` package, and `test/config/` is not a
  package. The circle generator loads the fleet generator by file path for that reason.
- `__pycache__/` is gitignored repo-wide. One `.pyc` used to be tracked; don't reintroduce it.

**Bash / git**
- In `echo "$(date) ... $?"` the command substitution runs first and resets `$?`. The
  runners capture the status into a variable before anything else.
- Bash reads a script as it runs. Don't edit `train.sh`/`eval.sh` in place while a run is
  using it; write a copy and `mv` it over.
- Don't run two commands that write `.git/index` in parallel (`git rm`, `git add`,
  `git rm --cached`). One silently undoes the other.

**Docker** (`compose.yaml`, `train.sh`, `eval.sh`)
- Run containers with `-u "$(id -u):$(id -g)" -e HOME=/tmp -e USER=csvil`. Without
  `-u`, outputs are root-owned; without `USER`, LeRobot's `getpass.getuser()` fails
  with `getpwuid(): uid not found`.
- `*_NUM_THREADS=1` per job, or parallel jobs oversubscribe the CPU.
  `docker compose run` has no `--cpus`.
- The `csvil` build uses `network: host`. BuildKit copies a systemd-resolved host's
  stub `resolv.conf`, so `apt-get` otherwise fails with "Temporary failure resolving".
  The `db-lacam` service lacks this and fails the same way if rebuilt.
- A VPN on the host (Cisco `cscotun0`) has also broken container networking.

**Experiments**
- Collisions count below 1.0 m (`d_collision`), while the expert plans with
  `d_safe` 1.2 m. Scoring at 1.2 m made the expert "fail" the ring, because its MPC
  sits right at that boundary.
- The collision check is a strict `distance < threshold` with no tolerance band, so
  boundary grazes count as crashes.
- With robot speed now 1.0, the 32-robot ring needs ≥368 steps. The circle eval budget
  is 400 steps; raise it to ~600 before trusting N=32 ring results.
- The MLP is deterministic, so on the fixed ring layout its 50 episodes collapse to one
  pass/fail.
- At 50 episodes per cell the 95% interval is about ±0.08–0.14. Don't claim cell
  differences smaller than that.
- Ring results at a policy's training fleet size are partly memorised: training rounds
  start some episodes from the exact evaluation ring.
- The MPC expert's cost grows super-quadratically with fleet size. Anything that runs
  it (`train_dagger.py`, `evaluate_policy.py`) is expensive past ~8 robots;
  `evaluate_scaling.py` exists to avoid it.

## Open items

- `eval_study1.sh` still uses fixed step budgets (200/400) and the old study-1 layout.
- After the study 1 retrain, update its section in `experiment_plan.md`; the current
  text describes the original h8 runs.
- `db-lacam` in `compose.yaml` still needs `network: host` for its build.
