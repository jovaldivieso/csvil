# Running the studies

How to train, evaluate and plot both studies. What they ask and what they found is in
[experiment_plan.md](experiment_plan.md).

## Setup

The CasADi expert dominates the cost and is CPU-bound, so train on a many-core x86_64
machine (the amd64 image is unusably slow under emulation on ARM).

```bash
git clone -b <branch> git@github.com:jovaldivieso/csvil.git && cd csvil
docker compose build csvil
tmux new -s study
```

`train.sh` and `run_study.sh` run each job in its own container with `-u $(id -u):$(id -g)` so files
stay host-owned, `HOME=/tmp` and `USER=csvil` because the host uid has no passwd entry
in the image, and `*_NUM_THREADS=1` so parallel jobs don't oversubscribe the cores.

## Training

`train.sh` trains every policy config against every expert config, one Docker container
per run:

```bash
./train.sh <experiment> <policy_dir|policy.yaml> <expert_dir|expert.yaml>
#   env: SEEDS="0 1 2" (default "0"), MAX_PARALLEL=8
```

Run `<policy>_<expert>_s<seed>` writes its checkpoints to
`outputs/<experiment>/models/<run>/` and its log to `outputs/<experiment>/logs/<run>.log`.
The script prints a summary at the end and exits non-zero if any run failed. Paths
must be inside the repository, because the container mounts only the repository.

The script passes `train_dagger.py` only the run identity (name, system, expert config,
policy config, seed, checkpoint dir). Each run's model and full DAgger schedule live in
its policy config.

**Study 2 (encoders).** The policy configs are generated, one directory per training
fleet size:

```
learning/config/study/n{02,04,06,08}/{deepset,transformer,gnn}_{mlp,flow}.yaml
```

Each file holds ring start/goal layouts with one entry per robot. A config therefore
only fits the expert config with the same number of robots, so train each directory
with its own fleet size:

```bash
for n in 08 06 04 02; do
  MAX_PARALLEL=$(nproc) ./train.sh study2 learning/config/study/n$n \
    test/config/study/unicycle2_fleet_$n.yaml
done
```

Pass the expert config as a single file: `test/config/study/` also holds the 16- and
32-robot evaluation configs, where the MPC expert is far too expensive to train with.
A mismatched pair fails at the first rollout with "Rollout #0 has 4 entries, expected
8".

Variants are `mlp` (horizon 1) and `flow` (horizon 8). The schedule (`training_schedule`)
is the same at every fleet size: 3 rounds of 100 trajectories x 200 steps, a third of
them from ring layouts, and a fixed 40000 gradient steps per round. To change it, edit
the constants in `learning/config/study/generate_study_policy_configs.py` and
regenerate:

```bash
python3 learning/config/study/generate_study_policy_configs.py
```

To try a one-off setting without regenerating, call `train_dagger.py` directly and add
the flag. A command-line flag overrides the config's value for that run only.

## Evaluation

[`test/evaluate_scaling.py`](../test/evaluate_scaling.py) rolls each checkpoint out
**without the expert**, so fleets up to 32 robots are feasible. [`eval.sh`](../eval.sh)
runs it for every checkpoint of an experiment:

```bash
./eval.sh <experiment> [arena|fleet|density|circle|all]
#   env: EPISODES=50 MAX_PARALLEL=8 RUNNER=docker|local SEED_START=50000
#        STEP_BUDGET_FACTOR=3 ACTION_NOISE=0.0
```

It reads `outputs/<experiment>/models/<run>/*_dagger_checkpoint.pt`, takes the training
seed from the trailing `_s<seed>` of the run name, and writes one CSV per run to
`outputs/<experiment>/eval/<scenario>/`, merged into
`outputs/<experiment>/eval/<scenario>.csv`.

### The three scenarios

What a scenario changes for the policy is the distribution of visible neighbours: the
encoder sees only neighbours within 4 m, never the fleet size. Two things drive it,
**density** and **fleet size**, and each scenario changes exactly one of them.

| Scenario | Configs | Changes |
|---|---|---|
| `arena` | `test/config/study/arena/unicycle2_n{02..32}.yaml` | N = 2…32 in **one fixed ±6.5 m workspace** (after GLAS): same task and path length throughout, density rises with the fleet |
| `fleet` | `test/config/study/fleet/unicycle2_n{02..32}.yaml` | fleet size 2…32 at the training density. Kept but **not part of `all`**: its box grows as √N, so path length rises with the fleet and it isolates no better than `arena` |
| `density` | `test/config/study/density/unicycle2_n{02,06}_d{025,05,1,133}.yaml` | 0.25×…1.33× the training density **at a fixed fleet size**: spacing shrinks, the ceiling on visible neighbours does not |
| `circle` | `test/config/study/circle/unicycle2_circle_{02..32}.yaml` | antipodal swap from fixed starts, the stress case (`--use-config-start`) |

Training keeps one density for every fleet size (about 0.056 robots/m²), so the training
fleet size is the only thing that differs between training runs. The `d1` config of a
density level is identical to the fleet config of the same N and anchors the two axes.

The density sweep stops at 3×: the sampler has to place 2N starts and goals at least
`d_safe` apart, which never failed up to 3× (38% of the area blocked), while at 4× it
failed for 20% of the 32-robot episodes and at 5× for 43% of the 8-robot ones.

### Step budget

Each config gets its own budget, since a fixed one would fail sparse configs by timeout
merely for having a larger workspace:

  `steps = ceil(3 · longest distance / (max_linear_vel · dt))`

The longest distance is the workspace diagonal, or with `--use-config-start` the longest
start-to-goal distance. Examples: 510 steps at N=2 random, 2038 at N=32 random, 360 for
rings up to N=8, 1102 for the 32-robot ring. The factor is 3 because avoiding costs time:
the CasADi expert alone needs about 1.7× the straight-line time on the ring (197 steps at
N=4, 204 at N=8, against 120). Episodes that succeed end early, so a generous budget only
costs time on the failures. `--steps` still forces a fixed budget, and the `steps` column
of each row records what was used.

**Success:** within the step budget, every robot is within 0.2 m and 1.1 rad of its goal
with speed and yaw rate below 0.1, and no pair of robots ever came within 1.0 m.

Episodes start from seed 50000, so all policies see the same start states, and flow's
action sampling is seeded per episode, so results are reproducible.

Each result row covers one checkpoint on one config: success, collision and timeout rate;
mean terminal position and heading error over all episodes; mean closest pair distance;
and mean policy time per control step for the whole fleet. The `density` column holds the
robots per m² of that config, which is the only way to tell the density levels of one
fleet size apart; for the ring it comes from the area the fixed layout spans (4× the
training density at N=8, 1.71× at N=32).

### Policy against the expert

[`eval_vs_expert.sh`](../eval_vs_expert.sh) runs
[`test/evaluate_policy.py`](../test/evaluate_policy.py), which rolls the expert out from
the same start as the policy and produces a PDF and an MP4 per chunk of episodes. It is
for looking at behaviour and for the expert as a reference, not for the study numbers:
the MPC makes it slow above about 8 robots.

```bash
CK=outputs/<experiment>/models/<run>/flow_dagger_checkpoint.pt \
OUT=outputs/plots/evaluate_policy/<run> EPISODES=200 ./eval_vs_expert.sh
```

### Study 1

Runs locally without Docker; reads `outputs/study1/models/`.

```bash
./eval_study1.sh                     # -> outputs/study1/eval/random/study1.csv
SCENARIO=circle ./eval_study1.sh     # -> outputs/study1/eval/circle/study1.csv
python3 test/plot_study1_results.py  # -> outputs/study1/plots/
```

The plot shows success rate against evaluation fleet size, one panel per scenario:
colour is the policy head, dash is the horizon, the band spans the three seeds.

### Study 2 (encoders)

Reads `outputs/<experiment>/models/`, written by `train.sh`.

```bash
./eval.sh study2 all                  # random + density + circle
for s in arena density circle; do for p in mlp flow; do
  python3 test/plot_study_results.py --results outputs/study2/eval/$s.csv \
    --policy $p --label $s --output-dir outputs/study2/plots/$s
done; done
```

Checkpoints from before the observation layout changed (`state_dim` 16 instead of 20 at
N=2) cannot be loaded any more and fail with "Canonical ego features must concatenate to
shape (B, 6)". The results under `outputs/study2/` from the first grid are in that group.

`plot_study_results.py` plots either axis. `--axis auto` (the default) reads it off the
results: several densities for one fleet size means the density sweep, otherwise the
fleet sweep. On the density axis the x-axis is the density as a multiple of the training
density, the in-distribution cells are the 1× ones, and each figure is written once per
evaluation fleet size (`..._n04`, `..._n08`), since the fleet sizes would otherwise share
a cell. Results without a `density` column predate it and can only be plotted as a fleet
sweep.

Per policy and scenario this writes a success matrix (training fleet size across,
evaluation fleet size down), success against evaluation fleet size pooled over training
sizes, and the same split into one panel per training size. Bars are 95% Wilson
intervals.
