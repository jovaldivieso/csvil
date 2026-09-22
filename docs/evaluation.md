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
**without the expert**, so fleets up to 32 robots are feasible. Every policy trained at
one fleet size is evaluated at all fleet sizes on two scenarios:

| Scenario | Layout | Steps | Episodes |
|---|---|---|---|
| `random` | random starts and goals; the goal area grows with the fleet | 200 | 50 |
| `circle` | robots on a ring swap to the opposite side | 400 | 50 (the deterministic MLP is collapsed to 1) |

**Success:** within the step budget, every robot is within 0.2 m and 1.1 rad of its goal
with speed and yaw rate below 0.1, and no pair of robots ever came within 1.0 m. The
criterion is the same for both studies; the scripts apply it by writing derived
scenario configs.

Episodes start from seed 50000, so all policies see the same start states, and flow's
action sampling is seeded per episode, so results are reproducible.

Each result row covers one checkpoint at one evaluation fleet size: success, collision
and timeout rate; mean terminal position and heading error over all episodes; mean
closest pair distance; and mean policy time per control step for the whole fleet.

### Study 1

Runs locally without Docker; reads `outputs/study1/models/`.

```bash
./eval_study1.sh                     # -> outputs/study1/eval/random/study1.csv
SCENARIO=circle ./eval_study1.sh     # -> outputs/study1/eval/circle/study1.csv
python3 test/plot_study1_results.py  # -> outputs/study1/plots/
```

The plot shows success rate against evaluation fleet size, one panel per scenario:
colour is the policy head, dash is the horizon, the band spans the three seeds.

### Study 2

Runs in Docker; reads `outputs/study2/models/`. It still expects the old run names
(`<encoder>_<variant>_n<NN>_s<seed>`), so it does not pick up runs trained with `train.sh` yet.

```bash
ENCODERS="deepset transformer gnn" POLICIES="mlp flow" ./run_study.sh eval
ENCODERS="deepset transformer gnn" POLICIES="mlp flow" ./run_study.sh eval circle
for s in random circle; do for p in mlp flow; do
  python3 test/plot_study_results.py --results outputs/study2/eval/$s/encoder_scaling.csv \
    --policy $p --label $s --output-dir outputs/study2/plots/$s
done; done
```

Per policy and scenario this writes a success matrix (training fleet size across,
evaluation fleet size down), success against evaluation fleet size pooled over training
sizes, and the same split into one panel per training size. Bars are 95% Wilson
intervals.
