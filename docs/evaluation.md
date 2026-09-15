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

`run_study.sh` runs each job in its own container with `-u $(id -u):$(id -g)` so files
stay host-owned, `HOME=/tmp` and `USER=csvil` because the host uid has no passwd entry
in the image, and `*_NUM_THREADS=1` so parallel jobs don't oversubscribe the cores.

## Training

```bash
MAX_PARALLEL=$(nproc) ENCODERS="..." POLICIES="..." SEEDS="..." \
TARGET_EPOCHS=400 MAX_TRAIN_STEPS=40000 BETA_DECAY_AFTER=0.5 \
./run_study.sh train <fleet sizes>
```

| Study | `ENCODERS` | `POLICIES` | `SEEDS` | fleet sizes |
|---|---|---|---|---|
| 1 | `deepset` | `mlp flow_h1 mlp_h8 flow` | `0 1 2` | `4` |
| 2 (flow) | `deepset transformer gnn` | `flow` | `0` | `2 4 6 8` |

Policy variants are `mlp` and `flow_h1` (predict one action), `mlp_h8` and `flow`
(predict an 8-step chunk). Study 2's `deepset_flow_n04_s0` is study 1's run; copy it
rather than retrain it.

Runs are named `<encoder>_<variant>_n<NN>_s<seed>` and land in
`outputs/train_dagger_multi_robot/`; move them to `outputs/study1/models/` or
`outputs/study2/models/` when done. `MAX_TRAIN_STEPS` gives every DAgger round the same
number of gradient steps in every run — `grep optimizer_steps logs/*.log` should read
40000. Expected single-core time per run: about 2 h at N=2, 3.5 h at N=4, 14 h at N=6
and 38 h at N=8.

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

Runs in Docker; reads `outputs/study2/models/`.

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
