# Encoder-scaling study

Trains the three neighbour encoders (`deepset`, `transformer`, `gnn`) on fleets of
2, 4, 6 and 8 `unicycle2` robots and evaluates every resulting policy on 2, 4, 6, 8,
16 and 32 robots.

## 1. Running a training on the remote machine

The CasADi expert dominates the cost and is CPU-bound, so use a many-core **x86_64**
box — on ARM the `linux/amd64` image runs under QEMU emulation and is unusably slow.

```bash
git clone -b <branch> git@github.com:jovaldivieso/csvil.git && cd csvil
docker compose build csvil
```

One training run is just `learning/train_dagger.py` in the container:

```bash
docker compose run --rm -T \
  -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  csvil \
  python learning/train_dagger.py \
    --experiment-name deepset_n08 \
    --system multi_robot \
    --expert-config test/config/study/unicycle2_fleet_08.yaml \
    --policy-config learning/config/study/deepset_mlp_config.yaml \
    --dagger-iterations 5 \
    --trajectories-per-iteration 13 \
    --steps-per-trajectory 200 \
    --target-epochs-per-round 10 \
    --action-noise-std 0.03 \
    --expert-mix-beta-start 0.5 \
    --expert-mix-beta-decay-rate 0.25 \
    --expert-mix-decay-after-success-rate 0.0 \
    --eval-episodes 50 \
    --seed 99
```

Only four arguments differ between runs:

| Argument | Varies with | Values |
|---|---|---|
| `--experiment-name` | both | `<encoder>_n<NN>`; names the output directory |
| `--expert-config` | fleet size | `test/config/study/unicycle2_fleet_{02,04,06,08}.yaml` |
| `--policy-config` | encoder | `learning/config/study/{deepset,transformer,gnn}_mlp_config.yaml` |
| `--trajectories-per-iteration` | fleet size | `50 / 25 / 17 / 13` for N = 2 / 4 / 6 / 8 |

Trajectories scale inversely with fleet size because each episode emits one LeRobot
episode *per robot* — a fixed count would give the 8-robot run four times the frames
and gradient steps of the 2-robot run. Everything else stays identical, `--seed 99`
included; that is what makes the runs comparable.

Results land in `outputs/train_dagger_multi_robot/<experiment-name>/`:
`mlp_dagger_checkpoint.pt` (latest), `mlp_dagger_iter_NNN.pt` (per round),
`results.csv` (final round only) and copies of both configs. Outside Docker, drop
everything up to `csvil` and run the `python ...` part directly.

**Docker flags that matter:**

- `-u "$(id -u):$(id -g)"` and `HOME=/tmp` — without them the container runs as root
  and every checkpoint it writes into the bind mount is root-owned.
- `*_NUM_THREADS=1` — IPOPT/BLAS and torch each grab every core otherwise, so
  parallel runs oversubscribe the machine. `docker compose run` has no `--cpus` flag.
- `compose.yaml` sets `network: host` on the `csvil` build. BuildKit copies the
  host's `/etc/resolv.conf` verbatim, so on a systemd-resolved host the build
  container inherits a stub nameserver and every `apt-get` package fails to resolve.

## 2. The study script

[`run_study.sh`](../run_study.sh) wraps the command above and runs the whole grid in
parallel, one container per policy.

```bash
tmux new -s study                                 # survives disconnect
MAX_PARALLEL=$(nproc) ./run_study.sh train        # all 12 runs
# alternatively:
./run_study.sh train 8                            # only the 8-robot fleet
ENCODERS="deepset" ./run_study.sh train 8         # one encoder, one fleet size
```

Extra arguments to `train` restrict the fleet sizes; with none it trains every
encoder on every size. Sizes run largest-first, because the biggest fleet dominates
wall clock and must not queue behind cheap runs. Progress goes to
`logs/<policy>.log`.

Pull the results back (~175 MB; leave `data/` behind, evaluation does not need the
collected datasets):

```bash
rsync -av user@host:~/csvil/outputs/train_dagger_multi_robot/ ./outputs/train_dagger_multi_robot/
```

## 3. Evaluation

```bash
./run_study.sh eval            # random scenario (default)
./run_study.sh eval circle     # antipodal-swap scenario
```

| scenario | configs | episodes | steps | merged output |
|---|---|---|---|---|
| `random` | `test/config/study/` | 50 | 200 | `outputs/study/encoder_scaling.csv` |
| `circle` | `test/config/study/circle/` | 1 | 400 | `outputs/study/circle/circle_scaling.csv` |

`random` uses the randomized-goal configs the policies trained on. `circle` is the
deterministic antipodal swap, where every robot crosses the centre toward a goal
that is another robot's start; it is deterministic, so one episode is the whole
result, and it needs 400 steps because the 32-robot ring is ≥184 steps just to cross.
Override per run with `EVAL_EPISODES`, `EVAL_STEPS`, `EVAL_NOISE`. Adding a scenario
is a row in the `SCENARIO_*` tables at the top of the script. Re-running replaces
results rather than appending.

Both modes call [`test/evaluate_scaling.py`](../test/evaluate_scaling.py), which
rolls out **the policy only** — `test/evaluate_policy.py` also rolls out the expert,
whose MPC cost grows super-quadratically and is impractical past ~8 robots. A
checkpoint carries its own architecture, so evaluation needs nothing but the `.pt`
and the configs:

```bash
python test/evaluate_scaling.py \
  --checkpoint outputs/train_dagger_multi_robot/deepset_n04/mlp_dagger_checkpoint.pt \
  --configs test/config/study/unicycle2_fleet_{02,04,06,08,16,32}.yaml \
  --episodes 50 --steps 200 --output-csv outputs/study/deepset_n04.csv
```

Columns: `success_rate`, `collision_rate`, `timeout_rate`, `mean_steps`,
`mean_goal_error_l2`, `mean_min_pair_distance`.

Plot the merged CSV with [`test/plot_study_results.py`](../test/plot_study_results.py):

```bash
python test/plot_study_results.py                                    # -> outputs/plots/random/
python test/plot_study_results.py --results outputs/study/circle/circle_scaling.csv \
                                  --output-dir outputs/plots/circle
```

It writes a per-encoder matrix (train fleet across, eval fleet down), a pooled
success-rate-vs-fleet-size line plot with 95% Wilson intervals, and the same
un-pooled as one panel per training fleet size.

### Reading the numbers

- `collision_rate` is `1 - success_rate` in every cell (timeouts are ~0), so the two
  matrices are mirror images — show one.
- Collision is `distance² < d_safe²` with no tolerance, and the expert's MPC drives
  that constraint exactly active, so grazes at the boundary score as crashes. Check
  `mean_min_pair_distance` before reading a collision rate as a safety failure.
- At 50 episodes/cell the 95% interval is ±0.08–0.14, wider than the spread across
  all twelve policies. Encoder differences are not resolvable at this sample size.
- The circle scenario is currently unsolvable by the expert itself, so those results
  say nothing about the policies until the ring symmetry is broken.
