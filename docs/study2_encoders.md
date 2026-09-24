# Study 2 — neighbour encoders

What this experiment asks, how it is trained and what it is evaluated on. How to run
the commands: [evaluation.md](evaluation.md). Repository orientation:
[agent_context.md](agent_context.md).

## Question

Which neighbour encoder — **DeepSet, Transformer or GNN** — handles surrounding robots
best, and how does the fleet size a policy was trained on affect its behaviour at other
fleet sizes and densities?

Every robot runs the same policy, decentralized: it sees its own state, its goal and the
neighbours within its 4 m visibility radius. The encoder turns that variable-length
neighbour set into a fixed 128-wide context. That is the part under study; everything
else is held equal.

## Training

Grid: **3 encoders × 4 training fleet sizes = 12 runs**, one seed, `mlp` head
(`data_mid`; `data_large` repeats N = 2, 4, 6 with more data).

| | |
|---|---|
| Encoders | DeepSet (φ [128,128], ρ [128], mean pooling), Transformer (128, 4 heads, 1 layer, dropout 0.1), GNN (128, 1 layer, sum over neighbours) |
| Head | `mlp`, predicts 1 action |
| Shared | 3×256 policy network, 2-step observation history, 128-wide context |
| Fleet sizes | 2, 4, 6, 8 |

**Method: DAgger with a CasADi MPC expert.** In each round the policy drives the
rollouts, the expert labels every visited state, the data is aggregated, and the policy
trains on everything collected so far.

| | |
|---|---|
| Rounds | 5 |
| Episodes per round | 480 / 180 / 95 / 60 for N = 2 / 4 / 6 / 8 (a third from ring layouts) |
| Frames per round | 144 000 in every cell — one episode yields one dataset episode per robot |
| Steps per episode | 150 / 200 / 250 / 300 |
| Training per round | 40 epochs over the whole aggregated dataset, no `max_train_steps` cap |
| Expert share (beta) | 0.5, −0.25 per round once in-training eval success exceeds 0.5 |
| Backtrack recovery | a stuck episode is replayed 75% expert-driven, +0.25 per failed attempt |
| Action noise | 0.03 |
| In-training eval | 20 episodes, scenario tolerances |

**Training scenario:** random starts and goals in a workspace scaled to
**0.167 robots/m²** at every fleet size (±1.73 / ±2.45 / ±3.0 / ±3.46 m for
N = 2 / 4 / 6 / 8), so training sizes differ only in the number of robots, not in
crowding. Robot 1 m (`d_collision`), planning distance `d_safe` 1.2 m, visibility 4 m,
top speed 1 m/s, dt 0.05.

**Episode length scales with the workspace** (250 steps at ±3 m, i.e. about 1.5× the
straight-line diagonal at every fleet size). A fixed budget would let the expert finish
almost no large-fleet episode, the in-training eval would never clear the beta-decay
gate, and large fleets would see far fewer arrivals than small ones.

The ring layouts are the training-time counterpart of the `circle` evaluation scenario,
varied by radius, start heading, symmetry breaking (jitter) and ellipse squash; all
goals are antipodal. Rollout 0 is the evaluation ring itself, so ring results at the
training fleet size are partly memorised.

Configs: `learning/config/study/data_mid_n{02,04,06,08}/<encoder>_mlp.yaml` (policy) and
`test/config/study/unicycle2_fleet_{02,04,06,08}.yaml` (expert), the latter supplying the
task while the policy config overrides the workspace to the training density. Both are
generated; never edit them by hand.

## Evaluation

The policy runs **without the expert** (`test/evaluate_scaling.py`), so fleets up to 32
robots are affordable. Every checkpoint sees the same episodes (seeds from 50000), and
flow's action sampling is seeded per episode, so results are reproducible.

**What the axes vary.** A policy sees the neighbours inside its 4 m sensing radius R.
Fleet size, density and workspace are tied by ρ = N/L², so only one of the three can be
pinned at a time, and what a scenario varies follows from which one that is:

    visible ≈ min(N − 1, ρ·πR²·boundary)        spacing ∝ 1/√ρ        R/L = R·√(ρ/N)

Three scenario sets are evaluated:

| Axis | What changes | What stays | Configs |
|---|---|---|---|
| **1. Arena** (headline) | N = 2…32 in **one fixed ±6.5 m workspace**, after GLAS: visible neighbours 0.15 → 6.0, nearest neighbour 7.7 → 1.7 m, conflicts per robot 0.08 → 4.0; density rises 0.07× → 1.13× of training | the task itself — box, goal distribution, **~7 m to drive at every N** | `test/config/study/arena/unicycle2_nNN.yaml` |
| **2. Density** | 0.25×, 0.5×, 1×, 1.25× the training density at N = 2 and 6: nearest neighbour 5.6 → 1.7 m, conflicts per metre 0.10 → 0.42 | fleet size, and with it the ceiling N−1 | `test/config/study/density/unicycle2_nNN_dF.yaml` |
| **3. Ring** | antipodal swap, N = 2…32, ringed at the training density: every robot crosses the centre toward a goal that is another robot's start | density, so only the fleet size varies here too | `test/config/study/circle/unicycle2_circle_NN.yaml` |

**The arena axis isolates best**, which is why it is the headline and why GLAS uses it:
the workspace, the goal distribution and hence the path length are identical at every
fleet size, so nothing varies but the number of robots sharing the space. It answers the
deployment question — how many robots fit in *this* arena before the policy breaks.

Two caveats attach to it. The sweep is out of distribution in two directions at once:
density is below training everywhere except at N=32, and the goals are further away than
in training (~7 m mean against 2.2–4.0 m in the training boxes), while the goal vector
enters the observation unscaled — GLAS clips it to the sensing radius precisely to avoid
this. Comparisons **along** the axis stay valid, since goal distance is constant across
N; the absolute level is depressed throughout, and most for the policies trained at the
smallest fleets.

**The density axis** fixes the ceiling on visible neighbours and compresses the geometry
instead. It separates *room* from *number*, which the arena axis cannot: the same six
robots in a tight box and in a wide one. Conflicts per metre scale with density, while
the path shortens (6.4 → 3.2 m at N=6), so conflicts per episode grow only about 2×. It
reaches 1.25× because the sampler has to place 2N starts and goals `d_safe` apart and
fails above that; 1× is the condition the policies trained under and is marked
in-distribution.

The **ring is sized by density, not by a fixed radius**: R = √(N/ρ)/2 at the training
density, giving 1.73 / 2.45 / 3.0 / 3.46 / 4.90 / 6.93 m for N = 2…32 — the circle
inscribed in the training box. One radius for every fleet (the earlier 3.0 m) would have
made ring density grow 4× from N=2 to N=8, mixing the two effects again. At N=32 the
robots start 1.36 m apart, 1.13× `d_safe`: feasible, but the tightest ring of the set.

**Dropped: the fleet-size axis at constant density** (N = 2…32 in boxes that grow as √N,
`test/config/study/fleet/`). It was meant as the clean fleet-size test, but the
measurements showed the box growth drags the path length with it (2.2 → 7.6 m) and
conflicts per robot rise 0.30 → 3.78, so it is no more isolated than the arena axis while
being further from the deployment question. The configs and the `fleet` scenario remain,
they are simply not part of `./eval.sh <experiment> all`; the fairest in-distribution test
of fleet-size generalisation, should it be wanted, is that axis restricted to N = 16, 32.

**Success, per episode:** within the step budget, every robot is within 0.2 m and 1.1 rad
of its goal with speed and yaw rate below 0.1, and no pair of robots ever came within
1.0 m.

**Success, per robot** (as in GLAS, eq. 6): a robot succeeds when it ends at its goal and
was never within 1.0 m of another. Fleet-level success needs all N robots at once, so it
falls as p^N and is uninformative at large fleets: measured on a trained checkpoint, the
episode success is 0.000 at both N=8 and N=16 while the per-robot rate distinguishes them
(0.081 against 0.047, with 68% and 80% of robots in a collision). Episodes therefore run
on after a collision — stopping at the first one would score every robot of the fleet as
failed because two of them touched.

**Recorded per checkpoint and scenario:** success, collision and timeout rate, mean
steps, mean final position and heading error, mean closest pair distance, robot density,
and policy time per control step.

**Figures** (`test/plot_study_results.py`): per encoder a matrix with the training fleet
size across and the axis down. A cell shows the success rate as colour and number, and
underneath the failure split `C .33 T .67` — collision against timeout, two different
failures: driving into a neighbour, or avoiding without ever arriving. In-distribution
cells are outlined (`--train-density` moves that row if a run trained elsewhere). Plus the pooled line plot along the axis with 95% Wilson intervals
and the same split into one panel per training fleet size.

On the ring a deterministic MLP produces **one** distinct episode per cell (fixed start,
no action noise), so `evaluate_scaling.py` collapses the repeats and the result is a
single pass or fail. More episodes only help there with `ACTION_NOISE` > 0, or for flow,
which samples its actions.

**Step budget:** 3 × the straight-line travel time (workspace diagonal, or the longest
start–goal distance for the ring). A fixed budget would make sparse scenarios fail by
timeout purely because their distances are longer. The factor 3 comes from a
measurement: on the ring the expert needs about 1.7× the straight line (197 steps at
N=4, 204 at N=8 for 6 m), because the robots have to give way around the centre; a
learned policy is slower than that.

**What the results answer:**
- the encoder comparison within a head (the primary question);
- generalisation to fleet sizes never trained on, especially 16 and 32;
- robustness to sparser and denser crowds than in training;
- the ring as a stress test for head-on conflicts, where a deterministic MLP has the
  known weakness of averaging "pass left or right" into "drive straight through";
- inference cost per control step, which matters for the real robots.

## The data axis: `data_mid` (main) and `data_large`

The Training section describes `data_mid`, the study's main run. `data_large` repeats it
with 300 000 instead of 144 000 frames per round at N = 2, 4, 6, so that the amount of
data is a controlled variable: density, episode length, rounds, epochs and schedule are
identical, and a difference between the two runs can have no other cause.

| | `data_mid` | `data_large` |
|---|---|---|
| Frames per round | 144 000 | 300 000 |
| Fleet sizes | 2, 4, 6, 8 | 2, 4, 6 |
| Episodes per round | 480 / 180 / 95 / 60 | 1000 / 375 / 200 |
| Rounds × epochs | 5 × 40 | 5 × 40 |
| Longest run | 34.8 h (N=8) | 54.4 h (N=6) |

An earlier grid at 0.0556 robots/m² with 5 × 200 episodes is **superseded**: its
checkpoints predate the current observation layout and no longer load.

Configs: `learning/config/study/data_{mid,large}_n<NN>/<encoder>_mlp.yaml`, generated by
`learning/config/study/generate_data_pilot_configs.py`.

## Limits and caveats

- **One seed per cell.** A difference between encoders cannot be told apart from
  training noise. Any claim of an encoder ranking needs the run repeated with 3 seeds.
- **The training seed currently varies only weights and batch order.** Episode layouts
  come from `initial_state_seed` in the expert config, so additional seeds would train
  on the same planned episodes until that is changed.
- **Density limit 3×**, which is the training density itself. The sampler has to place
  2N points (starts and goals) at least `d_safe` apart; up to 3× (38% of the area blocked)
  it never failed in 100 episodes at any fleet size, at 4× it failed 20% of the time at
  N=32 and at 5× 43% at N=8. The density axis therefore reaches from a quarter of the
  training density up to the training density, not beyond it.
- **Ring results at the training fleet size are partly memorised** (see above).
- **Cells do not share data.** Each policy partly drives its own DAgger rollouts, so a
  difference reflects both the policy and the data it collected.
- **Solvability** is never the limiting factor: the workspace has no walls
  (`workspace_bounds` only bounds sampling), so in an open plane every placed instance
  can be solved given enough time. The step budget and the expert's own quality are the
  practical limits.

## Cost

The expert runs at every step, and it dominates: about 0.16 / 0.46 / 0.83 / 1.5 s per
step at N = 2 / 4 / 6 / 8. Per `data_mid` run that is roughly 12 h (N=2), 18 h (N=4),
22 h (N=6) and 29 h (N=8) of single-core time; `data_large` reaches 54 h at N=6. With one
core per run all 12 (or 9) go in parallel and the largest fleet sets the pace, about
1.5 days for `data_mid` and 2.3 for `data_large`. Evaluation is cheap by comparison: no
expert, a few ms per control step.
