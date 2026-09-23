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

Grid: **3 encoders × 2 policy heads × 4 training fleet sizes = 24 runs**, one seed.

| | |
|---|---|
| Encoders | DeepSet (φ [128,128], ρ [128], mean pooling), Transformer (128, 4 heads, 1 layer, dropout 0.1), GNN (128, 1 layer, sum over neighbours) |
| Heads | `mlp` (predicts 1 action) and `flow` (flow matching, 10-action chunk, 3 Euler steps) |
| Shared | 3×256 policy network, 2-step observation history, 128-wide context |
| Fleet sizes | 2, 4, 6, 8 |

**Method: DAgger with a CasADi MPC expert.** In each round the policy drives the
rollouts, the expert labels every visited state, the data is aggregated, and the policy
trains on everything collected so far.

| | |
|---|---|
| Rounds | 5 |
| Episodes per round | 200: 67 ring layouts + 133 random goals |
| Steps per episode | 250 / 350 / 450 / 500 for N = 2 / 4 / 6 / 8 |
| Training per round | 40 epochs over the whole aggregated dataset, no `max_train_steps` cap |
| Expert share (beta) | 0.5, −0.25 per round once in-training eval success exceeds 0.5 |
| Action noise | 0.03 |
| In-training eval | 20 episodes, scenario tolerances |

The ring layouts are the training-time counterpart of the `circle` evaluation scenario:
radius 2–4 m, varied by start heading, symmetry breaking (jitter), ellipse and skewed
goals. Rollout 0 is the evaluation ring itself, so ring results at the training fleet
size are partly memorised.

**Episode length scales with the workspace** (250 steps at ±3 m). With a fixed 200 the
expert itself finishes almost no 8-robot episode: 98% of episodes need more, the
in-training eval then never clears the beta-decay gate, and large fleets would see far
fewer arrivals than small ones.

**Training scenario:** random starts and goals in a workspace that grows with the fleet
(±3.0 / ±4.24 / ±5.20 / ±6.0 m), so **every training fleet size has the same density**,
about 0.056 robots/m². Training sizes therefore differ only in the number of robots, not
in crowding. Robot 1 m (`d_collision`), planning distance `d_safe` 1.2 m, visibility
4 m, top speed 1 m/s, dt 0.05.

Configs: `learning/config/study/n{02,04,06,08}/<encoder>_<head>.yaml` (policy) and
`test/config/study/unicycle2_fleet_{02,04,06,08}.yaml` (expert). Both are generated;
never edit them by hand.

## Evaluation

The policy runs **without the expert** (`test/evaluate_scaling.py`), so fleets up to 32
robots are affordable. Every checkpoint sees the same episodes (seeds from 50000), and
flow's action sampling is seeded per episode, so results are reproducible.

Three scenario sets, each varying exactly one thing:

| Axis | What changes | Configs |
|---|---|---|
| **1. Fleet size** | N = 2, 4, 6, 8, 16, 32 at training density | `test/config/study/unicycle2_fleet_NN.yaml` |
| **2. Density** | 0.25×, 0.5×, 1×, 2×, 3× training density, at N = 4 and 8 | `test/config/study/density/unicycle2_nNN_dF.yaml` |
| **3. Ring** | antipodal swap, N = 2…32: every robot crosses the centre toward a goal that is another robot's start | `test/config/study/circle/unicycle2_circle_NN.yaml` |

Keeping the axes apart matters. In one fixed evaluation arena, density would rise 16×
from N=2 to N=32, and a "more robots" curve would really be showing "more crowding".
The 1× density configs are identical to the fleet configs of the same N, which anchors
the two axes to each other.

**Success:** within the step budget, every robot is within 0.2 m and 1.1 rad of its goal
with speed and yaw rate below 0.1, and no pair of robots ever came within 1.0 m.

**Recorded per checkpoint and scenario:** success, collision and timeout rate, mean
steps, mean final position and heading error, mean closest pair distance, and policy
time per control step.

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

## Data pilots (`data_mid`, `data_large`)

A run of the grid above costs days, almost all of it in the expert, so two smaller runs
ask first how much data a usable policy actually needs.

**Training.** Three neighbour encoders (DeepSet, Transformer, GNN) are trained with the
same `mlp` head at the fleet sizes N = 2, 4, 6, 8 (`data_mid`) and N = 2, 4, 6
(`data_large`), again with DAgger and the CasADi MPC expert: the policy drives the
rollouts, the expert labels every visited state, and each round trains 40 epochs over
everything collected so far. Two conditions are held equal across all cells, so that only
the fleet size and the encoder differ. First, **robot density**: every fleet is placed in
a workspace scaled to 0.167 robots/m² (±1.73 / ±2.45 / ±3.0 / ±3.46 m for N = 2 / 4 / 6 /
8, three times the density of the main study), and the episode budget scales with that box
(150 / 200 / 250 / 300 steps), so crowding and the time available per crossing are the same
everywhere. Second, **the amount of training data**: since every episode yields one dataset
episode per robot, a large fleet reaches the same dataset with far fewer episodes, so the
episodes per round are set to give each cell the same number of frames per round. A third
of each round starts from antipodal ring layouts, the rest from random goals; unlike the
main study the skewed-goal layout is left out, because its goal one seat past the antipode
does not force a head-on conflict.

| | `data_mid` | `data_large` |
|---|---|---|
| Frames per round | 144 000 | 300 000 |
| Fleet sizes | 2, 4, 6, 8 | 2, 4, 6 |
| Episodes per round | 480 / 180 / 95 / 60 | 1000 / 375 / 200 |
| Rounds × epochs | 5 × 40 | 5 × 40 |
| Longest run | 34.8 h (N=8) | 54.4 h (N=6) |

With density, episode length, rounds and epochs fixed, the two runs differ only in the
amount of data, so a difference between them can have no other cause. Both also use
backtrack recovery (`expert_mix_beta_recovery` 0.75, increment 0.25): a failed episode is
replayed 75% expert-driven rather than purely by the expert, so the recovered frames show
the policy getting out of the states it actually reaches.

**Evaluation** is the same as for the main study: every checkpoint is rolled out without
the expert on the same seeded episodes, along the three axes (fleet size, density, ring),
with the same success criterion and step budget. One difference matters when reading the
results: these policies train at 3× the study density, so their in-distribution point on
the density axis is the **3× level**, not 1×, and on the fleet-size and ring axes they are
out of distribution throughout.

Configs: `learning/config/study/data_{mid,large}_n<NN>/<encoder>_mlp.yaml`, generated by
`learning/config/study/generate_data_pilot_configs.py`.

## Limits and caveats

- **One seed per cell.** A difference between encoders cannot be told apart from
  training noise. Any claim of an encoder ranking needs the run repeated with 3 seeds.
- **The training seed currently varies only weights and batch order.** Episode layouts
  come from `initial_state_seed` in the expert config, so additional seeds would train
  on the same planned episodes until that is changed.
- **Density limit 3×.** The sampler has to place 2N points (starts and goals) at least
  `d_safe` apart. Up to 3× (38% of the area blocked) it never failed in 100 episodes at
  any fleet size; at 4× it failed 20% of the time at N=32, at 5× 43% at N=8.
- **Ring results at the training fleet size are partly memorised** (see above).
- **Cells do not share data.** Each policy partly drives its own DAgger rollouts, so a
  difference reflects both the policy and the data it collected.
- **Solvability** is never the limiting factor: the workspace has no walls
  (`workspace_bounds` only bounds sampling), so in an open plane every placed instance
  can be solved given enough time. The step budget and the expert's own quality are the
  practical limits.

## Cost

The expert runs at every step, and it dominates: about 0.16 / 0.46 / 0.83 / 1.5 s per
step at N = 2 / 4 / 6 / 8. Per run (5 rounds × 200 episodes) that is roughly 7 h (N=2),
27 h (N=4), 62 h (N=6) and 123 h (N=8) of single-core time. With 24 cores all 24 runs go
in parallel and N=8 sets the pace, about 5 days. Evaluation is cheap by comparison: no
expert, a few ms per control step.
