# Experiment plan: policy comparison and encoder comparison

Working document for the two planned studies. It holds the design, the metric
definitions, the known measurement problems that have to be fixed before results
mean anything, and the execution order. Keep the status boxes and the progress log
at the bottom current — this file is the shared record of what has actually been
run, not just what is intended.

Companion documents: [encoder_scaling_study.md](encoder_scaling_study.md) describes
how to *run* the existing encoder grid (remote machine, Docker flags,
`run_study.sh` usage). This file describes *what to run and why*.

---

## 1. The two studies

**Study 1 — policy family.** Does a conditional flow-matching policy beat a
deterministic MLP at imitating a multi-robot MPC expert? The hypothesis is that the
expert's action distribution is multimodal at interaction states (pass left / pass
right are both optimal), an MSE-trained MLP regresses to the mean of the two and
drives straight through, and a generative policy keeps the modes separate.

**Study 2 — neighbour encoder.** Which permutation-invariant neighbour encoder
(DeepSet / Transformer / GNN) gives the best decentralized collision avoidance, and
which generalizes best from its training fleet size to unseen fleet sizes?

Both use the same system (`unicycle2`, homogeneous fleet), the same expert (CasADi
MPC), the same decentralized observation schema, and the same DAgger loop.

---

## 2. Open decisions

Resolve these before phase 3; they change the grid sizes.

- [x] **Study 1 framing — decided 2026-09-12:** full 2x2
  `{mlp, flow} x {h=1, h=8}`, trained at N=4. See §6.
- [ ] **Study 2 train-fleet axis.** Is "does training fleet size matter" a question
  we want answered, or a nuisance axis to collapse to one training size so the
  budget goes to seeds instead? See §7.
- [ ] **Scenario set.** Does the density sweep (§4.3) get added? Depends on the
  outcome of the blind ablation in phase 1.

---

## 3. Shared setup

| | |
|---|---|
| System | `unicycle2`, homogeneous fleet via `multi_robot` |
| Expert | CasADi MPC, `horizon: 40`, `mode: mpc`, soft pairwise avoidance with `collision_slack_penalty_weight: 1e6` |
| Policy | decentralized: shared weights, ego observation + masked neighbour slots, actions combined only when stepping the simulator |
| Training | fresh-start DAgger ([learning/train_dagger.py](../learning/train_dagger.py)) |
| Safety radii | `d_safe: 1.2` (planner buffer), `d_collision: 1.0` (physical threshold) |
| Visibility | `inter_robot_visibility_radius: 4.0` |
| Success criterion | `pos_tol` 0.1 m, `theta_tol` 1.1 rad (wrapped), `vel_tol` 0.05 m/s, `omega_tol` 0.05 rad/s — all four simultaneously, per robot. Defined once as `TASK_TOLERANCES` in [generate_fleet_configs.py](../test/config/generate_fleet_configs.py) and emitted into every scenario config; policy configs must not override it |
| Eval fleet sizes | 2, 4, 6, 8, 16, 32 |
| Eval seed stream | `--seed-start 50000`, disjoint from training. **Identical across every policy** — this makes the whole evaluation paired; exploit it (§5.4) |

Existing infrastructure:

| Piece | File |
|---|---|
| Study driver (train + eval grid) | [run_study.sh](../run_study.sh) |
| Policy-only evaluation | [test/evaluate_scaling.py](../test/evaluate_scaling.py) |
| Policy-vs-expert evaluation (capped ~8 robots) | [test/evaluate_policy.py](../test/evaluate_policy.py) |
| Random fleet configs | [test/config/generate_fleet_configs.py](../test/config/generate_fleet_configs.py) |
| Antipodal ring configs | [test/config/generate_circle_configs.py](../test/config/generate_circle_configs.py) |
| Per-cell policy configs | [learning/config/study/generate_study_policy_configs.py](../learning/config/study/generate_study_policy_configs.py) |
| Plots | [test/plot_study_results.py](../test/plot_study_results.py) |

Known gaps in that infrastructure: no MSE-to-expert metric anywhere; `run_study.sh`
hardcodes `mlp_dagger_checkpoint.pt` and the MLP-only config templates (study 1
needs a policy-type axis); `evaluate_scaling.py` writes aggregates only; no
inference-time measurement; one training seed per cell.

---

## 4. Measurement problems to fix first

These four invalidate results if left alone. Phase 0 exists to clear them.

### 4.1 The circle scenario uses a different collision threshold than random

- [x] **Done 2026-09-12.** `d_collision: 1.0` added to all six circle configs by hand.
- [x] **Done 2026-09-12.** Both config generators were crashing on the current
  canonical template (it switched to the `{num_robots, system, config}` shorthand,
  while the generators indexed `template["robots"][0]`). Fixed via
  `first_robot_template` in
  [generate_fleet_configs.py](../test/config/generate_fleet_configs.py), reused by
  the circle generator. This is why the generated configs had silently frozen and
  never picked up `d_collision` from the template.
- [x] **Resolved 2026-09-12 — the tolerance belongs to the task, not the policy.**
  `TASK_TOLERANCES` in
  [generate_fleet_configs.py](../test/config/generate_fleet_configs.py) is now the
  single source, injected into every generated scenario config (random and circle),
  and `tolerance_overrides` is gone from the study policy configs and their
  generator. Rationale: the tolerance *defines* success, so it must be identical
  across every policy in a comparison and identical between training and
  evaluation; the scenario config is the file every policy shares, while a policy
  config is per-policy by construction. Regeneration is now safe and reproduces the
  committed configs byte for byte.

The random fleet configs set `d_collision: 1.0` against `d_safe: 1.2`. The circle
configs set no `d_collision` at all, and [core/config.py:760](../core/config.py#L760)
defaults it to `d_safe`. So on circle, a collision is declared at exactly the radius
the MPC's soft constraint drives to active. Every circle expert log says the same
thing:

```
Expert rollout collision (step=41, robots=(0,1), distance=1.200000, d_collision=1.200000)
```

That is a graze at the constraint boundary scored as a crash. **The "expert fails on
circle" observation is largely this artifact.** It also means circle and random
results are currently not comparable to each other.

There is a real problem underneath: a perfectly symmetric ring is degenerate for a
centralized MPC — the symmetric saddle has no tie-break. So even after the fix,
expect the nominal ring to be hard.

- [ ] Fix: promote circle from a single layout to a **family**. The jittered /
  heading-perturbed ring layouts that
  [generate_study_policy_configs.py](../learning/config/study/generate_study_policy_configs.py)
  already generates for *training* should also become *evaluation* layouts, so
  circle yields a ~30-episode success rate instead of a 1-episode coin flip.

### 4.2 `mean_goal_error_l2` is not angle-aware

- [x] **Done 2026-09-12.** [systems/goal_metrics.py](../systems/goal_metrics.py)
  splits the error per robot into position (m) and wrapped heading (rad), driven by
  each simulator's own `position_indices` / `angular_state_indices` rather than a
  hardcoded state layout. Wired into
  [evaluate_scaling.py](../test/evaluate_scaling.py) (CSV columns
  `mean_goal_position_error`, `mean_goal_heading_error`, replacing
  `mean_goal_error_l2`), [evaluate_policy.py](../test/evaluate_policy.py) and
  [plot_study_results.py](../test/plot_study_results.py). Verified: a 2*pi-wrapped
  heading on a 4-robot fleet now reports 0.0 where the raw L2 reported 12.57.

Dozens of rows in `outputs/study/circle/circle_scaling.csv` report
`success_rate=1.0` alongside `mean_goal_error_l2 ~ 6.25-6.28`. That is 2*pi: the
heading is correct but wrapped, and the raw L2 counts it. The metric is unusable as
a quality signal in its current form.

### 4.3 Aggregate-only records — decided: keep them

- [x] **Decided 2026-09-12: stay with one CSV row per (policy, fleet) cell.**
  Per-episode records were considered and rejected as too large a change.
- [ ] Fix instead: add the missing **aggregate** columns listed below.

Every policy is evaluated on the identical seed stream, so the design *could* have
been analysed as paired. Keeping aggregates gives that up. What it costs:

- No paired tests (McNemar / paired bootstrap). Comparisons use independent Wilson
  intervals, which need far more episodes — see §5.4.
- No re-deriving a metric after the fact; a new metric means re-simulating.
- No failure replay: the seed that collided is not recorded.

Extra **aggregate** columns remain cheap, since they are computed inside
`evaluate_fleet`'s existing loop. What was and was not added:

| Metric | Status |
|---|---|
| Self-describing rows: `pos_tol`, `theta_tol`, `vel_tol`, `omega_tol` | **added** |
| Per-robot success rate | added, then reverted — joint success rate is the reported metric (§4.4) |
| Near-miss rate | not added (deliberate) |
| Safety tail (`p05_min_pair_distance`) | not added (deliberate) |

### 4.4 The fleet-size axis currently measures conjunction, not degradation

- [x] **Decided 2026-09-12: joint success rate stays the reported metric.** A
  measured `mean_robots_at_goal_fraction` column was added and then removed; there
  is no per-robot column in the CSV.
- [ ] Fix: overlay the `p^N` null curve on joint-success plots. Needs no new column
  — it is derived from `success_rate` at plot time — and it is what keeps the
  fleet-size axis readable.

The conjunction effect below is therefore an **interpretation applied at write-up
time, not a measurement**. `success_rate^(1/N)` assumes robot outcomes are
independent, which a collision violates, so treat it as a null model to compare
against rather than as an estimate of per-robot reliability.

The random configs deliberately hold density roughly constant (the goal box grows as
sqrt(N/2); mean visible neighbours goes 0.50 at N=2 to 1.67 at N=16). Yet joint
success collapses. Pooling all 12 existing policies:

| N | joint success | implied per-robot `p = s^(1/N)` |
|---|---|---|
| 2 | 0.915 | 0.957 |
| 4 | 0.725 | 0.923 |
| 6 | 0.615 | 0.922 |
| 8 | 0.522 | 0.922 |
| 16 | 0.140 | 0.884 |
| 32 | 0.028 | 0.895 |

**Per-robot reliability is flat at ~0.92 from N=4 to N=32.** The scaling collapse in
the current headline plot is almost entirely `p^N`. That is a legitimate finding —
fleet reliability compounds — but it is not "the encoder degrades with fleet size",
which is how the plot reads today.

Consequences:

- Joint success stays as the deployment headline, with the `p^N` null overlaid so a
  reader can see whether a policy beats or trails pure conjunction.
- N=16 and N=32 are floor-saturated in joint success (0.00-0.06 for every policy)
  and carry no information about encoders. Keep them as a qualitative endpoint only.
- Because density is held constant, increasing N is **not** a hardness axis. If the
  encoders need to be separated, add a **density sweep**: fixed box, increasing N,
  so visible-neighbour count actually grows. That is the axis an encoder is for.

---

## 5. Metrics

### 5.1 Core set — every run, every scenario, both studies

| Metric | Definition / why |
|---|---|
| Joint success rate + Wilson CI | all robots at goal within tolerance, no collision, within step budget. Deployment headline |
| Failure decomposition | collision / timeout / deadlock. A timeout and a collision are different stories |
| Min pairwise distance | mean, 5th percentile, and **near-miss rate** (`d_collision < d < d_safe`) — separates grazes from hits |
| Terminal position error (m) | replaces the broken L2 |
| Terminal heading error (rad, wrapped) | ditto |
| Normalized makespan | steps for the last robot / straight-line lower bound. Works without an expert, so it extends to N=16/32; the circle generator already computes that bound |
| Control effort | integral of \|\|u\|\|^2, plus omega smoothness. MLP and flow should differ visibly here |

### 5.2 Study 1 additions

| Metric | Why |
|---|---|
| Open-loop action MSE on a held-out expert set | direct imitation fidelity — but see the caveat |
| **Closed-loop action divergence** | MSE(policy action, expert action queried at the policy's *own* visited state), along the policy rollout. Measures compounding distribution shift; this is what predicts rollout success |
| **Action multimodality** | sample k=16 actions per state from the flow policy, report spread. This is the mechanism under test |
| Inference wall-time per control step per robot; parameter count | flow runs 10 Euler steps per action ([flow_policy.py:134](../learning/models/flow_policy.py#L134)), roughly 10x the forward passes. Quantify the price |
| Expert MPC solve time on the same states | gives the distillation speedup headline, which nothing currently measures |

> **Caveat on MSE-to-expert.** The MLP is trained on exactly that objective
> ([mlp_policy.py:74](../learning/models/mlp_policy.py#L74)); the flow policy is
> trained on a velocity-field MSE and samples at inference. Action MSE therefore
> favours the MLP by construction, and a *lower* MSE is compatible with *worse*
> behaviour — mode-averaging has low MSE and drives into the other robot. Report it,
> never as the headline, always paired with the multimodality metric and closed-loop
> success. Never compare the two training losses directly; they are different
> objectives.

### 5.3 Study 2 additions

| Metric | Why |
|---|---|
| **Blind-policy ablation** (visibility radius 0, or neighbour branch zeroed) | establishes that the neighbour channel is used at all. See phase 1 |
| Encoder parameter count / forward-pass time | transformer and GNN cost more; is it bought back? |
| Optional: neighbour sensitivity | gradient of the action w.r.t. the nearest neighbour's features, or attention entropy. Mechanistic evidence the encoder attends to the right robot |

### 5.4 Statistics

- Analysis is **unpaired** (§4.3), so the episode budget is the binding constraint.
  At 80% power, two-sided 95%:

  | Difference to detect | Episodes per arm |
  |---|---|
  | 10 pp (0.72 vs 0.62) | 343 |
  | 10 pp (0.52 vs 0.42) | 387 |
  | 5 pp (0.72 vs 0.67) | 1327 |

- Budget **~350 episodes per cell** for any headline comparison. 50, the current
  value, resolves nothing: the marginal 95% interval is +-0.08-0.14, wider than the
  entire spread across all twelve existing policies.
- A 5 pp difference is out of reach at any realistic budget. Decide in advance
  whether the encoder question is worth answering if the true gap is that small.
- Prefer **3 seeds x 120 episodes** over **1 seed x 350**: same n, plus seed-level
  error bars, which is the variance component that actually matters.
- Treat any n=1 training run as an anecdote. The observed encoder spread today is
  fully consistent with initialization luck.

---

## 6. Study 1 — MLP vs flow

**Decided 2026-09-12: full 2x2 `{mlp, flow} x {h=1, h=8}`, trained at N=4.**

|  | h=1 | h=8 |
|---|---|---|
| **mlp** | cell A — the current baseline | cell C — the mode-averaging failure |
| **flow** | cell B — flow at the MLP's horizon | cell D — flow in its intended setting |

The 2x2 separates the architecture effect (A→B, C→D) from the horizon effect
(A→C, B→D) instead of confounding them, and cell C makes the hypothesis visible
rather than asserted: [generate_study_policy_configs.py](../learning/config/study/generate_study_policy_configs.py)
already documents the expectation — "a deterministic MSE-trained MLP regresses to
the conditional mean, so multi-step chunks average over distinct avoidance
manoeuvres". Cell C is the test of that sentence.

N=4 because the expert MPC dominates cost and grows super-quadratically in fleet
size; the checkpoint generalizes to any evaluation fleet size regardless, so the
training size mainly sets data cost. Do **not** cross policy with training fleet
size — that is study 2's axis.

### Confounds removed

The two shipped configs were not comparable as they stood:

| | [mlp config](../learning/config/multi_unicycle2_casadi_mlp_config.yaml) | [flow config](../learning/config/multi_unicycle2_casadi_flow_config.yaml) |
|---|---|---|
| `prediction_horizon` | 1 | 10 |
| `vel_tol` / `omega_tol` | 0.1 / 0.1 | 0.5 / 0.5 |
| `expert_mix_decay_after_eval_success` | 0.0 | 0.5 |

Horizon becomes a deliberate axis. Tolerances are no longer a policy-config concern
at all (§4.1). The DAgger schedule must be made identical across all four cells.

### What makes the cells comparable

**Decided 2026-09-12: every cell is trained through the DAgger pipeline.** There is
no separate offline-BC phase. The consequence has to be stated in the write-up:
each policy aggregates **its own** data, because with `beta < 1` the policy itself
drives part of every rollout, and a better policy visits different states. So a
success-rate gap between cells measures *the architecture plus the data that
architecture induced*, not the architecture alone. That is the deployable pipeline
and a legitimate thing to measure — it just cannot be reported as an isolated
statement about the policy head.

Everything else is held fixed. Verified against the code, not assumed:

| Requirement | Mechanism |
|---|---|
| Identical scenarios per round | identical `--round-seeds` **and** `--restart-round-seed`, so every cell samples the same initial states and goals in the same order. Only the trajectories differ, which is the irreducible part |
| Equal gradient steps | pin `--max-train-steps` — see the warning below |
| Identical DAgger schedule | same `--dagger-iterations`, `--trajectories-per-iteration`, `--steps-per-trajectory`, and the same beta schedule across all four cells |
| Identical encoder | confirmed by parameter count: 34,432 in both policies |
| Identical success criterion | `TASK_TOLERANCES` in the scenario config (§4.1) |
| Identical evaluation episodes | shared `--seed-start 50000` stream |
| One horizon axis, deliberately | the `VARIANTS` table, not the templates |

> **Pin `--max-train-steps`, and make it binding.** Episodes terminate early on
> success or collision, so a weaker policy collects *fewer frames per round*.
> `resolve_round_steps` is `ceil(epochs x frames / batch)`
> ([utils.py:81](../learning/dagger/utils.py#L81)), so under a fixed
> `target_epochs_per_round` the number of optimizer steps would silently track
> policy quality — the worse cell would also be trained less, and the two effects
> would be indistinguishable in the result. Set `target_epochs_per_round` high
> enough that the `--max-train-steps` cap binds for **every** cell, so all four get
> exactly the same number of gradient steps regardless of how much data they
> gathered. Confirm it bound by checking the `optimizer_steps=` line each run logs.

### What cannot be equalized — report, do not force

**Parameter count differs by ~8%** (real dimensions from the `deepset_n04`
checkpoint: state_dim 36, neighbor_feature_dim 8, neighbor_slots 3,
observation_horizon 2, hidden_dims 256x3):

| policy | horizon | encoder | head | total |
|---|---|---|---|---|
| mlp | 1 | 34,432 | 166,658 | 201,090 |
| flow | 1 | 34,432 | 183,554 | 217,986 |
| mlp | 8 | 34,432 | 170,256 | 204,688 |
| flow | 8 | 34,432 | 190,736 | 225,168 |

Flow's first layer takes `action_flat + obs_cond + time_embed` = 194 inputs against
the MLP's 128, which is 16,896 extra weights. Shrinking flow's hidden dims to match
would distort the architecture to chase an 8% difference that is not plausibly the
mechanism. Report both counts instead.

**The training losses are not comparable.** MLP minimises MSE(predicted action,
expert action) ([mlp_policy.py:74](../learning/models/mlp_policy.py#L74)); flow
minimises MSE(predicted velocity, target velocity field) at a random flow time
([flow_policy.py:122](../learning/models/flow_policy.py#L122)). Different
objectives, different scales. They cannot establish that both have converged.

### Prerequisites before any study 1 run

- [x] **Done 2026-09-12: flow configs exist.** Added
      `{deepset,transformer,gnn}_flow_config.yaml` templates and a `VARIANTS` table
      to [generate_study_policy_configs.py](../learning/config/study/generate_study_policy_configs.py).
      A variant is a policy head plus a prediction horizon — `mlp` (h=1), `flow`
      (h=8), `mlp_h8` (cell C), `flow_h1` (cell B) — emitted as
      `<encoder>_<label>_n<NN>_config.yaml`, 48 configs in all. The horizon lives in
      the variant table rather than the templates, so a template cannot silently
      disagree with the label it was generated under. `run_study.sh` resolves the
      checkpoint filename from the config's `policy_type` rather than the label,
      since `mlp_h8` is an mlp and `flow_h1` is a flow. Study 1's 2x2 is
      `ENCODERS="deepset" POLICIES="mlp flow_h1 mlp_h8 flow"`.
- [ ] **`torch.compile` asymmetry: left in place by decision (2026-09-12).**
      `FlowPolicy` compiles its network ([flow_policy.py:81](../learning/models/flow_policy.py#L81));
      `MLPPolicy` does not. No code was changed. This is a known, accepted confound
      on `mean_action_ms`, not an oversight — **read latency results with it in
      mind.** Benchmarked on CPU with real observations at fleet sizes 2/8/32:

      | policy | 2 | 8 | 32 |
      |---|---|---|---|
      | mlp h=1 | 0.91x | 0.94x | 1.04x |
      | flow h=8 | 0.86x | 0.81x | 0.84x |

      At ~200k parameters and batch sizes of 2-32 the wrapper's per-call overhead
      exceeds any fusion gain, and flow pays it once per Euler step rather than once
      per action. So the flow latency study 1 reports is roughly **15-19%
      pessimistic** relative to the same policy uncompiled, which biases the
      efficiency comparison against flow. It does not affect success rate, collision
      rate, or any behavioural metric. The trade-off may reverse on GPU or at larger
      model sizes.

- [ ] **No shared held-out validation metric exists.** Needed to show both policies
      plateaued. Use held-out action MSE on the frozen dataset — but only to detect
      plateau *within* one policy. It is biased toward the MLP by construction
      (that is the MLP's training objective, and mode-averaging scores well on it),
      so it must never rank across policies.

### Training

The whole 2x2 is one command. `run_study.sh` now carries a `SEEDS` axis and passes
`TARGET_EPOCHS` / `MAX_TRAIN_STEPS` through:

```bash
tmux new -s study1
MAX_PARALLEL=$(nproc) \
ENCODERS="deepset" \
POLICIES="mlp flow_h1 mlp_h8 flow" \
SEEDS="0 1 2" \
TARGET_EPOCHS=200 \
MAX_TRAIN_STEPS=4000 \
./run_study.sh train 4
```

12 runs, named `deepset_<variant>_n04_s<seed>`, landing in
`outputs/train_dagger_multi_robot/<name>/`. Progress in `logs/<name>.log`.

`TARGET_EPOCHS=200` with `MAX_TRAIN_STEPS=4000` makes the cap bind in every round:
at the default batch size of 64, the uncapped figure exceeds 4000 for any round
aggregating more than ~1,280 frames, and a round here gathers ~20,000. So every
cell trains for exactly 4000 steps per round regardless of how much data it
collected. **Verify from the `optimizer_steps=` line in each log** — if a run
reports fewer than 4000, the cap did not bind there and that cell was trained less
than the others.

Then evaluate, which needs only the checkpoints:

```bash
ENCODERS="deepset" POLICIES="mlp flow_h1 mlp_h8 flow" SEEDS="0 1 2" ./run_study.sh eval
ENCODERS="deepset" POLICIES="mlp flow_h1 mlp_h8 flow" SEEDS="0 1 2" ./run_study.sh eval circle
```

Results carry `train_seed`, the four tolerances, and `mean_action_ms`, so a row says
what produced it.

### Grid and cost

Unlike a shared-dataset design, **every run now pays the full expert cost**, since
DAgger queries the MPC for a label at every step of every rollout regardless of who
executes the action. Measured on this machine: **321 ms per expert step at N=4**,
so a 200-step episode is ~64 s single-core and a 100-episode round is ~107 min.
Episodes terminate early (mean ~130 steps), so in practice expect roughly

| | per run | 4 cells x 3 seeds | 4 cells x 5 seeds |
|---|---|---|---|
| single core | ~3.5 h | ~42 h | ~70 h |
| 8-way parallel | — | ~5-6 h | ~9 h |

Three seeds is the realistic default; five is an overnight run. Decide before
starting, because adding seeds later means re-running everything for consistency.

---

## 7. Study 2 — encoders

Keep the existing grid, with three changes:

1. **Cut the training-fleet axis from 4 levels to 2** (N=2 and N=8, the extremes).
   The question is about the *evaluation* fleet axis; train size is the
   generalization sub-question and two levels answer it.
2. **Spend the freed budget on 3 training seeds per cell.** 3 encoders x 2 train
   sizes x 3 seeds = 18 runs, versus the current 12 at n=1. At n=1, encoder effect
   and initialization luck are not separable.
3. **Drop N=32 from the discriminative analysis** (floor-saturated at joint success
   0.00-0.06 for every policy). Keep it as a qualitative endpoint.

---

## 8. Expert baseline

The expert is not 100%, and on circle currently reports 0% for artifactual reasons
(§4.1). Every result needs an expert reference on the **identical seeds**:

- **N <= 8:** run the expert on the same seed stream; report **relative success**
  (policy / expert) and **relative makespan**. This normalizes out scenario
  difficulty and is the honest form of "what did distillation cost".
- **N >= 16:** the MPC is impractical
  ([evaluate_scaling.py](../test/evaluate_scaling.py) exists for exactly this
  reason). Use the straight-line step bound as the denominator instead, and state
  clearly that the reference changes.
- Report expert solve time per step against policy inference time. This is the point
  of the project and nothing currently measures it.

---

## 9. Execution order

| Phase | Work | Cost | Status |
|---|---|---|---|
| **0. Instrument** | `d_collision` in circle configs [x]; wrapped angle error [x]; config generators unbroken [x]; tolerance single-source [x]; per-episode CSV [x] (decided against, §4.3); aggregate columns per §4.3 [ ]; per-step inference timing [ ]; jittered ring eval layouts [ ]; policy-type axis in `run_study.sh` [ ] | none | in progress |
| **1. Characterize** | Expert on all scenarios, same seeds, N<=8. **Blind-policy ablation.** Answers: is circle solvable after the fix, and does the neighbour channel matter at all | hours | [ ] |
| **2. Decide** | If blind ~= encoders, add the density-sweep scenario (§4.4) before study 2. Close the open decisions in §2 | none | [ ] |
| **3. Study 1** | Frozen dataset -> BC, 2 policies x 5 seeds; then DAgger, 2 policies x 3 seeds | moderate | [ ] |
| **4. Study 2** | 3 encoders x 2 train sizes x 3 seeds | largest | [ ] |
| **5. Evaluate** | All checkpoints x {random, circle-family, density} x fleet sizes, 100 episodes/cell, shared seed stream | moderate | [ ] |
| **6. Report** | Joint success-rate plots with the `p^N` null overlay; Wilson intervals | none | [ ] |

**Phase 1 is the one to start with.** It is cheap and it can invalidate the premise
of phase 4 before that compute is spent: if the blind policy matches the encoders,
the conclusion is "the task as configured does not require neighbour reasoning", and
no number of seeds changes that.

---

## 10. Progress log

Newest entries at the top. One line per run batch or decision: what was run, what
came out, what it changed.

| Date | Entry |
|---|---|
| 2026-09-12 | **Study 1 drops the offline-BC phase: all four cells train through the DAgger pipeline.** Cost measured at 321 ms/expert step at N=4, so ~3.5 h per run and ~5-6 h wall clock for 4 cells x 3 seeds at 8-way parallelism. Comparability now rests on identical round seeds, an identical DAgger schedule and a pinned `--max-train-steps` rather than on shared data; the write-up must say that each cell induced its own dataset. Logged the frames-per-round trap: episodes end early, so a weaker cell collects less data and, under fixed `target_epochs_per_round`, would also be trained less. |
| 2026-09-12 | Study 1 prerequisites 1 and 2 cleared. Flow templates + a `VARIANTS` table (mlp, flow, mlp_h8, flow_h1) now generate all 48 study policy configs, and `run_study.sh` derives the checkpoint name from `policy_type` rather than the variant label. `torch.compile` was left exactly as it is (flow compiles, mlp does not) by decision; benchmarking showed compiling costs flow 15-19% at this model size, so the flow latency numbers are pessimistic by about that much — recorded in §6 as an accepted confound rather than fixed. Existing mlp configs are semantically unchanged (model block and ring rollouts identical to HEAD; only the removed `tolerance_overrides` differs). Remaining prerequisite: a shared held-out validation metric for plateau detection. |
| 2026-09-12 | **Study 1 design settled: full 2x2 `{mlp, flow} x {h=1, h=8}` at N=4**, Phase A as offline BC on one frozen dataset (4 cells x 5 seeds = 20 runs), Phase B as DAgger on the two best-setting cells (2 x 3 = 6 runs). Verified against the code that identical data, a single dataset across both horizons, and exactly equal gradient steps are all supported. Three prerequisites logged in §6: study flow configs do not exist yet, `FlowPolicy` compiles its net while `MLPPolicy` does not (which would confound the new `mean_action_ms` column), and no shared held-out validation metric exists for plateau detection. |
| 2026-09-12 | Reverted `mean_robots_at_goal_fraction`: **joint success rate stays the reported metric.** The four tolerance columns stay. §4.4's conjunction point is now handled purely as a plot-time `p^N` null overlay derived from `success_rate` — no extra column, no independence assumption baked into the stored data. |
| 2026-09-12 | Added `mean_robots_at_goal_fraction` and the four tolerance columns to `evaluate_scaling.py`. Near-miss rate and safety tail deliberately omitted. Smoke test on `deepset_n04` showed the measured per-robot fraction (0.59 at N=8) and the implied `s^(1/N)` (0.89) are different quantities, the measured one depressed by collision truncation. Superseded same day, see above. |
| 2026-09-12 | Decided to keep the aggregate CSV (one row per policy/fleet cell) rather than move to per-episode rows. Consequence: analysis is unpaired, so the episode budget rises to ~350 per cell for a 10 pp difference (343-387 by the power calculation) and a 5 pp difference is unreachable. §4.4's conjunction point survives as a plot-time overlay. |
| 2026-09-12 | **Tolerance ownership resolved.** Root cause of the drift was not where the values were written but who reads them: `tolerance_overrides` is applied by `train_dagger.py`, `evaluate_policy.py` and `plot_expert_trajectories.py`, but `test/evaluate_scaling.py` has no such flag and reads the system config only. So the 12 study policies were **trained** at `theta_tol` 0.78 / `vel_tol` 0.1 / `omega_tol` 0.1 and **scored** at 1.1 / 0.05 / 0.05 — three of four criteria differed, evaluation being 2x stricter on both velocity terms. Fixed by making the scenario config the single source and deleting `tolerance_overrides` from the study policy configs. Verified: training and evaluation now resolve to identical values, and both generators reproduce the committed configs byte for byte. Caution note added to the README beside `tolerance_overrides`. **Another reason the existing 12 checkpoints' results are exploratory only** — they were trained against a different success criterion than they were scored on. |
| 2026-09-12 | **Circle is solvable by the expert after the `d_collision` fix.** Expert rolled out on the deterministic ring, `--use-config-start --num-traj 1 --num-steps 400`: N=2, 4, 6 and 8 all reach `goal reached in 1/1` for every robot, with `min_dist=1.2000` and `violations=0` against `d_collision=1.0` (0/136, 0/870, 0/2295, 0/4592 pairwise checks respectively). The min distance sits exactly on `d_safe` at every fleet size, confirming the MPC drives the soft constraint active and that the earlier 100% "expert collision rate" was entirely the threshold artifact. The symmetric ring is *not* degenerate for this expert up to N=8; N=16/32 untested (MPC cost). Logs and plots in `outputs/plots/circle_expert_fixed/`. |
| 2026-09-12 | Phase 0 partial: angle-wrap fix landed (§4.2); both config generators repaired (§4.1); regeneration blocked on tolerance drift (§4.1). |
| 2026-09-12 | Plan written. Existing assets: 12 MLP checkpoints (3 encoders x 4 train sizes, n=1 seed), `outputs/study/encoder_scaling.csv` (random, 50 ep/cell) and `outputs/study/circle/circle_scaling.csv` (1 ep/cell). Both predate the fixes in §4, so they are exploratory only. Identified the circle `d_collision` artifact, the 2*pi heading-error artifact, and the `p^N` conjunction effect. |
