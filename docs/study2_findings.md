# Study 2 — findings

Analysis of `outputs/data_mid_best` (MLP head) and `outputs/data_mid_flow` (flow head)
against the two research questions. Experiment design: [study2_encoders.md](study2_encoders.md).

Everything here is **one training seed per cell**. Nothing below separates an encoder
effect from training noise; the numbers say what the runs did, not what the encoders do.

---

## RQ1 — How does the in-training fleet size scale to other fleet sizes?

### What `mean_min_pair_distance` actually measures

`d_collision = 1.0 m` in every circle config. The column is the episode's *minimum over
time* of the closest pair, averaged over episodes
(`test/evaluate_scaling.py:389,431,473`). On `circle_det` it agrees with the collision
flag exactly — in all 192 rows, `mean_min_pair_distance < 1.0` ⟺ `collision_rate > 0`.

So it is a **continuous relaxation of the binary collision outcome**, not an independent
safety signal. That is exactly why it is the right graded metric on `circle_det`, where
one deterministic episode per cell makes success pass/fail and carries almost no
information.

**But it only means anything when the fleet attempts the swap.** A fleet that never
leaves its start ring keeps its starting separation, which at the training density is
1.91 m at N=16 and 1.36 m at N=32 — both comfortably above `d_collision`. Standing still
scores as maximally safe.

Progress below is `1 - mean_goal_position_error / 2R`, where `2R` is the ring diameter and
therefore the exact initial goal distance of an antipodal swap. 1.0 = swapped, 0.0 = never
moved.

### MLP: the trend is real on the ring — but only on the ring

Row means of `mean_min_pair_distance` over eval N = 3..16, and the fraction of ordered
train-size pairs where the larger training fleet keeps more clearance:

| Encoder | train N=2 | N=4 | N=6 | N=8 | concordance |
| :-- | --: | --: | --: | --: | --: |
| DeepSet | 0.70 | 0.68 | 0.89 | 1.06 | 76% |
| GNN | 0.34 | 0.50 | 0.75 | 1.01 | 88% |
| Transformer | 0.54 | 0.64 | 0.94 | 1.15 | 87% |

Your reading holds. Two things make it trustworthy:

- **No contaminated cells.** All 192 MLP cells completed the swap — progress ≥ 0.86, median
  0.99, and only two cells below 0.92. Every comparison is between fleets that all did the
  task and differ only in clearance.
- **The training budget works against it.** DeepSet's N=2 checkpoint had 1.27 M optimizer
  steps and scores 0.70 m; its N=6 checkpoint had 443 k and scores 0.89 m. The trend
  survives a budget gradient pointing the other way (see RQ2 for why budgets differ).

One caveat: the **eval N=2 column is erratic** and should not be read as
"downward generalization". Transformer trained at N=8 scores 2.72 m there — the highest
number in the table — with a 100% timeout. Large-fleet policies fail at N=2 by refusing to
settle, not by colliding, and the clearance metric reads that failure as safety.

### Flow: the same trend, hidden by a broken metric

The flow table looks unclear because **34 of its 192 cells never did the task**, and
29 of those 34 are the train-N=2 row. Those cells then top the clearance table:

| Cell | clearance | outcome | progress |
| :-- | --: | :-- | --: |
| DeepSet train N=2, eval 11–16 | **1.75 – 1.95 m** | 100% timeout | −7% … +4% |
| GNN train N=2, eval 10–16 | **1.22 – 1.40 m** | 70–100% timeout | −24% … +1% |

These are the largest clearances anywhere in the flow data, produced by fleets that burned
100% of the step budget and finished 8.3–10.3 m from goals they started 7.7–9.8 m from —
several of them *further* from their goals than when they set off, hence the negative
progress. The raw row means are therefore meaningless: DeepSet's train-N=2 row averages
0.95 m, the best of its four rows, purely on the strength of not moving.

Restrict the comparison to eval sizes where **all four** train rows completed the task and
the trend appears, at the same strength as the MLP's:

| Encoder | eval sizes compared | train N=2 | N=4 | N=6 | N=8 | concordance |
| :-- | :-- | --: | --: | --: | --: | --: |
| DeepSet | 3–8 | 0.22 | 0.33 | 0.32 | **0.83** | 81% |
| GNN | 3–5 | 0.38 | 0.47 | 0.31 | **0.85** | 78% |
| Transformer | 3–9 | 0.10 | 0.32 | 0.78 | **0.85** | 83% |

**Both heads give the same answer: a policy trained on a larger fleet generalizes better.**
The flow table only looked ambiguous because its metric rewards standing still.

### Flow has a second failure mode the MLP does not

Where each policy stops attempting the task (progress < 0.70):

| Head | train N=2 | N=4 | N=6 | N=8 |
| :-- | :-- | :-- | :-- | :-- |
| Flow, DeepSet | eval ≥ 9 | only N=32 | never | never |
| Flow, GNN | eval ≥ 6 | only N=32 | only N=32 | only N=32 |
| Flow, Transformer | eval ≥ 10 | only N=32 | never | never |
| MLP, all encoders | never | never | never | never |

A small-fleet MLP policy degrades continuously — it keeps driving and starts colliding. A
small-fleet flow policy **abandons the task outright** past a threshold fleet size. So the
in-training fleet size matters *more* for flow, not less; it just shows up in completion
rather than in clearance.

### The scenarios disagree — and that is the actual finding

`circle_det` says bigger training fleets generalize better. **`arena` says the training
fleet size has no effect on safety at all.** On arena the clearance concordance is
33 / 39 / 33% — *below* chance — and both clearance and collision rate are flat across
train N to two decimals:

| MLP arena, eval N ≤ 4 | train N=2 | N=4 | N=6 | N=8 |
| :-- | --: | --: | --: | --: |
| collision rate | 0.297 | 0.297 | 0.301 | **0.308** |
| timeout rate | 0.091 | 0.220 | 0.493 | **0.460** |
| success | 0.612 | 0.482 | 0.206 | 0.232 |

Training fleet size changes *nothing* about collisions there and only adds timeouts.

**What reconciles them is the visible-neighbour count, not the fleet size.** Every training
run uses the same density (0.1667 robots/m², `learning/config/study/data_mid_n*/`), so the
training fleet size fixes the neighbour count the encoder was fitted to: ≈1.0 / 2.5 / 3.5 /
4.3 for N = 2 / 4 / 6 / 8. Sorting all 360 MLP cells (arena + circle_det + density) by how
far the evaluated neighbour count sits from the trained one:

| Regime | cells | success | collision | timeout |
| :-- | --: | --: | --: | --: |
| Sees **fewer** neighbours than trained | 131 | 0.414 | 0.47 | **0.227** |
| **Matched** (±0.5) | 62 | **0.563** | 0.49 | 0.041 |
| Sees **more** neighbours than trained | 167 | 0.065 | **0.973** | 0.001 |

Two distinct failure modes: **surplus → collisions, deficit → timeouts.** That is exactly
the disagreement:

- `circle` pins density at the training density, so eval N > train N is pure *surplus*.
  Bigger training fleets are further from the cliff, so they win monotonically.
- `arena` holds the box fixed, so small N means low density — at eval N=2 the mean visible
  neighbour count is **0.25**, below even the N=2 policy's ≈1.0. Every policy is in
  *deficit*, the deficit grows with training fleet size, and the cost is timeouts.

So "a policy trained on a bigger fleet performs better" holds **only at the training density
and at fleet sizes at or above the training fleet.** It is not a general scaling law.

Upward extrapolation is also hard-bounded: past ≈ +3 neighbours over training, success is
≤ 0.02 and collision ≈ 1.00 for every encoder and both heads.

### Consequence for how RQ1 should be reported

Clearance alone cannot carry this claim. Either report completion/`robot_success_rate` as
primary with clearance as secondary, or gate clearance on a progress threshold. The pooled
"mean closest pair distance" tables in `outputs/data_mid_flow/tables/` inherit the same
artifact — DeepSet's pooled clearance rises from 0.29 m at N=8 to 0.69 m at N=11, which is
the stall leaking through.

---

## RQ2 — Which encoder works best for variable observations?

### Your MLP observation, quantified

`robot_success_rate` on `arena` (200 episodes/cell), averaged over the six eval sizes:

| Encoder | train N=2 | N=4 | N=6 | N=8 | mean | range | CV |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| DeepSet | 0.455 | 0.257 | 0.040 | 0.013 | 0.191 | 0.443 | 1.08 |
| GNN | 0.455 | 0.450 | *0.207* | 0.407 | 0.380 | 0.248 | 0.31 |
| Transformer | 0.426 | 0.433 | 0.393 | 0.226 | 0.370 | 0.206 | 0.26 |

Exactly as you described. Excluding the GNN N=6 anomaly its range collapses to **0.048**
(0.455 / 0.450 / 0.407) — essentially flat. Transformer is flat across N=2,4,6
(range 0.039) and drops only at N=8. DeepSet falls 35× from N=2 to N=8.

### The confound: `data_mid_best` is not budget-controlled

The 12 MLP checkpoints were curated from **different DAgger rounds**:

| Encoder | train N=2 | N=4 | N=6 | N=8 |
| :-- | :-- | :-- | :-- | :-- |
| DeepSet | r4 — 1,267,597 | r3 — 815,926 | r2 — 443,109 | r3 — 698,490 |
| GNN | r0 — **81,685** | r3 — 807,721 | r0 — **72,120** | r1 — 203,640 |
| Transformer | r0 — **81,674** | r4 — 1,212,321 | r4 — 1,088,469 | r4 — 1,027,690 |

That is an **18× spread in optimizer steps**. `data_mid_flow` has no such problem: all 12
runs are round 4 at 1.18–1.31 M steps (±5%).

Hashing each `data_mid_best` checkpoint against the `outputs/data_mid/models/*/mlp_dagger_iter_*.pt`
files it came from shows **only 5 of 12 are the argmax round**, and only 6 of 12 carry the
best round's in-training eval success. That selection ran on a 20-episode eval whose
standard error is ≈ ±10 pp, so the differences it selected on are mostly noise.

What this does and does not break:

- **RQ1 survives.** Its trend is *within* an encoder and runs against the budget gradient.
- **RQ2's DeepSet result is compromised.** DeepSet's collapse tracks its own budget at
  **r = 0.92** (1.27 M → 0.455, 816 k → 0.257, 698 k → 0.013, 443 k → 0.040). Across all
  12 runs r is only 0.11, so budget does not explain the whole ranking — but for DeepSet
  specifically it cannot be ruled out.
- **The GNN result is strengthened in one respect**: GNN reaches 0.455 at N=2 on 82 k steps,
  the same score DeepSet needs 1.27 M steps for — 15× more. That is a genuine
  sample-efficiency gap, and the budget asymmetry favours DeepSet.

### The GNN N=6 anomaly has a mechanical cause

`gnn_mlp_unicycle2_fleet_06_s0` is **the least-trained checkpoint in the entire study**:
round 0, 72,120 optimizer steps — pure behaviour cloning, before any DAgger aggregation.
Its own training summary records round 4 at 0.70 against round 0 at 0.60, so the copied
checkpoint is not even that run's best round. Treating it as an anomaly is correct, and the
cause is checkpoint curation, not the encoder.

### Under equal budget the ranking changes

`data_mid_flow` is the only budget-controlled comparison in the study. Same `arena` metric:

| Encoder | train N=2 | N=4 | N=6 | N=8 | mean | range | CV |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| DeepSet | 0.318 | 0.391 | 0.381 | 0.222 | 0.328 | 0.169 | 0.24 |
| GNN | 0.175 | 0.383 | 0.414 | 0.365 | 0.335 | 0.239 | 0.32 |
| Transformer | 0.308 | 0.417 | 0.357 | 0.359 | **0.360** | **0.109** | **0.12** |

Three things do not carry over from the MLP table:

1. **Transformer is now both the best and the most stable** across in-training fleet sizes.
2. **GNN's flatness does not reproduce.** Its train-N=2 run is its worst cell (0.175), the
   lowest single entry in the flow table.
3. **DeepSet's large gap disappears** on `arena` (0.328 vs 0.335 / 0.360). It persists only
   on the hard head-on ring — `circle`: DeepSet 0.057, GNN 0.128, Transformer 0.146.

### Scope limit: the question is only answerable for N ≤ 8

Pooled `robot_success_rate` at large fleets:

| Scenario | eval N ≥ 16, DeepSet | GNN | Transformer |
| :-- | --: | --: | --: |
| arena | 0.078 | 0.070 | 0.089 |
| circle | 0.002 | 0.005 | 0.001 |
| swap | 0.002 | 0.000 | 0.000 |

Everything is at or near zero on the ring and the corner swap past N=16. Only `arena` —
the fixed-workspace scenario — retains anything, and 0.07–0.09 is not a working policy. No
encoder claim can be made in the extrapolation regime; the comparison lives at N ≤ 8.

### No encoder buys extrapolation

Split by the same neighbour-count regimes, `robot_success_rate` per encoder:

| Regime | DeepSet | GNN | Transformer |
| :-- | --: | --: | --: |
| MLP — fewer than trained | 0.173 | **0.557** | 0.524 |
| MLP — matched | 0.478 | **0.648** | 0.594 |
| MLP — more (+0.5 … +3) | 0.096 | 0.106 | 0.145 |
| MLP — far more (> +3) | 0.016 | 0.014 | 0.012 |
| Flow — fewer than trained | 0.154 | **0.237** | 0.228 |
| Flow — matched | 0.122 | **0.157** | 0.150 |
| Flow — more (+0.5 … +3) | 0.013 | 0.011 | 0.014 |
| Flow — far more (> +3) | 0.011 | 0.003 | 0.008 |

**Once the neighbour count exceeds what a policy trained on, the three encoders are
indistinguishable.** The encoder choice buys nothing in the regime the research question is
really about. They differ only *inside* the trained range — and there the ordering is
consistent across both heads: **GNN ≈ Transformer > DeepSet**.

### Provisional answer

**Transformer**, on the only budget-controlled evidence available: best mean
(0.360) and by far the most stable across in-training fleet size (CV 0.12 vs 0.24 / 0.32).
GNN is competitive when trained at N ≥ 4 and the most sample-efficient of the three, but
its N=2 flow run fails. DeepSet is consistently worst in every regime in **both** tables,
which is the one encoder claim that survives the budget confound.

GNN carries a known structural risk its own config flags: it aggregates with a **sum**, so
its context magnitude grows with the neighbour count. The MLP ring data matches that
prediction — GNN's clearance at eval N = 16/32 is 0.107 m against DeepSet's 0.406 m and
Transformer's 0.279 m — but it does not reproduce in the flow table, so treat it as
suggestive, not established.

Cost cuts the other way: Transformer is ~3× the latency at N=32 (6.81 ms per control step
on arena, vs DeepSet 2.36 ms and GNN 2.05 ms). If large-fleet latency binds, GNN is the
better trade.

---

## What would settle either question

1. **Re-evaluate the MLP grid from a consistent round** — round 4 for all 12, or a genuine
   argmax chosen on an eval large enough to resolve it. Until then no cross-encoder claim
   from `data_mid_best` is safe.
2. **Gate clearance on progress**, or drop it in favour of completion, wherever a stalling
   policy can appear.
3. **Three seeds**, as [study2_encoders.md](study2_encoders.md#limits-and-caveats) already
   notes.
