# Study 2 — findings

Results from `outputs/data_mid_best` (MLP head) and `outputs/data_mid_flow` (flow head).
Experiment design: [study2_encoders.md](study2_encoders.md).

## Headline

1. **You do not need a large training fleet.** A policy trained on 4 robots beats one
   trained on 8 at *every* fleet size tested, in *both* policy heads, with **zero
   counterexamples in 36 cells** — while costing 62% as much to train.
2. **The neighbour encoder does not matter.** Under the only budget-controlled comparison,
   DeepSet, GNN and Transformer land within 2 percentage points of each other, each wins
   about a third of the cells, and the whole encoder effect is **the same size as the noise
   from re-running the same training**.

Both claims are *negative or cost claims*, which is what this data can actually support:
they pool over all 12 runs per head instead of resting on any single cell. The absolute
numbers are low throughout — no configuration produces a usable policy above ~6 robots —
so these are statements about relative cost and design levers, not about a working system.

---

## Finding 1 — training on 4 robots dominates training on 8

### The evidence

Per-robot success on `arena`, pooled over the three encoders, 200 episodes per cell:

| eval N | train N=2 | train N=4 | train N=6 | train N=8 |
| --: | --: | --: | --: | --: |
| **MLP** | | | | |
| 2 | 0.872 | 0.773 | 0.362 | 0.461 |
| 4 | 0.662 | 0.555 | 0.312 | 0.315 |
| 6 | 0.495 | 0.414 | 0.249 | 0.228 |
| 8 | 0.420 | 0.344 | 0.218 | 0.187 |
| 16 | 0.171 | 0.150 | 0.105 | 0.078 |
| 32 | 0.052 | 0.043 | 0.033 | 0.023 |
| **Flow** | | | | |
| 2 | 0.512 | 0.774 | 0.809 | 0.698 |
| 4 | 0.373 | 0.594 | 0.584 | 0.447 |
| 6 | 0.302 | 0.458 | 0.427 | 0.345 |
| 8 | 0.262 | 0.366 | 0.327 | 0.272 |
| 16 | 0.121 | 0.154 | 0.128 | 0.104 |
| 32 | 0.032 | 0.036 | 0.028 | 0.028 |

Counting every (head × encoder × eval N) cell and sign-testing the direction — a cell-level
test, so it is not inflated by treating correlated robots within an episode as independent:

| Comparison | wins | ties | losses | sign test |
| :-- | --: | --: | --: | --: |
| **train 4 vs train 8** | **31** | 5 | **0** | **p = 4.7 × 10⁻¹⁰** |
| train 4 vs train 6 | 24 | 7 | 5 | p = 2.7 × 10⁻⁴ |
| train 2 vs train 8 | 23 | 3 | 10 | p = 0.018 |
| train 2 vs train 4 | 10 | 7 | 19 | p = 0.97 (train 4 wins) |

At episode level, pooled over encoders for eval N ≤ 8: MLP 0.267 vs 0.124 (z = +12.5,
p = 8 × 10⁻³⁶), Flow 0.263 vs 0.216 (z = +3.8, p = 1.5 × 10⁻⁴). Both heads, same direction.

**Training on more robots did not buy generalization to more robots. It cost it.**

### It is also 1.6× cheaper

Training cost is dominated by the CasADi expert, which runs at every step and scales
super-quadratically in the fleet size (`learning/config/study/data_mid_n*/` headers):

| train N | cost per run | episode success (eval N ≤ 8) | success per training hour |
| --: | --: | --: | --: |
| 2 | 14.7 h | 0.339 (MLP) / 0.115 (flow) | 2.30 / 0.79 |
| **4** | **21.7 h** | **0.267 / 0.263** | **1.23 / 1.21** |
| 6 | 25.8 h | 0.109 / 0.291 | 0.42 / 1.13 |
| 8 | 34.8 h | 0.124 / 0.216 | 0.36 / 0.62 |

Against train N=8, train N=4 delivers **2.15× the success at 0.62× the cost (3.5× the value
per hour) for the MLP head, and 1.22× at 0.62× (2.0× per hour) for flow.**

### Why N=2 is too small — the neighbour cap

Training at N=2 is cheapest and wins outright for the MLP head, but it is the **worst**
choice for flow (0.115 vs 0.263 for N=4). The reason is a hard ceiling on what the encoder
can ever be shown: **the training fleet size caps the neighbour count at N − 1**, and at
these workspace sizes that cap is what binds.

Measured on the training distribution itself — the sampling box each run actually used
(`learning/config/study/data_mid_n*/`, which overrides the study config's bounds to reach
0.1667 robots/m²), with `test/training_neighbour_counts.py`:

| train N | training box | ceiling (N−1) | uniform starts | ring starts (⅓ of episodes) | **training mix** |
| --: | --: | --: | --: | --: | --: |
| 2 | ±1.732 | 1 | 0.99 | 1.00 | **0.99** |
| 4 | ±2.45 | 3 | 2.32 | 2.15 | **2.27** |
| 6 | ±3.0 | 5 | 3.06 | 2.69 | **2.93** |
| 8 | ±3.464 | 7 | 3.56 | 2.99 | **3.37** |

Note what does *not* apply here: the infinite-domain estimate `density · πR²` gives 8.4
neighbours at this density, but a disk of radius 4 m covers 50 m² while the N=8 training box
is only 48 m². The visibility radius spans the whole workspace at every training fleet size,
so the count is `(N − 1) ×` the chance a pair falls within 4 m, and it grows sub-linearly:
tripling from N=2 to N=8 while N−1 rises sevenfold.

A policy trained at N=2 has **never received an input with more than one neighbour
populated**, no matter how dense its training world was. N=4 is the smallest fleet that
exercises the aggregation at all.

The cap itself is arithmetic. That it is *why* flow's N=2 runs fail is a hypothesis this
data does not test — an equally good candidate is that those runs stall outright (see
[Metric caveat](#metric-caveat-clearance-rewards-standing-still)), which is a different
failure from encoding a neighbour set badly. Either way the recommendation is the same,
and it rests on the measured result rather than on the mechanism: **N=4 is the smallest
training fleet that is never worse than N=8.**

### Where this does not hold

On `circle` — an antipodal swap where every robot is aimed at the opposite side of the ring
— the ordering reverses: bigger training fleets win (MLP per-robot success 0.02 / 0.06 /
0.12 / 0.26 for train N = 2/4/6/8). Figure: `outputs/plots/reversal/reversal_mlp.pdf`,
from `test/plot_reversal_explanation.py`.

**Why: a bigger training fleet buys caution, and caution only pays where collisions are the
binding constraint.** One policy property, two opposite consequences.

The property is scenario-independent — the share of the step budget a policy consumes rises
with its training fleet size in both scenarios (MLP: arena 0.51 → 0.68 → 0.89 → 0.89, ring
0.51 → 0.64 → 0.67 → 0.68). What differs is what that hesitancy meets:

| Across train N = 2 → 8 | arena | circle |
| :-- | --: | --: |
| collision rate | **+0.01** (flat) | **−0.38** |
| timeout rate | **+0.15** | +0.06 |
| clearance | 0.94 / 0.99 / 0.91 / 0.91 (flat) | 0.55 / 0.60 / 0.81 / 1.06 |

- `arena` draws starts and goals at random, so most robots never conflict. Caution buys
  **no** collision reduction — the rate moves 0.01 across the whole training range, and
  clearance does not move either — while timeouts climb. Hesitancy is pure cost, so the
  smallest training fleet wins.
- `circle` puts all N paths through the centre at once: a symmetric head-on conflict that
  cannot be driven straight through. Yielding *is* the solution, collisions fall by 0.38 and
  timeouts stay near zero, so the largest training fleet wins.

**The competing explanation — that a policy must be evaluated at the neighbour count it
trained at — is refuted by the density sweep.** At N=6 and 1.25× the training density the
visible-neighbour count is 3.96, at or above what the N=6 and N=8 policies trained on
(2.93 and 3.37); the N=2 policy nevertheless wins by better than 2× (0.247 against 0.111 and
0.114), and it wins at *every* density level tested. Neighbour matching is correlated with
the outcome but is not the lever; caution is, and the training fleet size sets it.

**The other competing explanation — that the ring reversal is training-set contamination —
is confirmed only as a local effect, and is refuted as the cause of the trend.**

A third of every training round starts from an antipodal ring layout, and the ring radius
range is tied to the workspace half-width, which is tied to N at constant density. So the
eval circle at N = train N *is* the largest ring in that policy's training distribution
(training radii top out at 1.99 / 2.45 / 3.00 / 3.25 m for N = 2/4/6/8, against eval circle
radii of 1.73 / 2.45 / 3.00 / 3.46 m). Contamination is therefore real and measurable:

- **The diagonal is memorised.** Diagonal cells beat the mean of their two adjacent eval
  fleet sizes by **+0.33 (MLP)** and **+0.23 (flow)**; all 12 MLP diagonal cells score
  exactly 1.000. This is the caveat `study2_encoders.md` already records, now quantified.

But it cannot produce the row trend, on three counts:

1. The diagonal is 1 of 16 cells per row. Dropping it leaves the correlation between
   training fleet size and per-robot success at **+0.36 (MLP)** and **+0.30 (flow)**.
2. The stronger version of the hypothesis — that bigger training fleets practise *tighter*
   rings, which is true by construction (a ring of N robots at radius R has closest pair
   `2R·sin(π/N)`, so the median trained spacing falls **3.66 → 3.04 → 2.47 → 2.01 m** as
   train N goes 2 → 8) — makes a sharp prediction that fails. Train N=2 never saw a pair
   closer than 2.75 m, so it should break between eval N=7 (2.81 m) and N=8 (2.65 m).
   **It actually collapses at eval N=4 (3.46 m), well inside its trained range.** Spacing
   coverage is not the lever.
3. **`swap` settles it.** It is a symmetric head-on conflict — two corner stacks packed at
   d_safe, all paths crossing the origin — whose geometry appears nowhere in training (the
   training layouts are all rings: nominal, jitter, ellipse, heading variants) and whose
   density is 0.0555 against training's 0.1667. If ring content drove the reversal, `swap`
   should behave like `arena`. It behaves like `circle`:

| Flow, per-robot success, pooled over encoders | train 2 | train 4 | train 6 | train 8 |
| :-- | --: | --: | --: | --: |
| `arena` (random assignment) | 0.267 | **0.397** | 0.384 | 0.315 |
| `circle` (head-on, ring **is** in training) | 0.065 | 0.076 | **0.152** | 0.149 |
| `swap` (head-on, geometry **not** in training) | 0.005 | 0.038 | **0.089** | 0.045 |

The reversal tracks **task type**, not training-set overlap. `swap` is the clean case: a
conflict task with no contamination, and the small training fleets still lose badly.

**So finding 1 is scoped to tasks where collisions are not the binding constraint** —
random start/goal assignment, the GLAS-style benchmark setting. Where the task is a
symmetric head-on conflict, train on the largest fleet you can afford instead.

This mechanism is an **MLP-head result**. The flow head's N=2 policies stall — burning the
whole step budget without moving — so its step-budget curve is U-shaped rather than rising
(arena 0.93 / 0.79 / 0.74 / 0.84) and the test cannot be run on it. Its reversal is milder
in the same direction (arena peaks at train N=4, ring at N=6).

---

## Finding 2 — the encoder does not matter

### The evidence

`data_mid_flow` is the only budget-controlled comparison in the study (all 12 runs at
DAgger round 4, 1.18–1.31 M optimizer steps, ±5%). On `arena`, pooled over all 24 cells:

| Encoder | per-robot success | cells won (of 24) |
| :-- | --: | --: |
| DeepSet | 0.161 | 8 |
| GNN | 0.156 | 7 |
| Transformer | 0.176 | 9 |

A 2-point spread, and a **9 / 8 / 7 split against a 8 / 8 / 8 null** — indistinguishable
from chance. Compare the two design levers on identical cells (see
[Effect sizes: how they are computed](#effect-sizes-how-they-are-computed)):

| Head | encoder effect (SD) | training-fleet effect (SD) | ratio |
| :-- | --: | --: | --: |
| MLP | 0.128 | 0.142 | 1.1× |
| Flow | 0.057 | 0.083 | **1.5×** |

And the decisive comparison — against the noise floor of re-running the same training,
measured on the in-training eval where both quantities are available:

> **Flow: encoder effect 0.068 ≈ round-to-round noise 0.066.**
> Swapping the encoder moves the result about as much as re-rolling which DAgger round
> you keep. For MLP the noise is larger still (0.197 vs 0.157).

### Why the MLP table looks like it says otherwise

On MLP the encoders appear clearly ordered — DeepSet 0.094, GNN 0.197, Transformer 0.182,
with GNN winning 15 of 24 cells. That ordering is an artifact of checkpoint curation:

- The 12 MLP checkpoints come from **different DAgger rounds (0–4), spanning 72 120 to
  1 267 597 optimizer steps — an 18× budget spread.** Flow has no such spread.
- Only **5 of 12** are their run's argmax round (verified by hashing each against
  `outputs/data_mid/models/*/mlp_dagger_iter_*.pt`), and the selection ran on a 20-episode
  eval whose standard error is ≈ ±10 pp.
- DeepSet's deficit correlates with its own step count at r = 0.92 *within* the encoder —
  but that does **not** explain the gap *between* encoders, which runs the other way:
  DeepSet has more training than GNN at every fleet size and still loses (train N=6:
  DeepSet 0.040 at 443 k steps against GNN 0.207 at 72 k; train N=8: 0.013 at 698 k against
  0.407 at 204 k). See [Why DeepSet loses under the MLP head](#why-deepset-loses-under-the-mlp-head).
- The GNN dip at train N=6 is the least-trained checkpoint in the whole study — round 0,
  72 120 steps, pure behaviour cloning, and not even that run's best round. It is a curation
  error, not an encoder property, and it is **absent from the flow table** exactly as that
  explanation predicts.

### What can and cannot be claimed

Honest phrasing matters here. At *episode* level the flow differences are statistically
significant (DeepSet 0.133 vs GNN 0.159, p = 3.8 × 10⁻⁴). But that test answers *"did these
twelve particular networks differ?"* — not *"do these architectures differ?"* For an
architecture claim the unit of replication is the **training run**, of which there is **one
per cell**, and a 2.6 pp gap is far below the 16.6 pp round-to-round noise.

> **Claim: within this study's budget, the neighbour encoder is not a meaningful design
> lever — the choice is dominated by training-run variance. The training fleet size is.**

This is a bounded negative result, not a proof of equivalence: it rules out large effects,
not small ones. An encoder difference of order 0.02 could well be real and would need three
seeds to detect.

### Why DeepSet loses under the MLP head

The one encoder effect that is *not* small, and not a budget artifact. Under the MLP head
on `arena`, DeepSet falls from 0.885 to 0.005 across the training grid while GNN and
Transformer hold up. Three facts pin it down:

**It is not a collision failure.** Pooled over every cell, per-robot collision rate is the
same for all three encoders — 0.539 / 0.541 / 0.525 — and flat across training fleet size
(0.52–0.55 everywhere). The whole deficit is timeouts: 0.270 against 0.079 and 0.106.
Robots reach the goal region and fail to settle inside the 0.1 m tolerance, stopping 0.24 m
(train N=6) to 0.29 m (train N=8) short while burning 99% of the step budget.

**It is not the training budget** — that runs the other way. DeepSet has *more* training
than GNN at every fleet size and still loses:

| train N | DeepSet | GNN | Transformer |
| --: | :-- | :-- | :-- |
| 2 | 0.455 @ 1268 k | 0.455 @ **82 k** | 0.426 @ **82 k** |
| 4 | 0.257 @ 816 k | 0.450 @ 808 k | 0.433 @ 1212 k |
| 6 | 0.040 @ 443 k | 0.207 @ **72 k** | 0.393 @ 1088 k |
| 8 | 0.013 @ 698 k | 0.407 @ 204 k | 0.226 @ 1028 k |

**The three encoders are mathematically equivalent at one neighbour — and measure equal
there.** DeepSet pools with `sum(φ) / count.clamp(min=1)`
(`learning/models/deepset_encoder.py`), so at k = 1 mean ≡ sum ≡ identity, and attention
over a single token is that token's value. All three collapse to "one MLP over the single
neighbour, concatenated with ego". The N=2 runs see 0.99 neighbours, and they score
0.885 / 0.890 / 0.840 on arena and 0.743 / 0.718 / 0.670 on the 1× density cell — a dead
heat, *despite* DeepSet carrying 15× the optimizer steps.

The encoders separate only once k > 1, which is exactly where mean pooling starts doing
something the other two do not: **dividing by the count.** The context becomes a centroid,
invariant to how many robots are nearby and diluting each one by 1/k, so the single robot
that determines the manoeuvre is blurred into an average. Sum pooling keeps both the
magnitude and the individual contribution; attention can place its weight on the nearest.
DeepSet's training exposure rises 0.99 → 2.27 → 2.93 → 3.37 neighbours across the grid, and
its arena score falls monotonically with it.

That also fits the failure *mode*: a blurred context yields an indecisive action rather
than a wrong one, which costs convergence (timeouts, steady-state offset) and not safety
(collisions unchanged).

**Why the flow head hides it.** It does not hide it everywhere — on the head-on ring, flow
DeepSet is still clearly worst (0.057 against GNN 0.128 and Transformer 0.146). On arena
the difference is that an MLP head regresses the *conditional mean* action, so an ambiguous
context produces an averaged, hesitant command; the flow head samples a mode and still
commits. Blurring the conditioning signal hurts a mean-regressor more than a sampler.

**Caveat.** One seed per cell, so this is the best-supported reading rather than a
controlled result. The clean test is a mean-vs-sum ablation on DeepSet alone at a fixed
fleet size — the prediction is that `pool_type: sum` closes most of the gap at N ≥ 4 and
changes nothing at N = 2.

---

## How much of this is signal?

DAgger changes its own dataset every round, so a "training condition" is not fixed even
within one run. The five rounds of one run — same encoder, same fleet, same seed, same
schedule — are the closest thing to replicates available, and they move more than the
encoders do:

| | **within** one run across its 5 rounds | **between** encoders at fixed N + round | ratio |
| :-- | --: | --: | --: |
| MLP — SD | **0.197** | 0.157 | 1.25× |
| Flow — SD | **0.066** | 0.068 | 0.97× |
| MLP — range | 0.471 | 0.298 | 1.58× |
| Flow — range | 0.166 | 0.131 | 1.27× |

**Use the SD rows.** A max−min range grows with the number of groups compared, and here
that is 5 rounds against 3 encoders, which inflates the noise column. Correcting for it
does not change the conclusion but does change its wording: for flow the encoder effect and
the training noise are **the same size** (0.068 vs 0.066), not "noise is 1.27× larger".

DeepSet at N=2 goes 0.85 → 0.35 → 0.70 → 0.30 → 0.45 across its own rounds. For flow, the
20-episode in-training eval would produce 0.196 of *range* from sampling alone — larger than
the measured encoder gap, so that instrument cannot resolve encoders at all.

### Effect sizes: how they are computed

All three quantities are the same statistic — **the spread of `robot_success_rate` across
one factor, holding every other factor fixed, averaged over the held-fixed cells** — so
they are directly comparable, provided they come from the same instrument.

| Quantity | Held fixed (one cell each) | Varied within a cell | Cells |
| :-- | :-- | :-- | --: |
| encoder effect | training fleet size × eval fleet size | the 3 encoders | 24 |
| training-fleet effect | encoder × eval fleet size | the 4 training fleet sizes | 18 |
| round-to-round noise | encoder × training fleet size (= one run) | the 5 DAgger rounds | 12 |

```python
# effect of `over`, holding `group` fixed; SD per cell, then averaged over cells
piv = df.pivot_table(index=group, columns=over, values="robot_success_rate")
effect = piv.std(axis=1).mean()          # report this
spread = (piv.max(axis=1) - piv.min(axis=1)).mean()   # range: inflated by group count
```

**Two instruments, and they must not be mixed:**

- **Final eval** — `robot_success_rate` from `outputs/data_mid*/eval/arena.csv`, 200
  episodes per cell. Gives the encoder effect (0.057 flow) and the training-fleet effect
  (0.083 flow). *No round axis exists here*, because only one checkpoint per run was
  evaluated — so the noise floor cannot be measured on this instrument.
- **In-training eval** — `eval_success_rate` per DAgger round from
  `outputs/data_mid*/models/*/training_summary.yaml`, 20 episodes per round. This is the
  only instrument with a round axis, so the noise floor (0.066 flow) comes from here, and
  the encoder effect must be recomputed on it (0.068 flow) for the comparison to be valid.

The valid comparisons are therefore **0.083 vs 0.057** (final eval: fleet size is the
bigger lever, 1.5×) and **0.068 vs 0.066** (in-training: the encoder effect is the size of
the training noise). Comparing a final-eval effect against an in-training noise figure
mixes two different metrics, scenarios and sample sizes, and is not a valid comparison.
A [round sweep](#1-the-round-sweep--no-retraining-needed) would put all three on the final
eval and remove the caveat entirely.

**The dividing line: a result that pools many training runs survives; a result that reads a
single cell does not.** Finding 1 survives because it is consistent in *direction* across 36
cells with zero counterexamples — consistency, not magnitude, is what carries it. Finding 2
survives because it is a statement about an effect being *smaller* than the noise.

| Claim | Verdict |
| :-- | :-- |
| train 4 ≥ train 8 everywhere, at 62% of the cost | **holds** — 31/5/0, p = 4.7 × 10⁻¹⁰, both heads |
| Encoder is not a meaningful lever | **holds as a bounded negative** — effect ≈ noise floor (0.068 vs 0.066) |
| Caution explains the arena/ring reversal | **holds** — step budget rises with train N in both; collisions move 0.01 in arena vs −0.38 on the ring; and `swap`, a conflict task absent from training, reverses the same way |
| The ring reversal is training-set contamination | **refuted as the cause** — but the *diagonal* is memorised (+0.33 MLP, +0.23 flow over adjacent fleet sizes); off-diagonal the trend survives at r = +0.36 |
| Neighbour count is what a policy must be *matched* to | **refuted** — density sweep: train N=2 wins at every density, including 1.25× |
| Neighbour cap `min(N−1, density·πR²)` caps training exposure at 1 neighbour for N=2 | **holds as arithmetic**; that it *causes* flow's N=2 failure is a hypothesis, untested |
| Upward extrapolation dies at ≈ +3 neighbours | **holds as description** — all 24 runs, both heads |
| DeepSet degrades with training fleet size (MLP) | **holds** — and is *not* a budget artifact: DeepSet out-trains GNN at every fleet size and still loses |
| Mean pooling is the reason | **best available explanation** — the three encoders are mathematically equivalent at 1 neighbour and measure equal there; DeepSet separates only once k > 1 |
| GNN anomaly at train N=6 | **artifact** — round-0 checkpoint, 72 k steps |
| Any fine-grained encoder ranking from `data_mid_best` | **not usable** — 18× budget spread |

---

## Metric caveat: clearance rewards standing still

`mean_min_pair_distance` is the episode minimum of the closest pair, averaged over episodes.
With `d_collision = 1.0 m` it is a continuous relaxation of the collision flag — on
`circle_det` the two agree in all 192 rows.

**It is only meaningful when the fleet attempts the task.** A fleet that never leaves its
start ring keeps its starting separation: 1.91 m at N=16, 1.36 m at N=32, both above
`d_collision`. Standing still scores as maximally safe.

This contaminates the flow ring data: **34 of 192 cells never did the task**, 29 of them in
the train-N=2 row, and they top the clearance table — DeepSet train N=2 at eval 11–16 scores
**1.75–1.95 m**, the highest clearance anywhere in the flow data, on 100% timeouts having
finished *further* from its goals than it started. Progress
(`1 − mean_goal_position_error / 2R`) separates these cleanly; all 192 MLP cells completed
the swap (progress ≥ 0.86), so only the flow table is affected.

Report completion or `robot_success_rate` as primary, and gate clearance on progress.

---

## Scope limits

- **One seed per cell.** Nothing here separates an architecture effect from training noise,
  which is why finding 2 is phrased as a bound rather than an equivalence.
- **No usable policy above ~6 robots.** Episode success at eval N=8 is 0.015–0.018 and
  per-robot success 0.19–0.37; at N=16 and N=32 the ring and swap scenarios are at or near
  zero for everything. Finding 1 is a relative-cost claim inside a regime where nothing
  works well in absolute terms.
- **Finding 1 is scoped to deployment at or below the training density.** `circle` reverses
  it above.
- **Flow's head is stochastic**, so its 10-episode ring cells average over sampling while
  the deterministic MLP's single episode does not. Cross-head comparisons of the ring are
  not like-for-like.

## What would settle it

### 1. The round sweep — no retraining needed

All **120 per-round checkpoints still exist**
(`outputs/data_mid{,_flow}/models/*/[mf]*_dagger_iter_*.pt`: 5 rounds × 12 runs × 2 heads).
Evaluating every round instead of one curated checkpoint turns the round axis into **5
measurements per cell**, and fixes the budget confound for free since round 4 is round 4 for
everyone.

| Sweep | Cost | Buys |
| :-- | --: | :-- |
| MLP, arena N=2..8, 200 ep | ≈ 13 core-h (~2 h at 8-way) | whether the MLP encoder ordering is real |
| Both heads, arena N=2..8 | ≈ 30 core-h (~4 h at 8-way) | error bars on both findings |
| Both heads, arena + circle, all sizes | ≈ 150 core-h | adds the ring and the extrapolation cliff |

`arena` N=16/32 can be dropped: both are saturated near zero and cost 55% of the runtime.

The evidence is asymmetric, which is fine — if an ordering **flips** round to round the
non-robustness is settled; if it **holds** it is suggestive but not proof, because round 4
contains rounds 0–3's data.

### 2. Everything else

- **Three seeds** — the only thing that turns finding 2 from a bound into a measurement.
- **Extend the density axis above 1×** to test whether finding 1's scope limit is really
  about density rather than about the ring geometry.
- **Gate clearance on progress** wherever a stalling policy can appear.
