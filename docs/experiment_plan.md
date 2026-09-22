# Experiment plan

Two studies of decentralized DAgger policies for a `unicycle2` fleet imitating a
CasADi MPC expert. How to run them: [evaluation.md](evaluation.md). Study 2 in full:
[study2_encoders.md](study2_encoders.md).

## Study 1 — policy head and action horizon

**Question.** Does a flow-matching policy imitate the expert better than a deterministic
MLP? The hypothesis: where robots interact, the expert's actions are multimodal (pass
left or right), and an MLP trained with MSE to predict an 8-step action chunk averages
the two into driving straight through.

**Design.** Trained at fleet size N=4 with the DeepSet encoder, 3 seeds per policy.

| | predicts 1 action | predicts 8 actions |
|---|---|---|
| **MLP** | `mlp` | `mlp_h8` |
| **Flow** | `flow_h1` | `flow` |

All cells share the encoder, network size, scenario configs, DAgger schedule and a fixed
40k gradient steps per round.

**Status.** Done. Models, results and plot are in `outputs/study1/{models,eval,plots}`.

**Results**
- *Random goals:* the four cells are indistinguishable at every fleet size. Success is
  0.93–0.95 at N=2, 0.73–0.78 at N=4, 0.56–0.62 at N=8, 0.14–0.19 at N=16 and 0.04–0.05
  at N=32, and nearly every failure is a collision.
- *Ring:* every cell fails from N=8 on. `mlp_h8` solves the N=4 ring on all three
  seeds, and flow is weaker there (0.51 for `flow_h1`, 0.33 for `flow`) — but N=4 is the
  training size (see caveats). At N=2, `mlp_h8` scores 0.67 against `mlp`'s 1.00, the
  direction the hypothesis predicts, on three pass/fail seeds.
- *Speed:* flow takes 9–14 ms per control step against the MLP's 2–3 ms (N=2–8).

## Study 2 — neighbour encoders

Design, training setup, evaluation scenarios and caveats of the **current** run:
[study2_encoders.md](study2_encoders.md).

**Status.** Being retrained. The configs, the two evaluation axes (fleet size and
density) and the scenarios are new; the results below come from the earlier run, whose
checkpoints can no longer be loaded (their observation layout predates the `state_mask`
and neighbour-mask features).

**Earlier results** (N=2-8 trained, evaluated at N=2-32, one seed, older settings)
- *Random goals:* the encoders are indistinguishable under both policies. Flow with the
  GNN is lowest from N=8 on (0.53, 0.12, 0.01 at N=8, 16, 32).
- *Ring, flow:* the encoders separate. At N=2, DeepSet reaches 0.76, the Transformer 0.65
  and the GNN 0.42, and the GNN is lowest throughout. One possible cause is that the GNN
  sums over neighbours, so its input grows with the number it sees.

## Decisions

| Decision | Reason |
|---|---|
| Success needs position within 0.2 m, heading within 1.1 rad, speed and yaw rate below 0.1, for both studies | With the configs' 0.05 speed limit, flow fails by not stopping rather than by colliding. The limit decided the result: at N=4, flow 0.06 vs MLP 0.64 at 0.05, but 0.76 vs 0.82 at 0.5 with a 0.2 m position tolerance. |
| Tolerances belong in the scenario configs, never in policy configs | `evaluate_scaling.py` ignores policy-config overrides, so an override silently trains and scores policies against different criteria. |
| Collisions count below 1.0 m in every scenario | The ring configs used 1.2 m, exactly where the expert's MPC keeps robots apart, so the expert "failed". At 1.0 m the expert solves the ring up to N=8. |
| 40k gradient steps per round, fixed for every run | 4k starved flow (in-training success 0.0–0.2). A fixed number also stops a weaker policy, which collects less data, from being trained less. |
| Study 1 trains at one fleet size only | A decentralized policy runs at any fleet size; training size is study 2's question. |
| One result row per policy and fleet size, not per episode | Per-episode records were too large a change. Without paired tests, resolving a 10-point difference needs ~350 episodes per cell. |
| Flow keeps `torch.compile`, the MLP doesn't | Left as it was. Compiling makes flow 15–19% slower at this model size, so its timing reads high. |

## Caveats

- **Ring results at a policy's training fleet size are partly memorised.** Training
  rounds start some episodes from ring layouts, and the first one is the exact evaluation
  ring. Success there isn't generalisation; study 1's N=4 ring results are in this group.
- **Cells don't share data.** Each policy partly drives its own DAgger rollouts, so a
  difference reflects both the policy and the data it collected.
- **Study 2 has one seed per cell.** Its MLP grid was trained with older settings, so
  compare encoders within a policy, not across policies.
- **The MLP is deterministic**, so each MLP ring result is a single pass or fail.
- Terminal goal error is averaged over all episodes, successes included.

## Open

- Mark or exclude ring results at the training fleet size in the plots.
- Overlay the success rate expected from per-robot reliability alone (`p^N`): success
  needs every robot to succeed, so it falls with fleet size even if each robot is as
  reliable as before.
- More seeds for study 2 before claiming an encoder ranking.
- Retrain study 2's MLP grid with study 1's settings to compare encoders across
  policies.
