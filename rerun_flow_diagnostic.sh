#!/usr/bin/env bash
# One-cell diagnostic: is study 1's flow collapse caused by the training budget and
# the beta schedule, rather than by the architecture?
#
# Reruns deepset_flow_n04 seed 0 with the two settings that differ most from the
# known-good flow run in learning/config/multi_unicycle2_casadi_flow_config.yaml:
#
#   MAX_TRAIN_STEPS  4000 -> 40000   (study 1 trained ~12k steps total; the working
#                                     flow config works out to >1e5)
#   beta decay gate  0.0  -> 0.5     (hold the expert in control until the policy
#                                     can actually reach goals)
#
# Everything else -- encoder, hidden dims, horizons, scenario, tolerances, seed,
# DAgger schedule -- is left exactly as the study 1 run had it.
#
# Both knobs move together on purpose: this answers "can flow work here at all",
# not "which of the two was responsible". If it recovers, a second run with only
# MAX_TRAIN_STEPS raised separates them.
#
# Writes to a distinct experiment name so the original checkpoints are untouched.
# Runtime: ~3.5 h on one core; give it a tmux session.

set -uo pipefail

NAME="${NAME:-deepset_flow_n04_s0_rerun}"
STEPS="${STEPS:-40000}"
GATE="${GATE:-0.5}"

mkdir -p logs

echo "rerun: ${NAME} | max_train_steps=${STEPS} | beta_decay_after_eval_success=${GATE}"

docker compose run --rm -T \
  -u "$(id -u):$(id -g)" -e HOME=/tmp -e USER=csvil \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  csvil python learning/train_dagger.py \
    --experiment-name "$NAME" \
    --system multi_robot \
    --expert-config test/config/study/unicycle2_fleet_04.yaml \
    --policy-config learning/config/study/deepset_flow_n04_config.yaml \
    --dagger-iterations 3 \
    --trajectories-per-iteration 100 \
    --steps-per-trajectory 200 \
    --target-epochs-per-round 200 \
    --max-train-steps "$STEPS" \
    --seed 0 \
    --action-noise-std 0.03 \
    --expert-mix-beta-start 0.5 \
    --expert-mix-beta-decay-rate 0.25 \
    --expert-mix-decay-after-eval-success "$GATE" \
    --eval-episodes 20 \
  2>&1 | tee "logs/${NAME}.log"

status=${PIPESTATUS[0]}
echo
if (( status == 0 )); then
  echo "done. compare against the original run:"
  echo "  original: $(tail -1 outputs/study1/models/deepset_flow_n04_s0/results.csv)"
  echo "  rerun:    $(tail -1 "outputs/train_dagger_multi_robot/${NAME}/results.csv" 2>/dev/null)"
  echo "columns: train_loss, aggregation_success_rate, eval_success_rate, eval_mean_steps"
  echo
  echo "eval_success_rate was 0.10 originally. Anything near the MLP's 0.60-0.75"
  echo "means the collapse was budget/schedule, not the flow architecture."
else
  echo "FAILED (exit ${status}); see logs/${NAME}.log"
fi
