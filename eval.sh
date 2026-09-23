#!/usr/bin/env bash
# Evaluate every checkpoint of an experiment on one scenario, in parallel containers.
#
#   ./eval.sh <experiment> [scenario]        # scenario: random (default) | density | circle | all
#
#   env: EPISODES=50  MAX_PARALLEL=8  RUNNER=docker|local  SEED_START=50000
#        STEP_BUDGET_FACTOR=3  ACTION_NOISE=0.0
#
# Reads  outputs/<experiment>/models/<run>/<policy_type>_dagger_checkpoint.pt
# Writes outputs/<experiment>/eval/<scenario>/<run>.csv, merged into
#        outputs/<experiment>/eval/<scenario>.csv (one row per checkpoint x config)
#        outputs/<experiment>/eval/<scenario>/logs/<run>.log
#
# The scenarios are the study's two axes plus the ring; each changes one quantity:
#
#   random   fleet size 2..32 at the training density (workspace grows as sqrt(N))
#   density  0.25x .. 3x the training density at a fixed fleet size (N=4 and N=8)
#   circle   antipodal swap from the configs' fixed starts -- the stress case
#
# Every scenario derives its step budget per config from the distances that config
# produces (see step_budget in test/evaluate_scaling.py), because a fixed budget would
# fail sparse configs by timeout just for having a larger workspace.
#
# Training and evaluation are independent: this needs only the checkpoints and the
# configs, so it can run later, elsewhere, or again with other settings.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

EXPERIMENT="${1:-}"
SCENARIO="${2:-random}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
EPISODES="${EPISODES:-50}"
SEED_START="${SEED_START:-50000}"
STEP_BUDGET_FACTOR="${STEP_BUDGET_FACTOR:-3}"
ACTION_NOISE="${ACTION_NOISE:-0.0}"
RUNNER="${RUNNER:-docker}"

declare -A SCENARIO_GLOB=(
  [random]="test/config/study/unicycle2_fleet_*.yaml"
  [density]="test/config/study/density/*.yaml"
  [circle]="test/config/study/circle/*.yaml"
)
# The ring layouts are the configs' own 'start' entries, not sampled ones.
declare -A SCENARIO_FLAGS=( [random]="" [density]="" [circle]="--use-config-start" )

usage() {
  echo "usage: $0 <experiment> [${!SCENARIO_GLOB[*]}|all]" >&2
  echo "       (EPISODES=${EPISODES}, MAX_PARALLEL=${MAX_PARALLEL}, RUNNER=${RUNNER})" >&2
  exit 1
}

[[ -n "$EXPERIMENT" ]] || usage
[[ "$SCENARIO" == "all" || -n "${SCENARIO_GLOB[$SCENARIO]+x}" ]] || usage

MODEL_DIR="outputs/${EXPERIMENT}/models"
[[ -d "$MODEL_DIR" ]] || { echo "no models at ${MODEL_DIR}" >&2; exit 1; }

# -u/HOME/USER: without -u the container writes root-owned files into the bind mount,
# and the host uid has no passwd entry in the image, so getpass.getuser() fails unless
# USER is set. *_NUM_THREADS=1, or parallel jobs oversubscribe the CPU.
run_eval() {
  if [[ "$RUNNER" == docker ]]; then
    docker compose run --rm -T \
      -u "$(id -u):$(id -g)" -e HOME=/tmp -e USER=csvil \
      -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
      csvil python test/evaluate_scaling.py "$@"
  else
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python3 test/evaluate_scaling.py "$@"
  fi
}

await_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n; done
}

# Concatenate the per-run CSVs, keeping one header. $1 output file, rest inputs.
merge_csvs() {
  local merged="$1"; shift
  local first=1
  : > "$merged"
  for csv in "$@"; do
    [[ -f "$csv" ]] || continue
    if (( first )); then cat "$csv" >> "$merged"; first=0
    else tail -n +2 "$csv" >> "$merged"; fi
  done
  local rows; rows=$(wc -l < "$merged")
  (( rows > 0 )) && rows=$(( rows - 1 ))
  echo "merged ${rows} result rows into ${merged}"
}

eval_one() {
  local scenario="$1" run="$2"; shift 2
  local configs=("$@")
  local checkpoint
  # train_dagger.py names the checkpoint <policy_type>_dagger_checkpoint.pt, and the run
  # name does not always carry the type, so take whichever one is there.
  checkpoint="$(compgen -G "${MODEL_DIR}/${run}/*_dagger_checkpoint.pt" | head -n1)"
  if [[ -z "$checkpoint" ]]; then
    echo "[skip] ${scenario} ${run}: no checkpoint in ${MODEL_DIR}/${run}/"
    return
  fi
  # Training seed from the trailing _s<seed>, so results can be grouped by seed.
  local train_seed=""
  [[ "$run" =~ _s([0-9]+)$ ]] && train_seed="${BASH_REMATCH[1]}"

  local out_dir="outputs/${EXPERIMENT}/eval/${scenario}"
  local out_csv="${out_dir}/${run}.csv"
  local log="${out_dir}/logs/${run}.log"
  # evaluate_scaling.py appends, so a rerun would stack a second set of rows onto the
  # first. One CSV per run: concurrent appends to a shared file would interleave rows.
  rm -f "$out_csv"

  run_eval \
    --checkpoint "$checkpoint" \
    --configs "${configs[@]}" \
    --episodes "$EPISODES" \
    --step-budget-factor "$STEP_BUDGET_FACTOR" \
    --action-noise-std "$ACTION_NOISE" \
    --seed-start "$SEED_START" \
    --train-seed "$train_seed" \
    ${SCENARIO_FLAGS[$scenario]} \
    --output-csv "$out_csv" \
    > "$log" 2>&1
  # Capture before anything else runs: a $(...) ahead of $? would reset it.
  local status=$?
  local stamp; stamp="$(date +%H:%M:%S)"
  if (( status == 0 )); then
    echo "[${stamp}] ${scenario} ${run} OK"
  else
    echo "[${stamp}] ${scenario} ${run} FAILED (exit ${status}) -- last log lines:"
    tail -n 5 "$log" | sed "s|^|    ${run}\| |"
  fi
}

run_scenario() {
  local scenario="$1"
  local configs; mapfile -t configs < <(compgen -G "${SCENARIO_GLOB[$scenario]}" | sort)
  if (( ${#configs[@]} == 0 )); then
    echo "no configs for scenario '${scenario}' (${SCENARIO_GLOB[$scenario]});" \
         "run the generators in test/config/" >&2
    return 1
  fi
  local runs; mapfile -t runs < <(find "$MODEL_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  if (( ${#runs[@]} == 0 )); then
    echo "no run directories under ${MODEL_DIR}" >&2
    return 1
  fi

  local out_dir="outputs/${EXPERIMENT}/eval/${scenario}"
  mkdir -p "${out_dir}/logs"
  echo "scenario=${scenario}: ${#runs[@]} checkpoints x ${#configs[@]} configs," \
       "${EPISODES} episodes, step budget ${STEP_BUDGET_FACTOR}x, ${MAX_PARALLEL} at a time"

  for run in "${runs[@]}"; do
    await_slot
    eval_one "$scenario" "$run" "${configs[@]}" &
  done
  wait
  merge_csvs "outputs/${EXPERIMENT}/eval/${scenario}.csv" "${out_dir}"/*.csv
}

if [[ "$SCENARIO" == "all" ]]; then
  for scenario in random density circle; do run_scenario "$scenario"; done
else
  run_scenario "$SCENARIO"
fi
