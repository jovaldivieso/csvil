#!/usr/bin/env bash
# Encoder-scaling study driver: 3 encoders x 4 training fleet sizes, then
# every resulting policy evaluated on 6 fleet sizes.
#
#   ./run_study.sh train [fleet ...]  # DAgger runs -> outputs/train_dagger_multi_robot/<name>/
#   ./run_study.sh eval [scenario]    # 12 x 6 cells -> the scenario's merged CSV
#
# `train` with no argument trains every encoder on every fleet size (12 runs).
# Pass fleet sizes to restrict it -- `./run_study.sh train 8` trains the three
# encoders on 8 robots only. Restrict the encoders too with ENCODERS="deepset gnn".
#
# There is one policy config per (encoder, fleet size) rather than per encoder,
# because each one pins part of every DAgger round to hand-written antipodal-ring
# initial states -- the training-time counterpart of the `circle` scenario below --
# and those coordinates are per-robot. The 12 configs are generated from the three
# encoder templates by
#
#   python learning/config/study/generate_study_policy_configs.py
#
# so encoder changes go in the templates and layout changes in that script.
#
# Evaluation is one code path over a table of scenarios; a scenario is just a set
# of configs plus its rollout defaults, so adding one is a row in the SCENARIO_*
# tables below.
#
#   random (default) - the randomized-goal configs the policies trained on.
#   circle           - the deterministic antipodal swap, where every robot crosses
#                      the centre toward a goal that is another robot's start.
#
# Training and evaluation are independent: eval needs only the checkpoints and the
# configs, so it can run later, on a different machine, or be repeated with
# different settings without retraining anything.
#
# Override a scenario's rollout defaults with EVAL_EPISODES / EVAL_STEPS / EVAL_NOISE.

set -uo pipefail

MODE="${1:-}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
read -r -a ENCODERS <<< "${ENCODERS:-deepset transformer gnn}"
TRAIN_FLEET_SIZES=(8 6 4 2)

# Trajectories per DAgger round, inversely proportional to the fleet size: each
# episode emits one LeRobot episode *per robot*, so this holds frames-per-round
# (and therefore optimizer steps per round) constant at ~20k across fleet sizes.
#
# Mirrored by TRAJECTORIES_PER_ROUND in learning/config/study/generate_study_policy_configs.py,
# which sizes each policy config's ring-layout list as a fixed share of the round.
# Change one and re-run that script.
declare -A TRAJECTORIES=([2]=150 [4]=100 [6]=75 [8]=50)

EVAL_FLEET_SIZES=(02 04 06 08 16 32)

# Scenario table: config path template (%s is the zero-padded fleet size), extra
# flags, episodes, steps, output directory, merged CSV.
#
# circle needs 400 steps because its 32-robot ring is >=184 steps just to cross in
# a straight line -- the 200 that suits the random configs would score it a timeout
# before it could finish. It is also deterministic, so one episode is the whole
# result; raise EVAL_NOISE and EVAL_EPISODES together to sample robustness around
# the nominal swap instead.
declare -A SCENARIO_TEMPLATE=(
  [random]="test/config/study/unicycle2_fleet_%s.yaml"
  [circle]="test/config/study/circle/unicycle2_circle_%s.yaml"
)
declare -A SCENARIO_FLAGS=(    [random]=""    [circle]="--use-config-start" )
declare -A SCENARIO_EPISODES=( [random]=50    [circle]=1 )
declare -A SCENARIO_STEPS=(    [random]=200   [circle]=400 )
declare -A SCENARIO_OUTDIR=(   [random]="outputs/study" [circle]="outputs/study/circle" )
declare -A SCENARIO_MERGED=(
  [random]="outputs/study/encoder_scaling.csv"
  [circle]="outputs/study/circle/circle_scaling.csv"
)

# -u/HOME: without these the container writes root-owned files into the bind mount.
# *_NUM_THREADS=1: IPOPT/BLAS and torch each grab every core otherwise, so parallel
# runs would oversubscribe the machine and finish slower than running them serially.
docker_run() {
  docker compose run --rm -T \
    -u "$(id -u):$(id -g)" -e HOME=/tmp \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    csvil "$@"
}

await_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n; done
}

# $1 exit status, $2 label, $3 log file. Takes the status as an argument because
# a $(...) anywhere ahead of $? would run first and reset it.
report_status() {
  local status="$1" label="$2" log="$3"
  local stamp; stamp="$(date +%H:%M:%S)"
  if (( status == 0 )); then
    echo "[${stamp}] ${label} OK"
  else
    echo "[${stamp}] ${label} FAILED (exit ${status}) -- last log lines:"
    tail -n 5 "$log" | sed "s|^|    ${label}\| |"
  fi
}

# Concatenate per-policy CSVs, keeping one header. $1 output file, rest inputs.
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
  echo "Merged ${rows} result rows into ${merged}"
}

train_one() {
  local encoder="$1" fleet_size="$2"
  local padded; padded="$(printf %02d "$fleet_size")"
  local name="${encoder}_n${padded}"
  local policy_config="learning/config/study/${encoder}_mlp_n${padded}_config.yaml"

  # The per-fleet configs are generated, so a missing one means the generator has
  # not been run (or not for this fleet size) -- say so here rather than letting
  # train_dagger.py fail on the path a few seconds into the container.
  if [[ ! -f "$policy_config" ]]; then
    echo "[skip] train ${name}: no policy config at ${policy_config};" \
         "run python learning/config/study/generate_study_policy_configs.py"
    return
  fi

  docker_run python learning/train_dagger.py \
    --experiment-name "$name" \
    --system multi_robot \
    --expert-config "test/config/study/unicycle2_fleet_${padded}.yaml" \
    --policy-config "$policy_config" \
    --dagger-iterations 3 \
    --trajectories-per-iteration "${TRAJECTORIES[$fleet_size]}" \
    --steps-per-trajectory 200 \
    --target-epochs-per-round 10 \
    --action-noise-std 0.03 \
    --expert-mix-beta-start 0.5 \
    --expert-mix-beta-decay-rate 0.25 \
    --expert-mix-decay-after-success-rate 0.0 \
    --eval-episodes 20 \
    > "logs/${name}.log" 2>&1
  # Capture before anything else runs: a $(...) ahead of $? would reset it.
  local status=$?
  report_status "$status" "train ${name}" "logs/${name}.log"
}

# Build the config list for a scenario. The paths cannot be a single brace
# expansion held in a variable -- brace expansion does not happen on expansion.
scenario_configs() {
  local template="${SCENARIO_TEMPLATE[$1]}" size
  for size in "${EVAL_FLEET_SIZES[@]}"; do
    printf "${template}\n" "$size"
  done
}

eval_one() {
  local scenario="$1" encoder="$2" fleet_size="$3"
  local padded; padded="$(printf %02d "$fleet_size")"
  local name="${encoder}_n${padded}"
  local checkpoint="outputs/train_dagger_multi_robot/${name}/mlp_dagger_checkpoint.pt"
  local log="logs/${scenario}_${name}.log"

  if [[ ! -f "$checkpoint" ]]; then
    echo "[skip] ${scenario} ${name}: no checkpoint at ${checkpoint}"
    return
  fi

  local configs; mapfile -t configs < <(scenario_configs "$scenario")
  local out_csv="${SCENARIO_OUTDIR[$scenario]}/${name}.csv"

  # evaluate_scaling.py appends, so a re-run would stack a second set of rows onto
  # the first. Drop the previous result to keep this mode idempotent.
  rm -f "$out_csv"

  # One CSV per policy: concurrent appends to a shared file would interleave rows.
  docker_run python test/evaluate_scaling.py \
    --checkpoint "$checkpoint" \
    --configs "${configs[@]}" \
    --episodes "$EPISODES" \
    --steps "$STEPS" \
    --action-noise-std "$NOISE" \
    --seed-start 50000 \
    ${SCENARIO_FLAGS[$scenario]} \
    --output-csv "$out_csv" \
    > "$log" 2>&1
  local status=$?
  report_status "$status" "${scenario} ${name}" "$log"
}

mkdir -p logs

case "$MODE" in
  train)
    # Any extra arguments restrict which fleet sizes to train.
    if (( $# > 1 )); then
      shift
      requested=("$@")
      for fleet_size in "${requested[@]}"; do
        if [[ -z "${TRAJECTORIES[$fleet_size]+x}" ]]; then
          echo "unknown fleet size '${fleet_size}'; known: ${!TRAJECTORIES[*]}" >&2
          exit 1
        fi
      done
      # Descending: the largest fleet is by far the longest run, so it must start
      # immediately rather than being queued behind the cheap ones.
      mapfile -t TRAIN_FLEET_SIZES < <(printf '%s\n' "${requested[@]}" | sort -rn)
    fi

    echo "training fleets: ${TRAIN_FLEET_SIZES[*]} | encoders: ${ENCODERS[*]}"
    for fleet_size in "${TRAIN_FLEET_SIZES[@]}"; do
      for encoder in "${ENCODERS[@]}"; do
        await_slot
        train_one "$encoder" "$fleet_size" &
      done
    done
    wait
    echo "All training runs finished. Checkpoints under outputs/train_dagger_multi_robot/"
    ;;

  eval)
    scenario="${2:-random}"
    if [[ -z "${SCENARIO_TEMPLATE[$scenario]+x}" ]]; then
      echo "unknown scenario '${scenario}'; known: ${!SCENARIO_TEMPLATE[*]}" >&2
      exit 1
    fi
    EPISODES="${EVAL_EPISODES:-${SCENARIO_EPISODES[$scenario]}}"
    STEPS="${EVAL_STEPS:-${SCENARIO_STEPS[$scenario]}}"
    NOISE="${EVAL_NOISE:-0.0}"
    mkdir -p "${SCENARIO_OUTDIR[$scenario]}"

    echo "scenario=${scenario} episodes=${EPISODES} steps=${STEPS} action_noise=${NOISE}"
    for encoder in "${ENCODERS[@]}"; do
      for fleet_size in 2 4 6 8; do
        await_slot
        eval_one "$scenario" "$encoder" "$fleet_size" &
      done
    done
    wait
    merge_csvs "${SCENARIO_MERGED[$scenario]}" "${SCENARIO_OUTDIR[$scenario]}"/*_n??.csv
    ;;

  *)
    echo "usage: $0 train [fleet ...] | eval [${!SCENARIO_TEMPLATE[*]}]" >&2
    echo "       fleet sizes: ${!TRAJECTORIES[*]}   (MAX_PARALLEL=${MAX_PARALLEL}, ENCODERS=\"${ENCODERS[*]}\")" >&2
    exit 1
    ;;
esac
