#!/usr/bin/env bash
# Study evaluation driver: evaluates every checkpoint on 6 fleet sizes.
#
#   ./run_study.sh eval [scenario]    # every checkpoint x 6 fleet sizes -> merged CSV
#
# Training lives in train.sh. NOTE: this eval still expects the old study-2 layout --
# runs named <encoder>_<variant>_n<NN>_s<seed> under MODEL_ROOT -- so it does not find
# runs trained with train.sh yet.
#
# Checkpoints from before the policy axis existed are named <encoder>_n<NN>; eval
# falls back to that older directory for mlp runs, so the existing study results can
# be reproduced without renaming anything.
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
# Policy variant labels (the <variant> in run names), crossed with ENCODERS.
read -r -a POLICIES <<< "${POLICIES:-mlp}"
# Training seeds of the runs to evaluate.
read -r -a SEEDS <<< "${SEEDS:-0}"

EVAL_FLEET_SIZES=(02 04 06 08 16 32)

# Scenario table: config path template (%s is the zero-padded fleet size), extra
# flags, episodes, steps, output directory, merged CSV.
#
# circle needs 400 steps because its 32-robot ring is >=184 steps just to cross in
# a straight line -- the 200 that suits the random configs would score it a timeout
# before it could finish. The layout is fixed, so evaluate_scaling.py collapses a
# deterministic policy (the MLP) to one episode; flow samples its actions and keeps
# all 50.
declare -A SCENARIO_TEMPLATE=(
  [random]="test/config/study/unicycle2_fleet_%s.yaml"
  [circle]="test/config/study/circle/unicycle2_circle_%s.yaml"
)
declare -A SCENARIO_FLAGS=(    [random]=""    [circle]="--use-config-start" )
declare -A SCENARIO_EPISODES=( [random]=50    [circle]=50 )
declare -A SCENARIO_STEPS=(    [random]=200   [circle]=400 )
# Where eval reads checkpoints and writes results (study 2 layout by default).
MODEL_ROOT="${MODEL_ROOT:-outputs/study2/models}"
EVAL_ROOT="${EVAL_ROOT:-outputs/study2/eval}"

# Success criterion for eval: vel_tol = omega_tol = EVAL_VEL_TOL, pos_tol = EVAL_POS_TOL.
# Set both to empty to score with the tolerances in the scenario configs instead.
EVAL_VEL_TOL="${EVAL_VEL_TOL-0.1}"
EVAL_POS_TOL="${EVAL_POS_TOL-0.2}"
EVAL_TOL_DIR=""

declare -A SCENARIO_OUTDIR=(   [random]="${EVAL_ROOT}/random" [circle]="${EVAL_ROOT}/circle" )
declare -A SCENARIO_MERGED=(
  [random]="${EVAL_ROOT}/random/encoder_scaling.csv"
  [circle]="${EVAL_ROOT}/circle/encoder_scaling.csv"
)

# -u/HOME/USER: without -u the container writes root-owned files into the bind mount,
# but the host uid then has no entry in the image's passwd database, so anything
# calling getpass.getuser() (LeRobot's dataset stack does) dies with
# "KeyError: getpwuid(): uid not found". getuser() reads LOGNAME/USER/LNAME/USERNAME
# before it falls back to the passwd database, so setting USER is enough. A fixed
# literal rather than $(id -un) keeps runs identical across machines.
# *_NUM_THREADS=1: IPOPT/BLAS and torch each grab every core otherwise, so parallel
# runs would oversubscribe the machine and finish slower than running them serially.
docker_run() {
  docker compose run --rm -T \
    -u "$(id -u):$(id -g)" -e HOME=/tmp -e USER=csvil \
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

# Build the config list for a scenario. The paths cannot be a single brace
# expansion held in a variable -- brace expansion does not happen on expansion.
scenario_configs() {
  local template="${SCENARIO_TEMPLATE[$1]}" size path
  for size in "${EVAL_FLEET_SIZES[@]}"; do
    path="$(printf "${template}" "$size")"
    # Under a criterion override, use the derived copy written before the job loop.
    [[ -n "$EVAL_TOL_DIR" ]] && path="${EVAL_TOL_DIR}/$(basename "$path")"
    printf "%s\n" "$path"
  done
}

eval_one() {
  local scenario="$1" encoder="$2" policy="$3" fleet_size="$4" seed="$5"
  local padded; padded="$(printf %02d "$fleet_size")"
  local name="${encoder}_${policy}_n${padded}_s${seed}"
  # train_dagger.py names the checkpoint <policy_type>_dagger_checkpoint.pt, and a
  # variant label is not a policy type (mlp_h10 is an mlp), so take whichever exists.
  local checkpoint
  checkpoint="$(compgen -G "${MODEL_ROOT}/${name}/*_dagger_checkpoint.pt" | head -n1)"
  [[ -n "$checkpoint" ]] || checkpoint="${MODEL_ROOT}/${name}/<policy_type>_dagger_checkpoint.pt"

  # Fall back to the pre-policy-axis layout so checkpoints trained before this
  # script grew a POLICIES axis stay evaluable under their original names.
  if [[ ! -f "$checkpoint" && "$policy" == "mlp" && "$seed" == "0" ]]; then
    local legacy="${MODEL_ROOT}/${encoder}_n${padded}/mlp_dagger_checkpoint.pt"
    if [[ -f "$legacy" ]]; then
      checkpoint="$legacy"
      name="${encoder}_n${padded}"
    fi
  fi
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
    --train-seed "$seed" \
    ${SCENARIO_FLAGS[$scenario]} \
    --output-csv "$out_csv" \
    > "$log" 2>&1
  local status=$?
  report_status "$status" "${scenario} ${name}" "$log"
}

mkdir -p logs

case "$MODE" in
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

    # Derived configs for a criterion override, written once here rather than inside
    # each job so parallel jobs cannot race on the same file.
    if [[ -n "${EVAL_VEL_TOL}${EVAL_POS_TOL}" ]]; then
      EVAL_TOL_DIR="${SCENARIO_OUTDIR[$scenario]}/configs"
      mkdir -p "$EVAL_TOL_DIR"
      sed_args=()
      [[ -n "$EVAL_VEL_TOL" ]] && sed_args+=(
        -e "s/^\( *\)vel_tol: .*/\1vel_tol: ${EVAL_VEL_TOL}/"
        -e "s/^\( *\)omega_tol: .*/\1omega_tol: ${EVAL_VEL_TOL}/")
      [[ -n "$EVAL_POS_TOL" ]] && sed_args+=(-e "s/^\( *\)pos_tol: .*/\1pos_tol: ${EVAL_POS_TOL}/")
      for size in "${EVAL_FLEET_SIZES[@]}"; do
        src="$(printf "${SCENARIO_TEMPLATE[$scenario]}" "$size")"
        sed "${sed_args[@]}" "$src" > "${EVAL_TOL_DIR}/$(basename "$src")"
      done
      echo "criterion override: vel/omega=${EVAL_VEL_TOL:-unchanged} pos=${EVAL_POS_TOL:-unchanged}"
    fi

    echo "scenario=${scenario} episodes=${EPISODES} steps=${STEPS} action_noise=${NOISE}" \
         "| encoders: ${ENCODERS[*]} | policies: ${POLICIES[*]}"
    for encoder in "${ENCODERS[@]}"; do
      for policy in "${POLICIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
          for fleet_size in 2 4 6 8; do
            await_slot
            eval_one "$scenario" "$encoder" "$policy" "$fleet_size" "$seed" &
          done
        done
      done
    done
    wait
    # Both patterns: seeded runs and the pre-seed-axis checkpoints. Anything that
    # matches neither is not a per-policy result file and must not be merged in.
    merge_csvs "${SCENARIO_MERGED[$scenario]}" \
      "${SCENARIO_OUTDIR[$scenario]}"/*_n??.csv "${SCENARIO_OUTDIR[$scenario]}"/*_n??_s*.csv
    ;;

  *)
    echo "usage: $0 eval [${!SCENARIO_TEMPLATE[*]}]" >&2
    echo "       (MAX_PARALLEL=${MAX_PARALLEL}," \
         "ENCODERS=\"${ENCODERS[*]}\", POLICIES=\"${POLICIES[*]}\", SEEDS=\"${SEEDS[*]}\")" >&2
    exit 1
    ;;
esac
