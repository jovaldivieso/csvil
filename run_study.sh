#!/usr/bin/env bash
# Encoder-scaling study driver: 3 encoders x 4 training fleet sizes, then
# every resulting policy evaluated on 6 fleet sizes.
#
#   ./run_study.sh train    # 12 DAgger runs -> outputs/train_dagger_multi_robot/<name>/
#   ./run_study.sh eval     # 12 x 6 cells   -> outputs/study/encoder_scaling.csv
#
# Training and evaluation are independent: `eval` needs only the checkpoints and
# the fleet configs, so it can run later, on a different machine, or be repeated
# with different episode counts without retraining anything.
set -uo pipefail

MODE="${1:-}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
ENCODERS=(deepset transformer gnn)

# Trajectories per DAgger round, inversely proportional to the fleet size: each
# episode emits one LeRobot episode *per robot*, so this holds frames-per-round
# (and therefore optimizer steps per round) constant at ~20k across fleet sizes.
declare -A TRAJECTORIES=([2]=50 [4]=25 [6]=17 [8]=13)

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

train_one() {
  local encoder="$1" fleet_size="$2"
  local padded; padded="$(printf %02d "$fleet_size")"
  local name="${encoder}_n${padded}"

  docker_run python learning/train_dagger.py \
    --experiment-name "$name" \
    --system multi_robot \
    --expert-config "test/config/study/unicycle2_fleet_${padded}.yaml" \
    --policy-config "learning/config/study/${encoder}_mlp_config.yaml" \
    --dagger-iterations 5 \
    --trajectories-per-iteration "${TRAJECTORIES[$fleet_size]}" \
    --steps-per-trajectory 200 \
    --target-epochs-per-round 10 \
    --action-noise-std 0.03 \
    --expert-mix-beta-start 0.5 \
    --expert-mix-beta-decay-rate 0.25 \
    --expert-mix-decay-after-success-rate 0.0 \
    --eval-episodes 50 \
    --seed 99 \
    > "logs/${name}.log" 2>&1
  # Capture before anything else runs: a $(...) inside the echo would execute
  # first and reset $?, reporting every run as a success.
  local status=$?
  local stamp; stamp="$(date +%H:%M:%S)"
  if (( status == 0 )); then
    echo "[${stamp}] train ${name} OK"
  else
    echo "[${stamp}] train ${name} FAILED (exit ${status}) -- last log lines:"
    tail -n 5 "logs/${name}.log" | sed "s/^/    ${name}| /"
  fi
}

eval_one() {
  local encoder="$1" fleet_size="$2"
  local padded; padded="$(printf %02d "$fleet_size")"
  local name="${encoder}_n${padded}"
  local checkpoint="outputs/train_dagger_multi_robot/${name}/mlp_dagger_checkpoint.pt"

  if [[ ! -f "$checkpoint" ]]; then
    echo "[skip] ${name}: no checkpoint at ${checkpoint}"
    return
  fi

  # One CSV per policy: concurrent appends to a shared file would interleave rows.
  docker_run python test/evaluate_scaling.py \
    --checkpoint "$checkpoint" \
    --configs test/config/study/unicycle2_fleet_{02,04,06,08,16,32}.yaml \
    --episodes 50 \
    --steps 200 \
    --action-noise-std 0.0 \
    --seed-start 50000 \
    --output-csv "outputs/study/${name}.csv" \
    > "logs/eval_${name}.log" 2>&1
  local status=$?
  local stamp; stamp="$(date +%H:%M:%S)"
  if (( status == 0 )); then
    echo "[${stamp}] eval ${name} OK"
  else
    echo "[${stamp}] eval ${name} FAILED (exit ${status}) -- last log lines:"
    tail -n 5 "logs/eval_${name}.log" | sed "s/^/    ${name}| /"
  fi
}

mkdir -p logs outputs/study

case "$MODE" in
  train)
    # Descending fleet size: the N=8 runs are by far the longest, so they must
    # start immediately rather than being queued behind the cheap ones.
    for fleet_size in 8 6 4 2; do
      for encoder in "${ENCODERS[@]}"; do
        await_slot
        train_one "$encoder" "$fleet_size" &
      done
    done
    wait
    echo "All training runs finished. Checkpoints under outputs/train_dagger_multi_robot/"
    ;;

  eval)
    for encoder in "${ENCODERS[@]}"; do
      for fleet_size in 2 4 6 8; do
        await_slot
        eval_one "$encoder" "$fleet_size" &
      done
    done
    wait

    merged="outputs/study/encoder_scaling.csv"
    first=1
    : > "$merged"
    for csv in outputs/study/*_n??.csv; do
      [[ -f "$csv" ]] || continue
      if (( first )); then cat "$csv" >> "$merged"; first=0
      else tail -n +2 "$csv" >> "$merged"; fi
    done
    rows=$(( $(wc -l < "$merged") ))
    (( rows > 0 )) && rows=$(( rows - 1 ))
    echo "Merged ${rows} result rows into ${merged}"
    ;;

  *)
    echo "usage: $0 {train|eval}   (MAX_PARALLEL=${MAX_PARALLEL})" >&2
    exit 1
    ;;
esac
