#!/usr/bin/env bash
# Train every policy config against every expert config, in parallel Docker containers.
#
#   ./train.sh <experiment> <policy_dir|policy.yaml> <expert_dir|expert.yaml>
#
#   env: SEEDS="0 1 2"  (default "0")   MAX_PARALLEL=8
#
# Example, the 4-robot cell of the encoder study:
#
#   ./train.sh study2 learning/config/study/n04 test/config/study/unicycle2_fleet_04.yaml
#
# Policy configs with ring layouts (learning/config/study/n<NN>/) only fit the fleet size
# they were generated for, so pair a policy dir with expert configs of that size. A
# mismatch fails inside train_dagger.py and shows up in that run's log.
#
# Run <policy>_<expert>_s<seed> writes to
#   outputs/<experiment>/models/<run>/   checkpoints and saved configs
#   outputs/<experiment>/logs/<run>.log
#
# The DAgger schedule comes only from the policy config's 'training' section: this script
# passes train_dagger.py just the run identity, because any schedule flag would override
# the config.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

usage() {
  echo "usage: $0 <experiment> <policy_dir|policy.yaml> <expert_dir|expert.yaml>" >&2
  echo "       (SEEDS=\"${SEEDS:-0}\", MAX_PARALLEL=${MAX_PARALLEL:-8})" >&2
  exit 1
}

(( $# == 3 )) || usage
EXPERIMENT="$1"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
read -r -a SEEDS <<< "${SEEDS:-0}"

[[ "$EXPERIMENT" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid experiment name '${EXPERIMENT}'" >&2; exit 1; }

# Print the repo-relative YAML files a path argument names, one per line. The container
# mounts the repo at /workspace, so anything outside it would not exist in there.
#
# A directory holding both policy heads is the normal case, so the argument may also be a
# quoted glob ('.../n04/*_mlp.yaml') to train one head without splitting the directory.
# Quoted, because the shell would otherwise expand it into several arguments.
yaml_files() {
  local arg="$1" abs
  if [[ ! -e "$arg" && "$arg" == *[*?\[]* ]]; then
    local matches; mapfile -t matches < <(compgen -G "$arg" | sort)
    (( ${#matches[@]} )) || { echo "no files match: ${arg}" >&2; return 1; }
    local match
    for match in "${matches[@]}"; do yaml_files "$match" || return 1; done
    return 0
  fi
  abs="$(realpath -e -- "$arg" 2>/dev/null)" || { echo "no such path: ${arg}" >&2; return 1; }
  if [[ "$abs" != "$REPO_ROOT"/* ]]; then
    echo "${arg} is outside the repository (${REPO_ROOT}); the container cannot see it" >&2
    return 1
  fi
  if [[ -d "$abs" ]]; then
    local found; found="$(find "$abs" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) | sort)"
    [[ -n "$found" ]] || { echo "no .yaml files in ${arg}" >&2; return 1; }
    printf '%s\n' "$found" | sed "s|^${REPO_ROOT}/||"
  else
    [[ "$abs" == *.yaml || "$abs" == *.yml ]] || { echo "not a .yaml file: ${arg}" >&2; return 1; }
    printf '%s\n' "${abs#"$REPO_ROOT"/}"
  fi
}

POLICY_LIST="$(yaml_files "$2")" || exit 1
EXPERT_LIST="$(yaml_files "$3")" || exit 1
mapfile -t POLICIES <<< "$POLICY_LIST"
mapfile -t EXPERTS <<< "$EXPERT_LIST"

MODEL_DIR="outputs/${EXPERIMENT}/models"
LOG_DIR="outputs/${EXPERIMENT}/logs"
RESULT_DIR="$(mktemp -d)"
trap 'rm -rf "$RESULT_DIR"' EXIT
mkdir -p "$MODEL_DIR" "$LOG_DIR"

# -u/HOME/USER: without -u the container writes root-owned files into the bind mount,
# and the host uid has no passwd entry in the image, so getpass.getuser() (LeRobot's
# dataset stack calls it) fails unless USER is set.
# *_NUM_THREADS=1: IPOPT/BLAS and torch each grab every core otherwise, so parallel runs
# would oversubscribe the machine.
docker_run() {
  docker compose run --rm -T \
    -u "$(id -u):$(id -g)" -e HOME=/tmp -e USER=csvil \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    csvil "$@"
}

await_slot() {
  while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n; done
}

stem() {
  local name; name="$(basename "$1")"
  printf '%s' "${name%.*}"
}

train_one() {
  local policy="$1" expert="$2" seed="$3"
  local name; name="$(stem "$policy")_$(stem "$expert")_s${seed}"
  local log="${LOG_DIR}/${name}.log"

  docker_run python learning/train_dagger.py \
    --experiment-name "$name" \
    --system multi_robot \
    --expert-config "$expert" \
    --policy-config "$policy" \
    --seed "$seed" \
    --checkpoint-dir "$MODEL_DIR" \
    > "$log" 2>&1
  # Capture before anything else runs: a $(...) ahead of $? would reset it.
  local status=$?

  local stamp; stamp="$(date +%H:%M:%S)"
  if (( status == 0 )); then
    echo "[${stamp}] ${name} OK"
    touch "${RESULT_DIR}/${name}.ok"
  else
    echo "[${stamp}] ${name} FAILED (exit ${status}) -- last log lines:"
    tail -n 5 "$log" | sed "s|^|    ${name}\| |"
    touch "${RESULT_DIR}/${name}.failed"
  fi
}

total=$(( ${#POLICIES[@]} * ${#EXPERTS[@]} * ${#SEEDS[@]} ))
echo "experiment ${EXPERIMENT}: ${total} runs (${#POLICIES[@]} policies x ${#EXPERTS[@]} experts" \
     "x ${#SEEDS[@]} seeds), ${MAX_PARALLEL} at a time"
echo "  policies: ${POLICIES[*]}"
echo "  experts:  ${EXPERTS[*]}"
echo "  seeds:    ${SEEDS[*]}"

for expert in "${EXPERTS[@]}"; do
  for policy in "${POLICIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      await_slot
      train_one "$policy" "$expert" "$seed" &
    done
  done
done
wait

ok=$(find "$RESULT_DIR" -name '*.ok' | wc -l)
failed=$(find "$RESULT_DIR" -name '*.failed' | wc -l)
echo "done: ${ok} OK, ${failed} failed. Checkpoints in ${MODEL_DIR}/, logs in ${LOG_DIR}/"
(( failed == 0 ))
