#!/usr/bin/env bash
# Study 1 evaluation: every checkpoint under MODELS, one fixed success criterion
# (vel/omega 0.1, pos 0.2), one merged CSV per scenario. See docs/evaluation.md.
#
#   ./eval_study1.sh                          # random scenario
#   SCENARIO=circle ./eval_study1.sh          # antipodal ring
#   FLEETS="04" EPISODES=10 ./eval_study1.sh  # quick check

set -uo pipefail

MODELS="${MODELS:-outputs/study1/models}"
OUTDIR="${OUTDIR:-outputs/study1/eval}"
SCENARIO="${SCENARIO:-random}"
EPISODES="${EPISODES:-50}"
FLEETS="${FLEETS:-02 04 08 16 32}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
# One fixed criterion. It is a task definition, not a result, so it is stated in
# the caption rather than carried as a plot dimension. TOLERANCES="as_trained" or
# "as_trained study1" still works for a sensitivity check.
read -r -a TOLERANCES <<< "${TOLERANCES:-study1}"

case "$SCENARIO" in
  random) SRC_TEMPLATE="test/config/study/unicycle2_fleet_%s.yaml"; EXTRA_FLAGS=""; STEPS=200 ;;
  circle) SRC_TEMPLATE="test/config/study/circle/unicycle2_circle_%s.yaml"
          EXTRA_FLAGS="--use-config-start"; STEPS=400 ;;
  *) echo "unknown scenario '${SCENARIO}'; known: random circle" >&2; exit 1 ;;
esac
STEPS="${STEPS_OVERRIDE:-$STEPS}"

RESULTS="$OUTDIR/$SCENARIO"
mkdir -p "$RESULTS/configs"

# Tolerance variants are derived from the committed scenario configs at run time
# rather than checked in, so they cannot drift from the configs they copy.
for n in $FLEETS; do
  src="$(printf "$SRC_TEMPLATE" "$n")"
  [[ -f "$src" ]] || { echo "missing scenario config ${src}" >&2; exit 1; }
  sed -e 's/^\( *\)vel_tol: 0.05/\1vel_tol: 0.1/' \
      -e 's/^\( *\)omega_tol: 0.05/\1omega_tol: 0.1/' \
      -e 's/^\( *\)pos_tol: 0.1/\1pos_tol: 0.2/' \
      "$src" > "$RESULTS/configs/fleet_${n}_study1.yaml"
done

await_slot() { while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n; done; }

eval_one() {
  local run_dir="$1" tol="$2"
  local name; name="$(basename "$run_dir")"
  local checkpoint; checkpoint="$(ls "$run_dir"/*_dagger_checkpoint.pt 2>/dev/null | head -1)"
  if [[ -z "$checkpoint" ]]; then
    echo "[skip] ${name}: no *_dagger_checkpoint.pt in ${run_dir}"
    return
  fi
  # Trailing _s<digits> is the seed; everything before it identifies the cell.
  local seed="${name##*_s}"
  local configs=()
  for n in $FLEETS; do
    if [[ "$tol" == as_trained ]]; then
      configs+=("$(printf "$SRC_TEMPLATE" "$n")")
    else
      configs+=("$RESULTS/configs/fleet_${n}_study1.yaml")
    fi
  done
  local out="$RESULTS/${name}_${tol}.csv"
  # evaluate_scaling.py appends, so drop the previous result to keep re-runs
  # replacing rather than stacking.
  rm -f "$out"
  OMP_NUM_THREADS=1 python3 test/evaluate_scaling.py \
    --checkpoint "$checkpoint" \
    --configs "${configs[@]}" \
    --episodes "$EPISODES" --steps "$STEPS" --seed-start 50000 \
    --train-seed "$seed" \
    ${EXTRA_FLAGS} \
    --output-csv "$out" > "$RESULTS/${name}_${tol}.log" 2>&1 \
    && echo "  ${name} ${tol} OK" \
    || echo "  ${name} ${tol} FAILED; see $RESULTS/${name}_${tol}.log"
}

echo "scenario=${SCENARIO} episodes=${EPISODES} steps=${STEPS} fleets=${FLEETS}"
echo "models=${MODELS} -> ${RESULTS}"
shopt -s nullglob
for tol in "${TOLERANCES[@]}"; do
  for run_dir in "$MODELS"/*/; do
    await_slot
    eval_one "${run_dir%/}" "$tol" &
  done
done
wait

python3 - "$RESULTS" "${TOLERANCES[*]}" <<'PY'
import csv, glob, os, statistics, sys

results_dir, tolerances = sys.argv[1], sys.argv[2].split()
rows = []
for path in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
    stem = os.path.basename(path)[:-4]
    # Tolerance labels contain underscores, so match them as suffixes rather than
    # splitting on the last one.
    tol = next((t for t in tolerances if stem.endswith("_" + t)), None)
    if tol is None:
        continue
    name = stem[: -(len(tol) + 1)]
    cell, _, seed = name.rpartition("_s")
    for row in csv.DictReader(open(path)):
        row.update(cell=cell, seed=seed, tolerance=tol)
        rows.append(row)

if not rows:
    print("no results"); raise SystemExit

merged = os.path.join(results_dir, "study1.csv")
with open(merged, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print(f"\nwrote {merged}  ({len(rows)} rows)\n")

# Aggregate over seeds: a single seed is an anecdote, so the spread is shown next to
# the mean rather than hidden behind it.
groups = {}
for row in rows:
    groups.setdefault((row["tolerance"], int(row["eval_fleet_size"]), row["cell"]), []).append(row)

hdr = f"{'tolerance':12}{'N':>4}  {'cell':28}{'seeds':>6}{'success':>18}{'coll':>7}{'timeout':>9}{'pos_err':>9}{'ms/step':>9}"
print(hdr); print("-" * len(hdr))
for (tol, fleet, cell), group in sorted(groups.items()):
    successes = [float(r["success_rate"]) for r in group]
    mean = statistics.mean(successes)
    spread = f"{min(successes):.2f}-{max(successes):.2f}" if len(successes) > 1 else "     "
    print(f"{tol:12}{fleet:>4}  {cell:28}{len(group):>6}"
          f"{mean:>8.2f} [{spread:>9}]"
          f"{statistics.mean(float(r['collision_rate']) for r in group):>7.2f}"
          f"{statistics.mean(float(r['timeout_rate']) for r in group):>9.2f}"
          f"{statistics.mean(float(r['mean_goal_position_error']) for r in group):>9.3f}"
          f"{statistics.mean(float(r['mean_action_ms']) for r in group):>9.2f}")
PY
