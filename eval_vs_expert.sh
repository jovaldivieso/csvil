#!/usr/bin/env bash
# evaluate_policy.py (policy vs. expert) for one checkpoint at several fleet sizes,
# split into chunks of episodes that run in parallel, then pooled into one table.
#
#   CK=outputs/.../flow_dagger_checkpoint.pt OUT=outputs/plots/evaluate_policy/<name> \
#     ./eval_vs_expert.sh
#
#   env: FLEETS="2 4 6 8" EPISODES=200 CHUNK=10 STEPS=200 PARALLEL=$(nproc)
#        RUNNER=docker|local
#
# Episode i uses the same per-robot seeds as test/evaluate_scaling.py (50000 + i + 100*r),
# and success is the study criterion (0.2 m, speed and yaw rate below 0.1). The MPC
# expert runs next to the policy, which is what makes this slow: its cost grows steeply
# with the fleet size. Each chunk writes its own PDF, MP4 and log to OUT.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CK="${CK:?set CK to the flow_dagger_checkpoint.pt}"
OUT="${OUT:?set OUT to the output directory}"
FLEETS="${FLEETS:-2 4 6 8}"
EPISODES="${EPISODES:-200}"
CHUNK="${CHUNK:-10}"
STEPS="${STEPS:-200}"
PARALLEL="${PARALLEL:-$(nproc)}"
RUNNER="${RUNNER:-docker}"   # docker | local

# The container mounts only the repository, so both paths must lie inside it.
for path in "$CK" "$OUT"; do
  if [[ "$path" == /* || "$path" == *..* ]]; then
    echo "use a path relative to the repository root, inside it: ${path}" >&2
    exit 1
  fi
done
[[ -f "$CK" ]] || { echo "no checkpoint at ${CK}" >&2; exit 1; }
mkdir -p "$OUT"

run_chunk() {  # $1 fleet size, $2 first episode index
  local n="$1" start="$2" p; p="$(printf %02d "$n")"
  local stop=$(( start + CHUNK < EPISODES ? start + CHUNK : EPISODES ))
  local tag="n${p}_ep$(printf %03d "$start")"
  # Same episodes as evaluate_scaling.py: episode i, robot r -> seed 50000 + i + 100*r.
  local seeds
  seeds="$(python3 -c "print([[50000+i+100*r for i in range($start,$stop)] for r in range($n)])")"
  local cmd=(python test/evaluate_policy.py --system multi_robot --policy-type flow
    --config "test/config/study/unicycle2_fleet_${p}.yaml" --model-dir "$CK"
    --num-steps "$STEPS" --seeds "$seeds"
    --tolerance-overrides '{"pos_tol": 0.2, "vel_tol": 0.1, "omega_tol": 0.1}'
    --device cpu --output-path "$OUT/eval_${tag}.pdf")
  if [[ "$RUNNER" == docker ]]; then
    docker compose run --rm -T -u "$(id -u):$(id -g)" -e HOME=/tmp -e USER=csvil \
      -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
      csvil "${cmd[@]}" > "$OUT/eval_${tag}.log" 2>&1
  else
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "${cmd[@]}" > "$OUT/eval_${tag}.log" 2>&1
  fi
  echo "[$(date +%H:%M:%S)] ${tag} exit=$?"
}

# Largest fleet first: it is by far the slowest, so it must not queue behind the rest.
for n in $(printf '%s\n' $FLEETS | sort -rn); do
  for (( start = 0; start < EPISODES; start += CHUNK )); do
    while (( $(jobs -rp | wc -l) >= PARALLEL )); do wait -n; done
    run_chunk "$n" "$start" &
  done
done
wait

# Pool the chunks: rates are weighted by each chunk's episode count.
python3 - "$OUT" <<'EOF'
import glob, re, sys, collections
keys = ("success_rate", "policy_collision_rate", "policy_timeout_rate", "expert_collision_rate",
        "mean_policy_steps", "mean_expert_steps", "mean_policy_goal_error_l2",
        "mean_expert_goal_error_l2")
agg = collections.defaultdict(lambda: collections.defaultdict(float))
for log in sorted(glob.glob(f"{sys.argv[1]}/eval_n*_ep*.log")):
    n = int(re.search(r"eval_n(\d+)_", log).group(1))
    text = open(log).read()
    m = re.search(r"^num_trajectories: (\d+)", text, re.M)
    if not m:
        print(f"no summary in {log} (failed run?)"); continue
    k = int(m.group(1)); agg[n]["episodes"] += k
    for key in keys:
        v = re.search(rf"^{key}: ([\d.]+)", text, re.M)
        if v: agg[n][key] += k * float(v.group(1))
print(f"{'N':>3} {'eps':>4} {'success':>8} {'collide':>8} {'timeout':>8} {'exp_coll':>8} "
      f"{'pol_steps':>9} {'exp_steps':>9} {'pol_err':>8} {'exp_err':>8}")
for n in sorted(agg):
    a = agg[n]; e = a["episodes"]
    print(f"{n:>3} {int(e):>4} " + " ".join(f"{a[k]/e:>8.3f}" if "steps" not in k else f"{a[k]/e:>9.1f}" for k in keys))
EOF
