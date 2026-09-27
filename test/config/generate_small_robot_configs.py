"""Generate the 0.1 m-robot task: one expert config and the mlp/flow/safeflow policies.

Study 2 ran on a 1 m robot (``d_collision`` 1.0) doing 1 m/s. This writes the same
2-robot task for a **0.1 m robot doing 1.5 m/s and 6 rad/s**, and the three policy heads
trained on it. Both halves are emitted here, from one set of constants, because the
length scale has to appear in both and must not be able to disagree: the expert config
carries the robot and the box, the policy configs carry the ring layouts whose
coordinates are in the *same* metres.

Why a generator rather than four edited files: the policy configs hold 240 ring rollouts
of per-robot [x, y, theta, v, omega] and [x, y, theta], every coordinate of which is a
length. Scaling those by hand is what a script is for.

What scales, and what does not
------------------------------
The robot shrinks 10x, but the user-specified speeds do *not* shrink with it: 1.5 m/s at
0.1 m is 15 body lengths/s where the old robot did 1.0. So this is not a similarity
transform, and it is worth being explicit about which invariants survive.

Pure lengths, all x0.1:
    d_collision, d_safe, inter_robot_visibility_radius, workspace_bounds (both the
    expert's box and the policy configs' density override), pos_tol, and every
    coordinate of every ring layout -- including the radii and minimum-travel floor
    that generate_study_policy_configs.py states in absolute metres.

Set by the speeds rather than by the length scale:
    ``dt``. At the old 0.05 s a robot at 1.5 m/s covers 0.075 m per step, which is
    75% of d_collision -- two robots closing head-on would jump 1.5 diameters between
    consecutive collision checks, straight through each other, and the expert's own
    d_collision constraint is imposed only at knot points. dt 0.005 puts the per-step
    travel back at 7.5% of d_collision (the old task ran at 5.0%).

    Tolerances that are not lengths: vel_tol and omega_tol are held at the old
    *fraction of the limit* (5% of max_linear_vel, 2.5% of max_angular_vel) rather than
    scaled as lengths, which also keeps pos_tol/vel_tol -- the time a robot sitting at
    its goal takes to drift back out of tolerance -- at the old ~30 steps.

Assumed, because the platform was not specified:
    ``max_linear_accel`` 8.0 m/s^2 is the friction limit of a wheeled robot on a hard
    floor (mu*g, mu~0.8); ``max_angular_accel`` 250 rad/s^2 is the matching
    friction-limited yaw acceleration for a 0.1 m differential drive
    (mu*g*(track/2)/k^2, track 0.08 m, radius of gyration 0.035 m). Raise --accel if the
    platform is aerial -- 2 g is ordinary there, and it would let safeflow's
    prediction_horizon drop back to the flow head's 10 (see HEADS).

    These two numbers are the only free parameters left, and they are the ones that
    decide whether the task resembles the old one. At 8 m/s^2 a robot at 1.5 m/s stops
    in 0.141 m = 1.41 diameters, against the old robot's 0.17: the new task is ~8x more
    ballistic relative to the robot, and there is no choice of dt or box that undoes
    that. Reported, not hidden, by print_derivation().

The episode budget, which is the part that actually broke
------------------------------------------------------
An episode ends when unicycle2 reaches a goal *pose*, and a unicycle can only change
heading by turning -- so every episode finishes with an in-place rotation, and is_done
additionally wants |omega| below omega_tol, so the spin-down counts too. That rotation is
the one part of the task that barely benefits from the smaller robot: max_angular_vel is
set by the robot, not by the box, so a pi turn costs 0.55 s here against 1.70 s before,
a 3x saving where the translation got 10x cheaper. It is 23% of a 150-step episode on the
1 m robot and 73% of one here.

Sizing the budget from the box diagonal alone therefore fails, and did: at 150 steps the
expert reached only 3 of 8 goals, every failure sitting exactly on its goal
(position error 2e-4 m) and still rotating. Two hypotheses were tested and both were
wrong -- lengthening the expert horizon (40/80/120) made it strictly worse (3/5, 0/5,
0/5), and so did restoring the old angular cost weights. The budget was the whole story:
the modelled traverse+settle is 212 steps, the measured worst case over 8 seeded episodes
is 205, and 250 steps gives 8/8 with no collision. The min pairwise distances then match
the 1 m task's to two decimals (1.48/1.20/1.59/1.69 against 1.50/1.20/1.58/1.68), which
is the real evidence that the scaling is right.

Episodes per round follow from FRAMES_PER_ROUND rather than being set, so a longer budget
costs no expert time: 288 x 250 is the same 144 000 frames and the same 72 000 solves per
round as data_mid's 480 x 150.

Cost weights
------------
Q_diag / R_diag / collision_slack_penalty_weight are not dimensionless. Each weights a
squared physical quantity, so holding a *weight* fixed while its quantity shrinks 10x
silently re-tunes the expert. Every weight is therefore multiplied by
(old characteristic magnitude / new characteristic magnitude)^2, which leaves every cost
*term* at the magnitude the old configs produced -- the position term still dominates
the heading term by the same factor, and IPOPT still sees the same absolute numbers.
R becomes per-dimension (``R_diag``) because a_v and a_omega no longer scale alike.

One task, several schedules
---------------------------
``--name``/``--rounds``/``--episodes`` write another policy directory against the *same*
robot, so a long run and a cheap one stay comparable. Note that every invocation also
rewrites the one shared expert config: that is deliberate -- two schedules on two
different tasks would not be comparable -- but it means a ``--accel`` passed to one
invocation silently changes the task the other one's checkpoints were trained on. Pass
the same ``--accel`` to both, or regenerate both after changing it.

Episodes per round are derived from FRAMES_PER_ROUND unless ``--episodes`` is given, so
the long run pays nothing for a longer episode budget while a cheap run sizes itself.

Usage:
    # long run: 5 rounds, episodes derived to hold 144 000 frames/round
    python test/config/generate_small_robot_configs.py --name small_big
    # cheap run: same task, 3 rounds x 150 episodes
    python test/config/generate_small_robot_configs.py --name small_med \\
        --rounds 3 --episodes 150
    python test/config/generate_small_robot_configs.py --accel 20 600 --dry-run
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import sys
import textwrap
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "_generate_study_policy_configs",
    PROJECT_ROOT / "learning/config/study/generate_study_policy_configs.py",
)
_study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_study)

from core.config import validate_system_config  # noqa: E402

NUM_ROBOTS = 2
# The task this one is derived from: the 1 m robot of study 2, and the 3x-density box
# the data_mid / data_mid_flow pilots actually trained in (the expert config's own box is
# +-3.0; the pilots override it in the policy config's training section, and this keeps
# that arrangement).
REFERENCE_FLEET_CONFIG = PROJECT_ROOT / "test/config/study/unicycle2_fleet_02.yaml"
REFERENCE_PILOT_HALF_WIDTH = 1.732

EXPERT_OUT = PROJECT_ROOT / "test/config/study/unicycle2_fleet_02_small.yaml"
POLICY_OUT_ROOT = PROJECT_ROOT / "learning/config/study"

# Length scale: d_collision 1.0 m -> 0.1 m.
LENGTH_SCALE = 0.1

# Specified by hand, not derived: the robot's own limits.
MAX_LINEAR_VEL = 1.5
MAX_ANGULAR_VEL = 6.0
# Assumed (see the module docstring). --accel overrides.
DEFAULT_ACCEL = (8.0, 150.0)

# Per-step travel at max_linear_vel, as a fraction of d_collision. The old task ran at
# 0.050; 0.075 is the closest that a round dt gets at 1.5 m/s, and it keeps the
# collision checks far away from tunnelling.
DT = 0.005

# Heads, and the prediction horizon each is generated at.
#
# mlp 1 and flow 10 are the study's own values, unchanged: at dt 0.005 a 10-step chunk
# spans 0.075 m = 0.75 d_collision of travel, where the old 10 steps at dt 0.05 spanned
# 0.50, so the chunk is if anything deeper than before in the units that matter.
#
# safeflow is the exception and it is not cosmetic. Its projector enforces come-to-rest
# at the end of *this* horizon, and braking from 1.5 m/s at 8 m/s^2 takes 0.1875 s = 38
# steps. At 10 steps (0.05 s) the terminal-rest term is simply unreachable whenever the
# robot is moving at all, which is the regime the code's own comments record as making
# the solve "spend the entire horizon braking" -- the fallback path would carry the run.
# 40 steps covers the brake with the same margin the old configs had (brake time /
# horizon 0.94, against the old 0.67). Only action[0] is ever executed
# (rollouts.py:128), so this buys projector reach, not open-loop commitment; it costs
# inference time, roughly with the horizon.
HEADS = {
    "mlp": (1, "single-step regression"),
    "flow": (10, "action chunk; the generative head's intended setting"),
    "safeflow": (40, "covers the 38-step brake from max_linear_vel; see the generator"),
}
ENCODER = "deepset"

# The DAgger schedule of the data_mid_flow pilots, unchanged except that all three heads
# now share one setting so the comparison is between heads and nothing else. eval_episodes
# and the ring fraction are the data_mid_flow values (one round's worth of evaluation,
# half the round from rings); data_mid's mlp configs used 20 and a third, which would have
# scored the mlp against a different in-training gate than the two flow heads.
DAGGER_ROUNDS = 5
TARGET_EPOCHS = 40
RING_FRACTION = 0.5
ACTION_NOISE_STD = 0.03

# Frames per DAgger round. The study holds this equal across every cell (one episode
# yields one dataset episode per robot), and the expert dominates the wall clock at
# steps*episodes solves per round -- so fixing frames fixes both the dataset size and
# the ~15 h per run, and the episode count follows from the budget rather than being
# chosen. data_mid used 480 x 150; this is 288 x 250, the same 144 000 frames and the
# same 72 000 expert solves.
FRAMES_PER_ROUND = 144_000

# Margin over the modelled traverse+settle time, and the rounding grid. 1.18 is what the
# 1 m configs' own budgets work out to, and it is confirmed here: the modelled 212 steps
# against a measured worst case of 205 over 8 seeded episodes, with 250 giving 8/8 and
# 150 only 3/8. Do not lower the budget without rerunning that measurement -- a budget
# that cuts episodes off mid-rotation teaches the policy to arrive and keep spinning.
BUDGET_MARGIN = 1.18
BUDGET_GRID = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--accel", type=float, nargs=2, default=list(DEFAULT_ACCEL),
        metavar=("MAX_LINEAR_ACCEL", "MAX_ANGULAR_ACCEL"),
        help=f"the robot's acceleration limits, m/s^2 and rad/s^2 "
             f"(default {DEFAULT_ACCEL[0]} {DEFAULT_ACCEL[1]}, a wheeled robot's friction "
             f"limits). Every cost weight and the reported ratios follow from these.",
    )
    parser.add_argument(
        "--name", default="small_big",
        help="output directory prefix; configs land in "
             "learning/config/study/<name>_n02 (default small_big)",
    )
    parser.add_argument(
        "--rounds", type=int, default=DAGGER_ROUNDS,
        help=f"DAgger rounds (default {DAGGER_ROUNDS})",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="episodes per round. Omitted, it is derived from FRAMES_PER_ROUND so that "
             "a longer episode budget costs no expert time; give it explicitly to size a "
             "cheaper run, which then collects proportionally fewer frames.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="print the derivation and the checks, write nothing")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------

def scaled_weight(old_weight: float, old_magnitude: float, new_magnitude: float) -> float:
    """``old_weight`` re-expressed so its cost term keeps the magnitude it had.

    Every entry of Q_diag and R_diag multiplies the square of a physical quantity, so a
    weight left alone while its quantity changes by ``new/old`` changes that term's
    contribution by the square of it. This is the correction, and nothing more: cost is
    invariant to an overall positive factor, so only the *ratios* between the returned
    weights carry meaning -- the absolute level is chosen to match the old configs so
    IPOPT sees familiar numbers.
    """
    return old_weight * (old_magnitude / new_magnitude) ** 2


def derive(accel: tuple[float, float]) -> dict[str, object]:
    """Every number the two output files need, from the constants above."""
    max_linear_accel, max_angular_accel = accel
    reference = yaml.safe_load(REFERENCE_FLEET_CONFIG.read_text())
    ref_robot = reference["robots"][0]["config"]

    # Characteristic magnitudes, old -> new, one per state/action dimension. These are
    # what scaled_weight divides by; any consistent choice of "characteristic" works
    # because only ratios between weights matter, so the box half-width stands in for
    # position and each limit for its own dimension.
    ref_half_width = float(ref_robot["workspace_bounds"][1])
    half_width = round(ref_half_width * LENGTH_SCALE, 4)
    pilot_half_width = round(REFERENCE_PILOT_HALF_WIDTH * LENGTH_SCALE, 4)

    magnitudes = {
        "position": (ref_half_width, half_width),
        "theta": (1.0, 1.0),  # radians either way
        "v": (float(ref_robot["max_linear_vel"]), MAX_LINEAR_VEL),
        "omega": (float(ref_robot["max_angular_vel"]), MAX_ANGULAR_VEL),
        "a_v": (float(ref_robot["max_linear_accel"]), max_linear_accel),
        "a_omega": (float(ref_robot["max_angular_accel"]), max_angular_accel),
    }

    ref_q = [float(value) for value in reference["Q_diag"][:5]]
    ref_r = float(reference["R_weight"])
    q_block = [
        scaled_weight(ref_q[0], *magnitudes["position"]),
        scaled_weight(ref_q[1], *magnitudes["position"]),
        scaled_weight(ref_q[2], *magnitudes["theta"]),
        scaled_weight(ref_q[3], *magnitudes["v"]),
        scaled_weight(ref_q[4], *magnitudes["omega"]),
    ]
    r_block = [
        scaled_weight(ref_r, *magnitudes["a_v"]),
        scaled_weight(ref_r, *magnitudes["a_omega"]),
    ]

    return {
        "reference": reference,
        "ref_robot": ref_robot,
        "max_linear_accel": max_linear_accel,
        "max_angular_accel": max_angular_accel,
        "half_width": half_width,
        "pilot_half_width": pilot_half_width,
        "d_safe": round(float(reference["d_safe"]) * LENGTH_SCALE, 6),
        "d_collision": round(float(reference["d_collision"]) * LENGTH_SCALE, 6),
        "visibility": round(float(reference["inter_robot_visibility_radius"]) * LENGTH_SCALE, 6),
        "q_block": [round(value, 6) for value in q_block],
        "r_block": [round(value, 8) for value in r_block],
        # The one weight whose term is *linear*, not quadratic: the planner adds
        # `weight * sum(slack)` (casadi_planner.py), and slack relaxes a squared
        # distance, so it carries m^2 and the correction is (old/new)^1 = 100, not the
        # 10000 a quadratic term would want. Squaring it here would over-weight the
        # collision penalty a hundredfold against every other term -- the expert would
        # refuse to come near another robot at all.
        "slack_weight": float(reference["collision_slack_penalty_weight"])
        * (float(reference["d_safe"]) ** 2)
        / ((float(reference["d_safe"]) * LENGTH_SCALE) ** 2),
        # The projector's own terminal-velocity cost and its fallback acceptance band:
        # one weight on a squared velocity, one tolerance *in* velocity.
        "terminal_velocity_weight": round(scaled_weight(1.0, *magnitudes["v"]), 6),
        "fallback_terminal_velocity_tol": round(
            1e-2 * MAX_LINEAR_VEL / float(ref_robot["max_linear_vel"]), 6
        ),
        "tolerances": {
            "pos_tol": round(float(ref_robot["pos_tol"]) * LENGTH_SCALE, 6),
            "theta_tol": float(ref_robot["theta_tol"]),
            "vel_tol": round(
                float(ref_robot["vel_tol"]) * MAX_LINEAR_VEL / float(ref_robot["max_linear_vel"]), 6
            ),
            "omega_tol": round(
                float(ref_robot["omega_tol"]) * MAX_ANGULAR_VEL / float(ref_robot["max_angular_vel"]),
                6,
            ),
        },
    }


def traverse_steps(distance: float, max_vel: float, accel: float, dt: float) -> tuple[float, float]:
    """Steps to cover ``distance`` from rest to rest under bang-bang, capped at max_vel.

    Translation only -- see settle_steps for the other half of the episode. Not the same
    as distance/max_vel: at these accelerations the robot spends most of a crossing
    speeding up and slowing down, and on a short enough trip never reaches max_vel.
    """
    accel_distance = max_vel * max_vel / (2.0 * accel)
    if 2.0 * accel_distance >= distance:
        peak = math.sqrt(accel * distance)
        return 2.0 * peak / accel / dt, peak
    seconds = 2.0 * (max_vel / accel) + (distance - 2.0 * accel_distance) / max_vel
    return seconds / dt, max_vel


def settle_steps(max_angular_vel: float, max_angular_accel: float, dt: float) -> float:
    """Steps for the worst-case final in-place rotation, up to pi and back to rest.

    This term is why sizing an episode budget from the translation alone is wrong, and
    it is the mistake that made the first generated configs fail: unicycle2 reaches a
    goal *pose*, and a unicycle can only change heading by turning, so every episode
    ends with a rotation that the straight-line traverse does not account for.
    is_done additionally requires |omega| < omega_tol, so the spin-down counts too.

    It matters here and not in the 1 m configs because it is the one part of the episode
    that barely scales: max_angular_vel is capped by the robot, not by the box, so a pi
    turn costs 0.55 s against the old 1.70 s -- a 3x saving where the translation got 10x
    cheaper. Measured share of a 150-step episode: 23% on the 1 m robot, 73% here.
    """
    return (math.pi / max_angular_vel + max_angular_vel / max_angular_accel) / dt


def episode_budget(
    derived: dict[str, object], episodes_override: int | None = None
) -> tuple[int, int, float]:
    """(steps per episode, episodes per round, modelled traverse+settle steps).

    The budget has to cover both halves of an episode -- driving across the box and then
    turning to the goal heading -- and the episode count then follows from
    FRAMES_PER_ROUND, so that lengthening episodes costs nothing in expert time.
    """
    diagonal = 2.0 * float(derived["pilot_half_width"]) * math.sqrt(2.0)
    traverse, _ = traverse_steps(
        diagonal, MAX_LINEAR_VEL, float(derived["max_linear_accel"]), DT
    )
    settle = settle_steps(MAX_ANGULAR_VEL, float(derived["max_angular_accel"]), DT)
    needed = traverse + settle
    steps = int(round(needed * BUDGET_MARGIN / BUDGET_GRID) * BUDGET_GRID)
    episodes = (
        int(episodes_override)
        if episodes_override is not None
        else int(round(FRAMES_PER_ROUND / (steps * NUM_ROBOTS)))
    )
    return steps, episodes, needed


def print_derivation(derived: dict[str, object], episodes_override: int | None = None) -> None:
    """The dimensionless comparison, old task against new, so a regression is visible.

    Every row is a ratio a reader can check. The three that do not carry over are
    printed with their factor rather than quietly rounded, because they are the whole
    reason this is a new task and not a rescaled one.
    """
    ref_robot = derived["ref_robot"]
    old = dict(
        d_coll=float(derived["reference"]["d_collision"]),
        half=REFERENCE_PILOT_HALF_WIDTH,
        dt=float(derived["reference"]["dt"]),
        v=float(ref_robot["max_linear_vel"]),
        w=float(ref_robot["max_angular_vel"]),
        av=float(ref_robot["max_linear_accel"]),
        aw=float(ref_robot["max_angular_accel"]),
        horizon=int(derived["reference"]["horizon"]),
        flow_h=10,
    )
    new = dict(
        d_coll=float(derived["d_collision"]),
        half=float(derived["pilot_half_width"]),
        dt=DT,
        v=MAX_LINEAR_VEL,
        w=MAX_ANGULAR_VEL,
        av=float(derived["max_linear_accel"]),
        aw=float(derived["max_angular_accel"]),
        horizon=int(derived["reference"]["horizon"]),
        flow_h=HEADS["safeflow"][0],
    )

    def ratios(p: dict[str, float]) -> dict[str, float]:
        box, diagonal = 2.0 * p["half"], 2.0 * p["half"] * math.sqrt(2.0)
        stop = p["v"] ** 2 / (2.0 * p["av"])
        steps, peak = traverse_steps(diagonal, p["v"], p["av"], p["dt"])
        return {
            "box / d_collision": box / p["d_coll"],
            "travel per step / d_collision": p["v"] * p["dt"] / p["d_coll"],
            "stop distance / d_collision": stop / p["d_coll"],
            "turn radius at v_max / d_collision": (p["v"] / p["w"]) / p["d_coll"],
            "expert lookahead / d_collision": p["v"] * p["horizon"] * p["dt"] / p["d_coll"],
            # Reported because it is the one ratio that moves most and is easy to miss:
            # the robot is 10x smaller but its braking time only falls 1.8x
            # (accelerations are physically capped, not scaled), so 40 steps went from
            # 2.0 s of lookahead to 0.2 s. Worth knowing, but *measured not to be* the
            # thing that decides whether the expert settles -- sweeping the horizon at
            # 40/80/120 made the goal-spinning strictly worse (3/5, 0/5, 0/5 success),
            # because a longer horizon gives the optimizer more room to exploit a
            # cheap angular channel. Do not "fix" a settling problem here.
            "brake time / expert horizon": (p["v"] / p["av"]) / (p["horizon"] * p["dt"]),
            "brake time / safeflow horizon": (p["v"] / p["av"]) / (p["flow_h"] * p["dt"]),
            "steps to max_linear_vel": p["v"] / p["av"] / p["dt"],
            "steps to max_angular_vel": p["w"] / p["aw"] / p["dt"],
            "rad turned per step": p["w"] * p["dt"],
            "diagonal traverse, steps": steps,
            "peak speed on the diagonal": peak,
        }

    # Which deviations matter. A ratio that drifts is only a problem if drifting in that
    # direction changes the task rather than the discretisation, so each row says so
    # rather than leaving a bare factor to be eyeballed.
    NOTES = {
        "travel per step / d_collision": "finer is safer; 0.075 is far from tunnelling",
        "stop distance / d_collision": "MATERIAL: set by max_linear_accel, see docstring",
        "turn radius at v_max / d_collision": "MATERIAL: fixed by the given v and omega",
        "expert lookahead / d_collision": "longer is safer, costs solve time",
        "brake time / expert horizon": "informational; raising horizon does NOT help settling",
        "brake time / safeflow horizon": "under 1 means the projector can brake in time",
        "steps to max_linear_vel": "MATERIAL: same cause as stop distance",
        "steps to max_angular_vel": "benign",
        "rad turned per step": "finer is safer",
        "diagonal traverse, steps": "sizes the episode budget",
        "peak speed on the diagonal": "absolute m/s, not a ratio: v_max is just reachable",
    }
    old_r, new_r = ratios(old), ratios(new)
    print(f"{'dimensionless':38} {'1 m robot':>10} {'0.1 m':>10}  factor")
    for key in old_r:
        factor = new_r[key] / old_r[key] if old_r[key] else float("inf")
        note = NOTES.get(key, "")
        print(f"  {key:36} {old_r[key]:10.3f} {new_r[key]:10.3f} {factor:6.2f}x  {note}")
    steps, episodes, needed = episode_budget(derived, episodes_override)
    settle = settle_steps(MAX_ANGULAR_VEL, float(derived["max_angular_accel"]), DT)
    print(
        f"\n  Episode budget {steps} steps = {needed:.0f} needed "
        f"({new_r['diagonal traverse, steps']:.0f} traverse + {settle:.0f} settle) "
        f"x {BUDGET_MARGIN}; {episodes} episodes/round = "
        f"{episodes * steps * NUM_ROBOTS} frames, {episodes * steps} expert solves."
    )


# ---------------------------------------------------------------------------
# expert config
# ---------------------------------------------------------------------------

def build_expert_config(derived: dict[str, object]) -> dict[str, object]:
    robot_config = {
        "dt": DT,
        "max_linear_accel": derived["max_linear_accel"],
        "max_angular_accel": derived["max_angular_accel"],
        "max_linear_vel": MAX_LINEAR_VEL,
        "max_angular_vel": MAX_ANGULAR_VEL,
        "randomize_goal": True,
        **derived["tolerances"],
        "workspace_bounds": [-derived["half_width"], derived["half_width"]],
    }
    return {
        "dt": DT,
        "d_safe": derived["d_safe"],
        "d_collision": derived["d_collision"],
        "inter_robot_visibility_radius": derived["visibility"],
        "horizon": int(derived["reference"]["horizon"]),
        "mode": "mpc",
        "R_diag": derived["r_block"] * NUM_ROBOTS,
        "terminal_cost_multiplier": float(derived["reference"]["terminal_cost_multiplier"]),
        "collision_slack_penalty_weight": float(derived["slack_weight"]),
        "terminal_velocity_weight": derived["terminal_velocity_weight"],
        "fallback_terminal_velocity_tol": derived["fallback_terminal_velocity_tol"],
        "initial_state_seed": int(derived["reference"]["initial_state_seed"]),
        "action_noise_seed": int(derived["reference"]["action_noise_seed"]),
        "Q_diag": list(derived["q_block"]) * NUM_ROBOTS,
        "robots": [
            {"system": "unicycle2", "config": copy.deepcopy(robot_config)}
            for _ in range(NUM_ROBOTS)
        ],
    }


class _FlowListDumper(yaml.SafeDumper):
    """Renders scalar lists inline, so bound pairs stay on one line."""


def _represent_list(dumper: yaml.Dumper, data: list) -> yaml.Node:
    inline = all(isinstance(item, (int, float, str, bool)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


_FlowListDumper.add_representer(list, _represent_list)


def render_expert(config: dict[str, object], derived: dict[str, object]) -> str:
    q_block = derived["q_block"]
    q_lines = []
    for robot_idx in range(NUM_ROBOTS):
        values = ", ".join(str(value) for value in q_block)
        prefix = "Q_diag: [" if robot_idx == 0 else " " * 9
        suffix = "]" if robot_idx == NUM_ROBOTS - 1 else ","
        q_lines.append(f"{prefix}{values}{suffix}")
    scalars = {k: v for k, v in config.items() if k not in {"Q_diag", "robots"}}
    header = (
        f"# {NUM_ROBOTS}x unicycle2 fleet, 0.1 m robot at {MAX_LINEAR_VEL} m/s "
        f"/ {MAX_ANGULAR_VEL} rad/s.\n"
        f"# Generated by test/config/generate_small_robot_configs.py -- do not edit by hand.\n"
        f"#\n"
        f"# {REFERENCE_FLEET_CONFIG.name} at a {LENGTH_SCALE}x length scale. Lengths scale;\n"
        f"# the speeds were specified and do not, so dt falls to {DT} to keep the per-step\n"
        f"# travel ({MAX_LINEAR_VEL * DT:.4f} m) well inside d_collision, and every cost weight\n"
        f"# is restated so its term keeps the magnitude the 1 m configs produced.\n"
        f"# max_linear_accel / max_angular_accel are *assumed* wheeled-robot friction limits,\n"
        f"# not measured -- they set how ballistic the task is (stopping distance "
        f"{MAX_LINEAR_VEL ** 2 / (2 * float(derived['max_linear_accel'])):.3f} m =\n"
        f"# {MAX_LINEAR_VEL ** 2 / (2 * float(derived['max_linear_accel'])) / float(derived['d_collision']):.2f} "
        f"d_collision, against 0.17 on the 1 m robot). Rerun with --accel to change them.\n"
        f"#\n"
        f"# Box +-{derived['half_width']} here; the policy configs override it to "
        f"+-{derived['pilot_half_width']}\n"
        f"# (3x density), as the data_mid pilots did.\n"
    )
    return (
        header
        + yaml.safe_dump(scalars, sort_keys=False, default_flow_style=None)
        + "\n".join(q_lines) + "\n"
        + yaml.dump({"robots": config["robots"]}, Dumper=_FlowListDumper, sort_keys=False)
    )


# ---------------------------------------------------------------------------
# policy configs
# ---------------------------------------------------------------------------

def scale_layout_constants(derived: dict[str, object]) -> None:
    """Rescale the ring-layout module constants that are stated in absolute metres.

    generate_study_policy_configs.py holds NOMINAL_RADIUS, RADIUS_RANGE and
    MIN_TRAVEL_DISTANCE as metres for a 1 m robot; MIN_SEPARATION_PER_D_SAFE is already
    in units of d_safe and so needs nothing. Patching the module rather than
    reimplementing build_rollouts keeps one copy of the layout logic -- the jitter
    redraws, the ellipse relaxation and the box fitting are all separation-critical.
    """
    _study.NOMINAL_RADIUS *= LENGTH_SCALE
    _study.RADIUS_RANGE = tuple(value * LENGTH_SCALE for value in _study.RADIUS_RANGE)
    _study.MIN_TRAVEL_DISTANCE *= LENGTH_SCALE


def validate_layouts(validated: dict, initial_states, goal_states, d_safe: float) -> float:
    """Push the layouts through the same normalizers train_dagger.py uses.

    _study.validate does this too, but reloads the fleet config from disk, which would
    make --dry-run depend on a file it deliberately has not written -- and, worse, would
    silently validate against a stale config on a rerun. Takes the in-memory validated
    config instead.
    """
    from core.factory import DynamicsFactory
    from systems.initial_state_utils import (
        normalize_goal_state_specs,
        normalize_initial_state_specs,
    )

    simulator = DynamicsFactory.create(system_name="multi_robot", config=validated)
    states = normalize_initial_state_specs(simulator, initial_states)
    goals = normalize_goal_state_specs(simulator, goal_states)
    if len(states) != len(initial_states) or len(goals) != len(goal_states):
        raise SystemExit("Normalizer collapsed the rollout list; check the nesting.")
    closest = math.inf
    for state in states:
        positions = [
            (float(state[state_slice.start]), float(state[state_slice.start + 1]))
            for state_slice in simulator.robot_state_slices
        ]
        closest = min(closest, _study.min_pairwise_distance(positions))
    if closest < d_safe:
        raise SystemExit(f"Starting pair at {closest:.4f} < d_safe {d_safe}.")
    return closest


def training_schedule(
    derived: dict[str, object], steps: int, episodes: int, rounds: int
) -> dict[str, object]:
    return {
        "dagger_iterations": rounds,
        "trajectories_per_iteration": [episodes] * rounds,
        "steps_per_trajectory": steps,
        "target_epochs_per_round": [TARGET_EPOCHS] * rounds,
        "action_noise_std": ACTION_NOISE_STD,
        "expert_mix_beta_start": 0.5,
        "expert_mix_beta_decay_rate": 0.25,
        "expert_mix_decay_after_eval_success": 0.5,
        "expert_mix_beta_recovery": 0.75,
        "expert_mix_beta_recovery_increment": 0.25,
        # One round's worth, the data_mid_flow setting: the in-training evaluation gates
        # the beta decay, and at 20 episodes that gate sits inside its own noise. It also
        # takes its first min(rings, episodes) episodes from the ring list, so this count
        # is what reproduces the training mix exactly.
        "eval_episodes": episodes,
        "workspace_bounds": [-derived["pilot_half_width"], derived["pilot_half_width"]],
    }


def main() -> None:
    args = parse_args()
    derived = derive(tuple(args.accel))

    print(f"0.1 m robot: v={MAX_LINEAR_VEL} w={MAX_ANGULAR_VEL} "
          f"a=({derived['max_linear_accel']}, {derived['max_angular_accel']}) dt={DT}\n")
    print_derivation(derived, args.episodes)

    expert_config = build_expert_config(derived)
    validated = validate_system_config(system_name="multi_robot", raw_config=expert_config)
    print(f"\nexpert config validates. d_safe={derived['d_safe']} "
          f"d_collision={derived['d_collision']} visibility={derived['visibility']}")
    print(f"  Q_diag per robot {derived['q_block']}")
    print(f"  R_diag per robot {derived['r_block']}")
    print(f"  collision_slack_penalty_weight {derived['slack_weight']:.6g}")
    print(f"  tolerances {derived['tolerances']}")

    if not args.dry_run:
        EXPERT_OUT.write_text(render_expert(expert_config, derived))
        print(f"\nwrote {EXPERT_OUT.relative_to(PROJECT_ROOT)}")

    # The fleet has to be placeable in the scaled box before anything is trained on it:
    # the sampler redraws until every pair clears d_safe and then gives up, and at a 10x
    # smaller box with a 10x smaller d_safe the packing density is unchanged -- which is
    # exactly the claim worth checking rather than assuming.
    mean_visible = _study_placeable(validated)
    print(f"  fleet is placeable; mean visible neighbours ~{mean_visible:.2f}")

    scale_layout_constants(derived)
    steps, episodes, _ = episode_budget(derived, args.episodes)
    count = round(episodes * RING_FRACTION)
    max_radius = min(_study.RADIUS_RANGE[1], float(derived["pilot_half_width"]))
    initial_states, goal_states, kind_counts = _study.build_rollouts(
        NUM_ROBOTS, count, max_radius, float(derived["d_safe"]), antipodal_only=True
    )
    closest = validate_layouts(
        validated, initial_states, goal_states, float(derived["d_safe"])
    )
    print(f"  {count} ring rollouts, radii {_study.RADIUS_RANGE[0]:.4g}-{max_radius:.4g}, "
          f"closest starting pair {closest:.4f} (d_safe {derived['d_safe']})")

    if args.dry_run:
        print("\ndry run: no policy configs written")
        return

    layout_summary = textwrap.fill(
        "Layouts: " + ", ".join(f"{kind} x{n}" for kind, n in sorted(kind_counts.items())) + ".",
        width=86, initial_indent="  # ", subsequent_indent="  # ",
    )
    out_dir = POLICY_OUT_ROOT / f"{args.name}_n{NUM_ROBOTS:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # train.sh trains every YAML in the directory, so a file left over from a removed
    # head would silently keep being trained.
    for stale in out_dir.glob("*.yaml"):
        stale.unlink()

    template_dir = PROJECT_ROOT / "learning/config/study"
    for head, (prediction_horizon, horizon_note) in HEADS.items():
        template_path = template_dir / f"{ENCODER}_{head}_config.yaml"
        out_path = out_dir / f"{ENCODER}_{head}.yaml"
        text = (
            f"# N={NUM_ROBOTS} {ENCODER} {head} on the 0.1 m robot: "
            f"{args.rounds} rounds x {episodes} episodes.\n"
            f"# Generated by test/config/generate_small_robot_configs.py -- do not edit by hand.\n"
            f"#\n"
            f"# Trained against {EXPERT_OUT.relative_to(PROJECT_ROOT)}, but with starts and\n"
            f"# goals drawn from +-{derived['pilot_half_width']} instead of that config's "
            f"+-{derived['half_width']}:\n"
            f"# {NUM_ROBOTS / (2.0 * float(derived['pilot_half_width'])) ** 2:.2f} robots/m^2, "
            f"3x the study's training density, as the data_mid pilots used.\n"
            f"# {args.rounds} rounds x {episodes} episodes x {TARGET_EPOCHS} "
            f"epochs x {steps} steps;\n"
            f"# {episodes * steps * NUM_ROBOTS} frames per round "
            f"(one dataset episode per robot).\n"
            f"# The schedule is the data_mid_flow one, identical across all three heads.\n"
            + _study.template_body(template_path.read_text(), prediction_horizon, horizon_note)
            + "\ntraining:\n"
            + _study.format_schedule(training_schedule(derived, steps, episodes, args.rounds))
            + "  # No tolerance_overrides: convergence tolerances belong to the scenario\n"
            + "  # config, so training and evaluation share one definition of success.\n"
            + f"  # {count} of the {episodes} episodes per round "
            f"({RING_FRACTION:.0%}) start from antipodal\n"
            + f"  # ring layouts of radius {_study.RADIUS_RANGE[0]:.4g}-{max_radius:.4g}; "
            f"closest starting pair\n"
            + f"  # {closest:.4f} (d_safe={derived['d_safe']}). Per-robot, so this file only "
            f"fits {NUM_ROBOTS} robots.\n"
            + layout_summary + "\n"
            + _study.format_rollouts("initial_states", initial_states)
            + _study.format_rollouts("goal_states", goal_states)
        )
        out_path.write_text(text)
        print(f"wrote {out_path.relative_to(PROJECT_ROOT)}  "
              f"(prediction_horizon {prediction_horizon})")


def _study_placeable(validated: dict) -> float:
    """generate_fleet_configs.assert_fleet_is_placeable, reused rather than copied."""
    _fleet_spec = importlib.util.spec_from_file_location(
        "_generate_fleet_configs", Path(__file__).resolve().parent / "generate_fleet_configs.py"
    )
    fleet = importlib.util.module_from_spec(_fleet_spec)
    _fleet_spec.loader.exec_module(fleet)
    return fleet.assert_fleet_is_placeable(validated, NUM_ROBOTS)


if __name__ == "__main__":
    main()
