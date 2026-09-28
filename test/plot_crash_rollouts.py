"""One crash rung, four agents: trajectories and the acceleration each one commands.

Companion to test/evaluate_crash.py, and deliberately not part of it. The evaluator runs
50 action-noised episodes per cell because it is estimating *rates*; a figure needs one
reproducible episode. Averaging trajectories would be worse than useless here -- two
robots that correctly swerve opposite ways average into driving straight through each
other -- and recording 50 x 9 x 3 rollouts to keep one in fifty would bloat the evaluation
for nothing. So this rolls out on its own, with noise off and a fixed torch seed, reusing
evaluate_crash's checkpoint loading and the same stepping helpers evaluate_fleet calls, so
the episode drawn here is the episode that script would have scored.

Two figures per rung:

* **trajectories** - the expert and all three policy heads overlaid on one scenario, with
  d_collision footprints at the closest approach, so "did it swerve, and how late" is
  visible rather than inferred from a success rate.
* **actions** - commanded acceleration against time, one panel per action dimension
  (linear and angular are different units and must not share an axis), with the robot's
  limits drawn and the saturated band shaded. This answers how hard a policy is driving
  into its bounds: a head that solves the rung by pinning +-max_linear_accel and
  chattering is not the same result as one that solves it smoothly, and the success
  column cannot tell them apart.

The expert is drawn as a neutral reference rather than a fourth coloured series: the
categorical palette was validated as a set of three, and adding a hue would invalidate
that without re-running the check. It is a ceiling, not a peer.

Usage:
    python test/plot_crash_rollouts.py --models outputs/small_big/models \
        --config test/config/study/crash/unicycle2_crash_v0600.yaml
    python test/plot_crash_rollouts.py --models outputs/small_big/models --all-rungs
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test"))

from core.config import load_and_validate_system_config  # noqa: E402
from core.factory import DynamicsFactory, PlannerFactory  # noqa: E402
from evaluate_crash import crash_step_budget, initial_speed  # noqa: E402
from evaluate_scaling import build_policy, config_start_state, read_checkpoint  # noqa: E402
from learning.dagger import ObservationHistoryBuffer, build_decentralized_joint_action  # noqa: E402
from plot_crash_results import GRID, INK, INK_MUTED, SERIES_STYLE  # noqa: E402

CRITICAL = "#c0362c"
EXPERT_STYLE = (INK_MUTED, "D", (0, (5, 2)), "Expert (MPC)")
# Fraction of a limit above which a command counts as saturated, for the summary box.
SATURATION = 0.95


def roll_out_policy(simulator, policy, device, start, steps, observation_horizon):
    """States and commanded actions for one deterministic policy episode.

    Mirrors evaluate_scaling.evaluate_fleet's inner loop -- same observation, same
    decentralized joint action, same termination test -- but keeps the action it
    commanded, which the evaluator discards. Execution noise is deliberately absent: the
    figure should show what the policy decided, not what the simulator perturbed.
    """
    torch.manual_seed(0)
    if hasattr(policy, "reset"):
        policy.reset()
    state = simulator.reset(np.asarray(start, dtype=float).copy())
    history = (
        ObservationHistoryBuffer(observation_horizon, int(simulator.num_robots))
        if observation_horizon > 1 else None
    )
    states, actions = [np.asarray(state, dtype=float).copy()], []
    for _ in range(steps):
        observation = simulator.observe(state, validate=False)
        action = build_decentralized_joint_action(
            simulator, policy, observation, device,
            observation_horizon=observation_horizon, history_buffer=history,
        )
        actions.append(np.asarray(action, dtype=float).copy())
        state = simulator.step(state, action, validate=False)
        states.append(np.asarray(state, dtype=float).copy())
        if simulator.should_terminate_rollout(state):
            break
    return np.stack(states), np.stack(actions)


def roll_out_expert(simulator, planner, start, steps):
    from planning.casadi_planner import PlannerSolveError

    planner.reset()
    state = simulator.reset(np.asarray(start, dtype=float).copy())
    states, actions = [np.asarray(state, dtype=float).copy()], []
    for _ in range(steps):
        try:
            action = planner(simulator.observe(state, validate=False))
        except PlannerSolveError:
            break
        actions.append(np.asarray(action, dtype=float).copy())
        state = simulator.step(state, action, validate=False)
        states.append(np.asarray(state, dtype=float).copy())
        if simulator.should_terminate_rollout(state):
            break
    return np.stack(states), np.stack(actions)


def style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot_trajectories(ax, rollouts, simulator, d_collision, v0) -> None:
    slices = simulator.robot_state_slices
    for name, (states, _actions) in rollouts.items():
        color, _marker, dash, label = style_for(name)
        for robot_index, state_slice in enumerate(slices):
            xs = states[:, state_slice.start]
            ys = states[:, state_slice.start + 1]
            ax.plot(xs, ys, color=color, linestyle=dash, linewidth=2.0,
                    label=label if robot_index == 0 else None, zorder=3)
            ax.scatter(xs[0], ys[0], color=color, s=26, zorder=4)

        # Footprints at the closest approach: where the scenario was actually decided.
        positions = np.stack([
            np.stack([states[:, s.start], states[:, s.start + 1]], axis=1) for s in slices
        ])
        gaps = np.linalg.norm(positions[0] - positions[1], axis=1)
        tightest = int(np.argmin(gaps))
        for robot_index in range(len(slices)):
            ax.add_patch(plt.Circle(
                positions[robot_index, tightest], d_collision / 2.0,
                color=color, alpha=0.18, linewidth=0, zorder=2,
            ))

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Trajectories at $v_0$ = {v0:g} m/s\n"
        f"shaded discs: d_collision footprint at each run's closest approach",
        fontsize=11, color=INK, loc="left",
    )
    style_axes(ax)


def style_for(name: str):
    """Colour and marker from the shared palette, but solid lines for every policy.

    Only the expert is dashed, which is what distinguishes the reference from the three
    things being compared. Note this drops line style as a secondary encoding: the
    palette's tightest pair (Flow vs SafeFlow) separates by dE 9.9 under deuteranopia,
    which clears the >= 8 target but no longer has a dash pattern backing it up, so these
    figures should not be printed in greyscale.
    """
    if name == "expert":
        return EXPERT_STYLE
    color, marker, _dash, label = SERIES_STYLE[name]
    return color, marker, "-", label


def plot_actions(axes, rollouts, simulator, dt, v0) -> None:
    robot = simulator.simulators[0]
    limits = [float(robot.max_action[0]), float(robot.max_action[1])]
    titles = [
        ("linear acceleration", "$a_v$ (m/s$^2$)", "max_linear_accel"),
        ("angular acceleration", r"$a_\omega$ (rad/s$^2$)", "max_angular_accel"),
    ]
    action_slices = simulator.robot_action_slices if hasattr(
        simulator, "robot_action_slices") else None

    summary: dict[str, list[tuple[float, float]]] = {}
    for dimension, (ax, (title, ylabel, limit_name)) in enumerate(zip(axes, titles)):
        limit = limits[dimension]
        extremes: list[float] = []
        ax.axhline(limit, color=CRITICAL, linewidth=1.3, zorder=1)
        ax.axhline(-limit, color=CRITICAL, linewidth=1.3, zorder=1)

        for name, (_states, actions) in rollouts.items():
            color, _marker, dash, label = style_for(name)
            # Robot 0 only: the layout is exactly head-on, so robot 1 mirrors it and
            # plotting both doubles the ink without adding information.
            column = dimension if action_slices is None else action_slices[0].start + dimension
            series = actions[:, column]
            times = np.arange(len(series)) * dt
            ax.plot(times, series, color=color, linestyle=dash, linewidth=1.8,
                    label=label, zorder=3)
            saturated = float(np.mean(np.abs(series) >= SATURATION * limit))
            # Commands the robot cannot execute. unicycle2.step clips to +-max_action
            # (systems/unicycle2.py:80), so anything past the line is an action the
            # policy asked for and silently did not get -- worth separating from
            # "pushed hard", which is what saturation alone measures.
            # 0.1% tolerance, not a strict >. The expert solves its bound as an explicit
            # constraint, so IPOPT returns values a few ulps past it (8.0000001 > 8);
            # a strict test reports that as the expert demanding the impossible, which
            # is a libel on the one agent here that genuinely respects its limits.
            over = float(np.mean(np.abs(series) > limit * 1.001))
            summary.setdefault(name, []).append((saturated, over))
            extremes.append(float(np.max(np.abs(series))))

        # Fit the commanded values, not just the limits: cropping at the bound would hide
        # exactly the overshoot this panel exists to show.
        span = max(extremes + [limit]) * 1.08
        ax.set_ylim(-span, span)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Commanded {title} - robot 0, {limit_name} = {limit:g}\n"
            f"what the policy asked for; the simulator clips to this bound",
            fontsize=11, color=INK, loc="left",
        )
        style_axes(ax)

    return summary


def summary_text(summary) -> str:
    """One table: how hard each agent pushed, and how often it asked for the impossible."""
    header = (
        f"{'':16}{'linear a_v':>24}{'angular a_omega':>26}\n"
        f"{'':16}{'at limit':>12}{'over limit':>12}{'at limit':>13}{'over limit':>13}"
    )
    lines = [header]
    for name, dims in summary.items():
        _c, _m, _d, label = style_for(name)
        (lin_sat, lin_over), (ang_sat, ang_over) = dims
        lines.append(
            f"{label:<16}{lin_sat:>11.1%}{lin_over:>12.1%}{ang_sat:>13.1%}{ang_over:>13.1%}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", type=Path, required=True,
                        help="models directory, e.g. outputs/small_big/models")
    parser.add_argument("--config", type=Path, default=None, help="one crash rung")
    parser.add_argument("--all-rungs", action="store_true",
                        help="every config in test/config/study/crash/")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <models>/../plots/crash")
    parser.add_argument("--no-expert", action="store_true",
                        help="skip the CasADi reference rollout")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.all_rungs:
        configs = sorted((PROJECT_ROOT / "test/config/study/crash").glob("unicycle2_crash_v*.yaml"))
    elif args.config:
        configs = [args.config]
    else:
        raise SystemExit("pass --config or --all-rungs")

    runs = []
    for run_dir in sorted(p for p in args.models.iterdir() if p.is_dir()):
        found = sorted(run_dir.glob("*_dagger_checkpoint.pt"))
        if found:
            runs.append(found[0])
    if not runs:
        raise SystemExit(f"no checkpoints under {args.models}")

    output_dir = args.output_dir or (args.models.parent / "plots" / "crash")
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_path in configs:
        raw = yaml.safe_load(Path(config_path).read_text())
        v0 = initial_speed(raw)
        steps = crash_step_budget(raw)
        validated = load_and_validate_system_config("multi_robot", config_path)
        simulator = DynamicsFactory.create(system_name="multi_robot", config=validated)
        start = config_start_state(validated)
        dt = float(raw["robots"][0]["config"]["dt"])
        d_collision = float(raw["d_collision"])

        rollouts = {}
        if not args.no_expert:
            planner = PlannerFactory.create("casadi", simulator=simulator, config=validated)
            rollouts["expert"] = roll_out_expert(simulator, planner, start, steps)
            print(f"v0={v0:g}  expert: {len(rollouts['expert'][1])} steps", flush=True)

        for checkpoint_path in runs:
            checkpoint = read_checkpoint(str(checkpoint_path), device)
            policy_type = str(checkpoint.get("policy_type", "mlp")).lower()
            policy = build_policy(
                checkpoint, device,
                simulator=simulator if policy_type == "safeflow" else None,
                planner_config=validated if policy_type == "safeflow" else None,
            )
            rollouts[policy_type] = roll_out_policy(
                simulator, policy, device, start, steps,
                int(checkpoint.get("observation_horizon", 1)),
            )
            print(f"v0={v0:g}  {policy_type}: {len(rollouts[policy_type][1])} steps", flush=True)

        # Ordered so the expert draws first and the policies read on top of it.
        ordered = {name: rollouts[name] for name in ("expert", "mlp", "flow", "safeflow")
                   if name in rollouts}

        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
        plot_trajectories(ax, ordered, simulator, d_collision, v0)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")
        fig.tight_layout()
        stem = Path(config_path).stem
        for extension in ("pdf", "png"):
            fig.savefig(output_dir / f"{stem}_trajectories.{extension}", dpi=200,
                        facecolor=fig.get_facecolor())
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
        fig.patch.set_facecolor("#fcfcfb")
        for ax in axes:
            ax.set_facecolor("#fcfcfb")
        summary = plot_actions(axes, ordered, simulator, dt, v0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False,
                   fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 0.175))
        fig.suptitle(f"Commanded acceleration at $v_0$ = {v0:g} m/s", fontsize=12, color=INK)
        fig.text(0.5, 0.145, summary_text(summary), ha="center", va="top", fontsize=8.5,
                 family="monospace", color=INK,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#fcfcfb", edgecolor=GRID))
        fig.tight_layout(rect=(0, 0.24, 1, 0.97))
        for extension in ("pdf", "png"):
            fig.savefig(output_dir / f"{stem}_actions.{extension}", dpi=200,
                        bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  wrote {output_dir / stem}_{{trajectories,actions}}.{{pdf,png}}\n", flush=True)


if __name__ == "__main__":
    main()
