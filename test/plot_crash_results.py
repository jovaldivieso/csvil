"""Plot the head-on crash ladder: success and clearance against initial speed.

The crash scenario fixes everything the encoder study varied -- one fleet size, one
encoder, one layout -- and sweeps a single quantity, the speed the robots are already
carrying when the episode starts. So the matrix figures in plot_study_results.py (training
fleet size across, evaluation fleet size down) have nothing to show here: every cell would
be N=2. What this draws instead is the difficulty curve, one line per policy head.

Two panels, because success and clearance answer different questions and must not share
an axis:

* **Success rate** -- did the fleet get both robots to their goals without colliding.
  95% Wilson intervals, the same interval plot_study_results.py uses, because at 50
  episodes a 0.7 and a 0.8 are not distinguishable and the figure should say so.
* **Clearance** -- how close the two robots actually came. Plotted as the *worst*
  episode, with a band up to the mean. A mean alone is the wrong summary for a safety
  claim: a policy averaging 0.15 m while one episode grazed 0.101 m is not the same
  policy as one that never went below 0.14 m, and it is the near-miss that decides
  whether the thing is deployable. Rungs where the safety filter's QP went infeasible are
  dropped rather than drawn: those episodes abort at step 1 and their logged distance is
  the start separation, so plotting them shows SafeFlow's clearance *rising* on the rungs
  where it has actually given up. See CLEARANCE_MAX and --clearance-max.

Reference lines come from the scenario config, not from constants here: d_collision (a
contact) and d_safe (the expert's planning buffer). The CasADi expert holds exactly d_safe
on all five rungs, so that line doubles as the achievable ceiling.

Usage:
    python test/plot_crash_results.py --results outputs/small_med/eval/crash.csv
    python test/plot_crash_results.py --results ... --with-failures
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Categorical slots 1-3 of the validated default palette, in fixed order. Checked for
# adjacent-pair separation in OKLab: worst normal-vision dE 24.0 (hard floor 15), worst
# CVD dE 9.9 (target >= 8, deuteranope flow/safeflow). That 9.9 clears the target but not
# by much, which is why every series also carries its own dash pattern and marker --
# identity never rests on hue alone, and the figure survives greyscale printing.
#
# Two comparisons reuse the same three slots, because only one of them is ever on a
# figure: the small-robot experiment varies the policy head at a fixed encoder, and the
# 1 m experiments (data_mid_best and friends) vary the encoder at a fixed head. Each is
# three series in a fixed order, so the validated palette applies unchanged -- what must
# never happen is both axes at once, which would want nine hues and the skill's rule is
# that a ninth series is never a generated colour.
HEAD_STYLE = {
    "mlp": ("#2a78d6", "o", "-", "MLP"),
    "flow": ("#eb6834", "s", "--", "Flow"),
    "safeflow": ("#1baf7a", "^", "-.", "SafeFlow"),
}
ENCODER_STYLE = {
    "deepset": ("#2a78d6", "o", "-", "DeepSet"),
    "transformer": ("#eb6834", "s", "--", "Transformer"),
    "gnn": ("#1baf7a", "^", "-.", "GNN"),
}
GROUPINGS = {"policy_type": HEAD_STYLE, "encoder_type": ENCODER_STYLE}

# Rebound by main() once --group-by is known; the module-level default keeps
# plot_crash_rollouts.py's import working for the policy-head comparison.
SERIES_STYLE = HEAD_STYLE
SERIES_ORDER = tuple(HEAD_STYLE)


def use_grouping(column: str) -> None:
    """Point the module's series style/order at whichever axis is being compared."""
    global SERIES_STYLE, SERIES_ORDER
    SERIES_STYLE = GROUPINGS[column]
    SERIES_ORDER = tuple(SERIES_STYLE)

# Ink, never series colour, for text (see the palette's text tokens).
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
# Status colour, reserved: this marks a physical contact, not a fourth series. Ships with
# a label so it never signals by colour alone.
CRITICAL = "#c0362c"

# Above this, a "closest approach" is not a measurement. Once the safety filter's QP goes
# infeasible the episode aborts at step 1, and evaluate_crash.py still logs a min pair
# distance -- the robots' *start* separation, which in this scenario is ~0.33 m, roughly
# three times d_safe. Plotted, it becomes a line that climbs as the scenario gets harder,
# which reads as the policy keeping more room exactly where it has in fact stopped
# steering. Anything this far above d_safe is one of those aborts (the head-on configs
# close the gap within a few steps), so the rung is dropped rather than drawn.
CLEARANCE_MAX = 0.2


def save_figure(fig, stem: Path, titled=("pdf",), untitled=("png",), **savefig_kwargs) -> None:
    """Write the figure once with its titles and once without.

    The PDF keeps every title, so the file is self-describing when opened on its own.
    The PNG drops them, because that is the one that goes into a paper or a slide where
    a caption already says what the figure is and a baked-in title collides with it.
    Axis labels, legends and annotations stay in both -- those are part of reading the
    data, not a heading.
    """
    for extension in titled:
        fig.savefig(f"{stem}.{extension}", **savefig_kwargs)
    if not untitled:
        return
    suptitle = fig._suptitle.get_text() if fig._suptitle is not None else None
    axis_titles = [ax.get_title() for ax in fig.axes]
    if suptitle is not None:
        fig.suptitle("")
    for ax in fig.axes:
        ax.set_title("")
    for extension in untitled:
        fig.savefig(f"{stem}.{extension}", **savefig_kwargs)
    # Restore, so a caller that reuses the figure is not silently handed a bare one.
    if suptitle is not None:
        fig.suptitle(suptitle)
    for ax, title in zip(fig.axes, axis_titles):
        ax.set_title(title, loc="left")


def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - stays inside [0, 1] where the normal approximation does not.

    Lifted from plot_study_results.py deliberately: the two figures report the same
    quantity and must not disagree about its uncertainty.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def initial_speed(config_path: str) -> float:
    """The rung's initial speed, read from the config's own start state.

    Taken from the file rather than parsed out of its name: the name is a rounded
    convenience (v1375) and a renamed config would silently plot at the wrong x.
    """
    raw = yaml.safe_load((PROJECT_ROOT / config_path).read_text())
    robot = raw["robots"][0]
    start = robot.get("start") or robot["config"]["start"]
    return float(start[3])  # unicycle2 state = [x, y, theta, v, omega]


def scenario_thresholds(config_path: str) -> tuple[float, float]:
    raw = yaml.safe_load((PROJECT_ROOT / config_path).read_text())
    return float(raw["d_collision"]), float(raw["d_safe"])


def read_rows(results_csv: Path) -> list[dict]:
    with results_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{results_csv} has no data rows.")
    return rows


def group(rows: list[dict], column: str = "policy_type") -> dict[str, list[dict]]:
    """Rows by the compared axis, each sorted along the ladder.

    ``column`` is "policy_type" for the head comparison and "encoder_type" for the
    encoder one. Grouping by the wrong column does not error -- it silently stacks every
    run onto one series with several conflicting values per rung -- so this checks that
    the column actually separates the rows and says so if it does not.
    """
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        record = dict(row)
        record["v0"] = float(row["v0"]) if row.get("v0") else initial_speed(row["config"])
        by_policy[str(row[column]).lower()].append(record)
    rungs = len({row["config"] for row in rows})
    for name, series in by_policy.items():
        if len(series) > rungs:
            raise SystemExit(
                f"'{column}' does not separate these runs: '{name}' has {len(series)} rows "
                f"for {rungs} rungs, so several checkpoints would be drawn as one line. "
                f"Pick a different --group-by, or filter the runs "
                f"(e.g. evaluate only the fleet_02 checkpoints)."
            )
    for series in by_policy.values():
        series.sort(key=lambda record: record["v0"])
    return by_policy


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


def label_rungs(ax, speeds: list[float], v_max: float) -> None:
    """Tick only at the rungs, each labelled in m/s and as a share of the robot's limit.

    A twinned top axis was tried first and collided with the panel titles; the ladder has
    five rungs, so ticking the data itself is both cleaner and more honest than an evenly
    spaced grid that implies speeds nothing was measured at.
    """
    ax.set_xticks(speeds)
    # ax.set_xticklabels([f"{v:g}\n({100 * v / v_max:.0f}%)" for v in speeds])
    # Name what a tick *is*. Every rung is a separate config that differs from its
    # neighbours in exactly one number, so a reader should not have to infer that the
    # x-axis and the config list are the same thing.
    ax.set_xlabel("initial speed $v_0$")


def plot_success(ax, by_policy: dict[str, list[dict]]) -> None:
    for name in SERIES_ORDER:
        series = by_policy.get(name)
        if not series:
            continue
        color, marker, dash, label = SERIES_STYLE[name]
        xs = [record["v0"] for record in series]
        ys = [float(record["success_rate"]) for record in series]
        episodes = [int(record["episodes"]) for record in series]
        lows, highs = zip(*[
            wilson_interval(rate * n, n) for rate, n in zip(ys, episodes)
        ])
        ax.errorbar(
            xs, ys,
            yerr=[[y - lo for y, lo in zip(ys, lows)], [hi - y for y, hi in zip(ys, highs)]],
            color=color, marker=marker, linestyle=dash, linewidth=2.0, markersize=8,
            capsize=3, elinewidth=1.2, label=label, zorder=3,
        )
        # Direct label in ink; the coloured marker it sits beside carries identity.
        ax.annotate(
            label, (xs[-1], ys[-1]), textcoords="offset points", xytext=(9, 0),
            va="center", fontsize=9, color=INK,
        )
    ax.set_ylim(-0.04, 1.08)
    ax.set_ylabel("success rate")
    ax.set_title("Both robots reach their goals, no contact", fontsize=11, color=INK, loc="left")
    style_axes(ax)


def measured(value: float, ceiling: float) -> float:
    """``value``, or NaN if it is too large to be a real closest approach.

    NaN rather than a dropped element so the x positions stay aligned: matplotlib breaks
    a line and skips a marker at NaN, so the rung leaves a visible gap instead of a
    segment interpolated straight across it.
    """
    return value if value <= ceiling else float("nan")


def plot_clearance(
    ax, by_policy: dict[str, list[dict]], d_collision: float, d_safe: float,
    clearance_max: float = CLEARANCE_MAX,
) -> None:
    ax.axhspan(0.0, d_collision, color=CRITICAL, alpha=0.10, zorder=0)
    ax.axhline(d_collision, color=CRITICAL, linewidth=1.4, linestyle="-", zorder=1)
    ax.axhline(d_safe, color=INK_MUTED, linewidth=1.2, linestyle=":", zorder=1)

    has_worst = False
    dropped: list[str] = []
    for name in SERIES_ORDER:
        series = by_policy.get(name)
        if not series:
            continue
        color, marker, dash, label = SERIES_STYLE[name]
        xs = [record["v0"] for record in series]
        means = [measured(float(record["mean_min_pair_distance"]), clearance_max)
                 for record in series]
        worst_raw = [record.get("min_min_pair_distance") for record in series]
        if all(value not in (None, "") for value in worst_raw):
            has_worst = True
            worst = [measured(float(value), clearance_max) for value in worst_raw]
            # Losing the worst episode means no episode on that rung ran long enough to
            # measure, so its mean cannot be a clearance either. The reverse is allowed:
            # a rung where only some episodes aborted keeps its (real) worst marker and
            # simply draws no band, since the aborts inflate the mean above it.
            means = [float("nan") if math.isnan(w) else m for w, m in zip(worst, means)]
            ax.fill_between(xs, worst, means, color=color, alpha=0.16, linewidth=0, zorder=2)
            ax.plot(xs, worst, color=color, marker=marker, linestyle=dash, linewidth=2.0,
                    markersize=8, label=label, zorder=3)
            ax.plot(xs, means, color=color, linestyle=dash, linewidth=1.0, alpha=0.55, zorder=2)
            labelled = worst
        else:
            # Older CSVs predate the worst-episode column; the mean is all there is.
            ax.plot(xs, means, color=color, marker=marker, linestyle=dash, linewidth=2.0,
                    markersize=8, label=label, zorder=3)
            labelled = means
        # No direct end-of-line label: the three series converge on d_safe over most of
        # the ladder, so the labels landed on top of each other and on the d_safe line.
        # Identity is carried by the legend plus each series' own dash pattern and marker.
        gaps = [f"{x:g}" for x, y in zip(xs, labelled) if math.isnan(y)]
        if gaps:
            dropped.append(f"{label} at v0 {', '.join(gaps)}")
    if dropped:
        print(f"clearance: dropped rungs above {clearance_max:g} m "
              f"(aborted episodes, distance is the start separation): "
              f"{'; '.join(dropped)}")

    # Bound to the data rather than to zero: anchoring at 0 turned the collision band
    # into two thirds of the panel and squeezed every series into a thin strip, which is
    # the opposite of what the panel is for.
    finite = [value for line in ax.get_lines()[2:] for value in line.get_ydata()
              if math.isfinite(value)]
    low = min(finite + [d_collision]) if finite else d_collision
    high = max(finite + [d_safe]) if finite else d_safe
    pad = max(0.06 * (high - low), 0.004)
    ax.set_ylim(low - pad, high + pad)
    ax.set_ylabel("min robot distance (m)")
    subtitle = "worst episode (solid), band up to the mean" if has_worst else "mean over episodes"
    # ax.set_title(
    #     f"Clearance - {subtitle}\nexpert holds d_safe on every rung",
    #     fontsize=11, color=INK, loc="left",
    # )
    ax.annotate(
        f"contact (d_collision {d_collision:g} m)", (0.015, d_collision),
        xycoords=("axes fraction", "data"), xytext=(0, -11), textcoords="offset points",
        fontsize=8, color=CRITICAL,
    )
    ax.annotate(
        f"d_safe {d_safe:g} m",
        (0.015, d_safe), xycoords=("axes fraction", "data"), xytext=(0, 5),
        textcoords="offset points", fontsize=8, color=INK_MUTED,
    )
    style_axes(ax)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, required=True,
                        help="merged crash CSV, e.g. outputs/<exp>/eval/crash.csv")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <results parent>/../plots/crash")
    parser.add_argument("--group-by", choices=sorted(GROUPINGS), default="policy_type",
                        help="which axis the series compare: 'policy_type' for the "
                             "mlp/flow/safeflow experiment (default), 'encoder_type' for "
                             "an all-MLP grid like data_mid_best")
    parser.add_argument("--clearance-max", type=float, default=CLEARANCE_MAX,
                        help="drop clearance points above this many metres -- they are "
                             "solver aborts logging the start separation, not a measured "
                             f"closest approach (default: {CLEARANCE_MAX:g})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_grouping(args.group_by)
    rows = read_rows(args.results)
    by_policy = group(rows, args.group_by)
    missing = [name for name in SERIES_ORDER if name not in by_policy]
    if missing:
        print(f"note: no rows for {', '.join(missing)} -- plotting the heads that are present")

    first_config = rows[0]["config"]
    d_collision, d_safe = scenario_thresholds(first_config)
    raw = yaml.safe_load((PROJECT_ROOT / first_config).read_text())
    v_max = float(raw["robots"][0]["config"]["max_linear_vel"])

    speeds = sorted({record["v0"] for series in by_policy.values() for record in series})
    output_dir = args.output_dir or (args.results.parent.parent / "plots" / "crash")
    output_dir.mkdir(parents=True, exist_ok=True)

    # One metric per file. They answer different questions and are read at different
    # points in an argument, so a reader who wants the clearance figure should not have
    # to crop the success one out of it.
    for name, draw in (
        ("crash_success", lambda ax: plot_success(ax, by_policy)),
        ("crash_clearance", lambda ax: plot_clearance(ax, by_policy, d_collision, d_safe,
                                                       args.clearance_max)),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        draw(ax)
        label_rungs(ax, speeds, v_max)

        # Legend present for >= 2 series even though every line is also directly
        # labelled, so identity never rests on position alone.
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                       frameon=False, fontsize=9, labelcolor=INK,
                       bbox_to_anchor=(0.5, -0.005))
        fig.tight_layout(rect=(0, 0.06, 1, 1.0))
        save_figure(fig, output_dir / name, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"wrote {output_dir / name}.{{pdf,png}} (png without the title)")
        plt.close(fig)


if __name__ == "__main__":
    main()
