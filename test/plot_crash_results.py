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
  whether the thing is deployable.

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
SERIES_STYLE = {
    "mlp": ("#2a78d6", "o", "-", "MLP"),
    "flow": ("#eb6834", "s", "--", "Flow"),
    "safeflow": ("#1baf7a", "^", "-.", "SafeFlow"),
}
SERIES_ORDER = ("mlp", "flow", "safeflow")

# Ink, never series colour, for text (see the palette's text tokens).
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
# Status colour, reserved: this marks a physical contact, not a fourth series. Ships with
# a label so it never signals by colour alone.
CRITICAL = "#c0362c"


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


def group(rows: list[dict]) -> dict[str, list[dict]]:
    """Rows by policy type, each sorted along the ladder."""
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        record = dict(row)
        record["v0"] = initial_speed(row["config"])
        by_policy[str(row["policy_type"]).lower()].append(record)
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


def plot_clearance(
    ax, by_policy: dict[str, list[dict]], d_collision: float, d_safe: float
) -> None:
    ax.axhspan(0.0, d_collision, color=CRITICAL, alpha=0.10, zorder=0)
    ax.axhline(d_collision, color=CRITICAL, linewidth=1.4, linestyle="-", zorder=1)
    ax.axhline(d_safe, color=INK_MUTED, linewidth=1.2, linestyle=":", zorder=1)

    has_worst = False
    for name in SERIES_ORDER:
        series = by_policy.get(name)
        if not series:
            continue
        color, marker, dash, label = SERIES_STYLE[name]
        xs = [record["v0"] for record in series]
        means = [float(record["mean_min_pair_distance"]) for record in series]
        worst_raw = [record.get("min_min_pair_distance") for record in series]
        if all(value not in (None, "") for value in worst_raw):
            has_worst = True
            worst = [float(value) for value in worst_raw]
            ax.fill_between(xs, worst, means, color=color, alpha=0.16, linewidth=0, zorder=2)
            ax.plot(xs, worst, color=color, marker=marker, linestyle=dash, linewidth=2.0,
                    markersize=8, label=label, zorder=3)
            ax.plot(xs, means, color=color, linestyle=dash, linewidth=1.0, alpha=0.55, zorder=2)
            endpoint = worst[-1]
        else:
            # Older CSVs predate the worst-episode column; the mean is all there is.
            ax.plot(xs, means, color=color, marker=marker, linestyle=dash, linewidth=2.0,
                    markersize=8, label=label, zorder=3)
            endpoint = means[-1]
        ax.annotate(label, (xs[-1], endpoint), textcoords="offset points", xytext=(9, 0),
                    va="center", fontsize=9, color=INK)

    # Bound to the data rather than to zero: anchoring at 0 turned the collision band
    # into two thirds of the panel and squeezed every series into a thin strip, which is
    # the opposite of what the panel is for.
    finite = [v for v in ax.get_lines()[2:] for v in v.get_ydata()]
    low = min(finite + [d_collision]) if finite else d_collision
    high = max(finite + [d_safe]) if finite else d_safe
    pad = max(0.06 * (high - low), 0.004)
    ax.set_ylim(low - pad, high + pad)
    ax.set_ylabel("closest approach between robots (m)")
    subtitle = "worst episode (solid), band up to the mean" if has_worst else "mean over episodes"
    ax.set_title(
        f"Clearance - {subtitle}\nexpert holds d_safe on every rung",
        fontsize=11, color=INK, loc="left",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.results)
    by_policy = group(rows)
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
        ("crash_clearance", lambda ax: plot_clearance(ax, by_policy, d_collision, d_safe)),
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
