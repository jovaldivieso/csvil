"""Plot the study 1 grid: success rate against evaluation fleet size.

One panel per scenario, one line per cell of the {mlp, flow} x {h=1, h=8} grid,
shaded to the min-max across training seeds.

The 2x2 is encoded on two channels rather than four arbitrary hues, so the factorial
structure is readable directly off the chart: **colour is the policy head** and
**dash is the action horizon**. That also keeps identity off colour alone.

The convergence tolerance is a task definition rather than a result, so it is fixed
for the whole figure and stated in the subtitle instead of becoming an axis.

Usage:
    python test/plot_study1_results.py
    python test/plot_study1_results.py --results-dir outputs/study1/eval --output-dir outputs/study1/plots
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

# Shared with test/plot_study_results.py so the two studies' figures sit together.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8983"
GRID = "#e3e2de"

# Colour = policy head, dash = horizon. Both hues are the encoder study's, and the
# pair passes the categorical checks against this surface (CVD dE 24.7 protan,
# 33.6 normal, contrast >= 3:1).
HEAD_COLORS = {"mlp": "#2a78d6", "flow": "#eb6834"}
# 8 is the first study-1 run's chunk, 10 the retrain's; both stay so old and new
# result files plot side by side without one being drawn as the other.
HORIZON_DASH = {1: (None, None), 8: (5, 2), 10: (2, 2)}

# Mirrors VARIANTS in learning/config/study/generate_study_policy_configs.py.
VARIANTS = {
    "mlp": ("mlp", 1),
    "flow_h1": ("flow", 1),
    "mlp_h8": ("mlp", 8),
    "flow": ("flow", 8),
    # Retrain (STUDY1_VARIANTS): chunked cells at horizon 10.
    "mlp_h10": ("mlp", 10),
    "flow_h10": ("flow", 10),
}
SCENARIO_TITLES = {"random": "Random goals", "circle": "Antipodal ring"}


def variant_of(cell: str) -> str | None:
    """'deepset_mlp_h8_n04' -> 'mlp_h8'. None for anything not in the grid."""
    parts = cell.split("_")
    if len(parts) < 3:
        return None
    variant = "_".join(parts[1:-1])
    return variant if variant in VARIANTS else None


def load_scenario(results_dir: Path, scenario: str) -> list[dict[str, str]]:
    path = results_dir / scenario / "study1.csv"
    return list(csv.DictReader(path.open())) if path.exists() else []


def series_from(rows) -> dict[str, dict[int, list[float]]]:
    """{variant: {fleet_size: [success_rate per seed]}}"""
    series: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        variant = variant_of(row["cell"])
        if variant is None:
            continue
        fleet = int(row["eval_fleet_size"])
        series.setdefault(variant, {}).setdefault(fleet, []).append(float(row["success_rate"]))
    return series


def tolerance_note(rows) -> str:
    if not rows:
        return ""
    row = rows[0]
    return (
        f"success = every robot within pos {row['pos_tol']} m, heading {row['theta_tol']} rad, "
        f"speed {row['vel_tol']} m/s, yaw rate {row['omega_tol']} rad/s, with no collision"
    )


def draw_panel(axis, series, fleets, title: str, show_y_label: bool) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(title, fontsize=11, color=TEXT_PRIMARY, pad=10, loc="left")
    axis.set_xlabel("Evaluation fleet size", fontsize=9, color=TEXT_SECONDARY)
    if show_y_label:
        axis.set_ylabel("Success rate", fontsize=9, color=TEXT_SECONDARY)

    # Fleet sizes are unevenly spaced (2, 4, 8, 16, 32); plotting them at their own
    # index keeps the gaps uniform without a log axis to misread.
    positions = {fleet: index for index, fleet in enumerate(fleets)}
    axis.set_xticks(list(positions.values()))
    axis.set_xticklabels([str(f) for f in fleets], fontsize=9, color=TEXT_SECONDARY)
    axis.set_ylim(-0.04, 1.04)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.tick_params(axis="y", labelsize=9, colors=TEXT_SECONDARY, length=0)
    axis.tick_params(axis="x", length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)

    for variant, (head, horizon) in VARIANTS.items():
        by_fleet = series.get(variant)
        if not by_fleet:
            continue
        present = [f for f in fleets if f in by_fleet]
        x = [positions[f] for f in present]
        means = [statistics.mean(by_fleet[f]) for f in present]
        lows = [min(by_fleet[f]) for f in present]
        highs = [max(by_fleet[f]) for f in present]
        color = HEAD_COLORS[head]
        dashes = HORIZON_DASH[horizon]

        # Seed spread as a band rather than error bars: with three seeds the range is
        # the honest statement, and bars would clutter four overlapping series.
        if any(high > low for low, high in zip(lows, highs)):
            axis.fill_between(x, lows, highs, color=color, alpha=0.10, linewidth=0)
            for edge in (lows, highs):
                axis.plot(x, edge, color=color, linewidth=0.7, alpha=0.35, zorder=2)
        line, = axis.plot(x, means, color=color, linewidth=2.0, marker="o", markersize=6,
                          markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3)
        if dashes[0] is not None:
            line.set_dashes(list(dashes))


def legend_handles() -> list[Line2D]:
    handles = []
    for variant, (head, horizon) in VARIANTS.items():
        dashes = HORIZON_DASH[horizon]
        # No marker on the legend handle: the marker's surface-coloured edge breaks a
        # short solid line into two segments, so it reads as dashed and the horizon
        # channel becomes unreadable. Line style alone carries it here.
        handle = Line2D([], [], color=HEAD_COLORS[head], linewidth=2.0,
                        label=f"{head}  h={horizon}")
        if dashes[0] is not None:
            handle.set_dashes(list(dashes))
        handles.append(handle)
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "outputs/study1/eval")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/study1/plots")
    parser.add_argument("--scenarios", nargs="+", default=["random", "circle"])
    args = parser.parse_args()

    panels = [(s, load_scenario(args.results_dir, s)) for s in args.scenarios]
    panels = [(s, rows) for s, rows in panels if rows]
    if not panels:
        raise SystemExit(f"no study1.csv under {args.results_dir}; run ./eval_study1.sh first")

    figure, axes = plt.subplots(
        1, len(panels), figsize=(5.4 * len(panels), 4.2), facecolor=SURFACE, squeeze=False,
    )
    for index, (scenario, rows) in enumerate(panels):
        series = series_from(rows)
        fleets = sorted({int(r["eval_fleet_size"]) for r in rows})
        seeds = len({r["seed"] for r in rows})
        title = f"{SCENARIO_TITLES.get(scenario, scenario)}  ·  {seeds} seed{'s' if seeds != 1 else ''}"
        draw_panel(axes[0][index], series, fleets, title, show_y_label=index == 0)

    figure.suptitle("Study 1 — policy head and action horizon across fleet sizes",
                    fontsize=13, color=TEXT_PRIMARY, x=0.055, ha="left", y=1.005)
    figure.text(0.055, 0.945, tolerance_note(panels[0][1]), fontsize=8.5, color=TEXT_MUTED, ha="left")
    figure.text(0.055, 0.035, "Band spans the min-max across training seeds. Colour is the policy "
                              "head, dash is the action horizon.",
                fontsize=8.5, color=TEXT_MUTED, ha="left")
    # handlelength has to be long enough that a dash pattern is legible at all.
    figure.legend(handles=legend_handles(), loc="upper right", bbox_to_anchor=(0.985, 1.01),
                  frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, ncol=2,
                  columnspacing=1.6, handlelength=3.0, handletextpad=0.7)
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = args.output_dir / f"study1_success_by_fleet.{suffix}"
        figure.savefig(path, bbox_inches="tight", facecolor=SURFACE, dpi=200)
        print(f"wrote {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
