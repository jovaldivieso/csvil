"""Plot encoder-scaling study results produced by test/evaluate_scaling.py.

The study evaluates on two axes, and each one varies exactly one quantity (see
docs/study2_encoders.md): evaluation **fleet size** at the training density, and
evaluation **density** at a fixed fleet size. ``--axis`` picks which one is plotted;
``auto`` reads it off the results, since a density sweep holds several densities per
fleet size and a fleet sweep exactly one.

Three figures per axis:

* ``*_matrix.pdf`` - one panel per encoder, training fleet size across, the axis down,
  one metric per cell. The in-distribution cells are outlined: trained and evaluated on
  the same fleet size, or evaluated at the training density.
* ``*_by_<axis>.pdf`` - the same success rates as lines along the axis with 95% Wilson
  intervals, pooled over training fleet size. The matrix shows the cells; this shows
  whether the differences between them survive the sample.
* ``*_by_<axis>_facets.pdf`` - one panel per training fleet size, so the pooled figure's
  assumption is visible rather than implied. Pooling buys a tighter interval but would
  hide a real training-fleet effect if one existed.

On the density axis every figure is written once per evaluation fleet size, because a
density sweep holds several fleet sizes whose rows would otherwise share a cell.

Usage:
    python test/plot_study_results.py
    python test/plot_study_results.py --metric collision_rate
    python test/plot_study_results.py --results outputs/study2/eval/density.csv --policy flow
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RESULTS = PROJECT_ROOT / "outputs/study2/eval/random/encoder_scaling.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/study2/plots/random"

ENCODER_ORDER = ("deepset", "transformer", "gnn")
ENCODER_LABELS = {"deepset": "DeepSet", "transformer": "Transformer", "gnn": "GNN"}

# Sequential blue ramp (light -> dark) for magnitude; one hue, never a rainbow.
SEQUENTIAL_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
# Categorical slots 1-3, validated for CVD separation against the light surface.
SERIES_COLORS = {"deepset": "#2a78d6", "transformer": "#eb6834", "gnn": "#1baf7a"}

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8983"
GRID = "#e3e2de"

METRIC_LABELS = {
    "success_rate": "Success rate",
    "collision_rate": "Collision rate",
    "timeout_rate": "Timeout rate",
    "mean_steps": "Mean steps",
    "mean_goal_position_error": "Mean goal position error (m)",
    "mean_goal_heading_error": "Mean goal heading error (rad)",
    "mean_min_pair_distance": "Mean min pair distance",
    "mean_action_ms": "Mean policy call (ms/step)",
    "mean_action_ms_per_robot": "Mean policy call (ms/step/robot)",
}
RATE_METRICS = {"success_rate", "collision_rate", "timeout_rate"}


def training_density() -> float:
    """Robots per m^2 every training fleet size is placed at.

    Read off the generator rather than written down here, so the reference the density
    axis is expressed in cannot drift from the configs the policies were trained on.
    test/config is not a package, and 'test' would shadow the standard library's module,
    so it is loaded by path -- the same way generate_density_configs.py does it.
    """
    path = PROJECT_ROOT / "test/config/generate_fleet_configs.py"
    spec = importlib.util.spec_from_file_location("_generate_fleet_configs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = __import__("yaml").safe_load(module.TEMPLATE_PATH.read_text())
    half_width = module.goal_half_width(module.REFERENCE_FLEET_SIZE, template["d_safe"])
    return module.REFERENCE_FLEET_SIZE / (2.0 * half_width) ** 2


@dataclass(frozen=True)
class Axis:
    """What varies along the plotted axis, and which cells count as in-distribution."""

    name: str
    value: Callable[[dict[str, str]], float]
    tick: Callable[[float], str]
    axis_label: str
    title: str
    # A cell is in-distribution when the policy is scored under the condition it trained
    # on: its own fleet size on the fleet axis, the training density on the density axis.
    in_distribution: Callable[[int, float], bool]
    note: str


def fleet_axis() -> Axis:
    return Axis(
        name="fleet",
        value=lambda row: float(int(row["eval_fleet_size"])),
        tick=lambda value: f"{int(value)}",
        axis_label="Evaluated on (robots)",
        title="evaluation fleet size",
        in_distribution=lambda train_size, value: float(train_size) == value,
        note="trained and evaluated on the same fleet size",
    )


def density_axis() -> Axis:
    reference = training_density()
    return Axis(
        name="density",
        # As a multiple of the training density: the absolute value (0.0139 robots/m^2)
        # says nothing without it, and the sweep is defined in those multiples. Rounded
        # to two decimals so the sweep's levels come out as the round numbers they are:
        # the CSV stores the density rounded to four decimals, which would otherwise
        # turn 3x into 3.0006x and leave the 1x level just off the in-distribution test.
        value=lambda row: round(float(row["density"]) / reference, 2),
        tick=lambda value: f"{value:g}x",
        axis_label="Evaluated at (x training density)",
        title="evaluation density",
        in_distribution=lambda train_size, value: abs(value - 1.0) < 0.005,
        note="evaluated at the density every policy trained at",
    )


def detect_axis(rows: list[dict[str, str]]) -> str:
    """'density' when a fleet size appears at several densities, else 'fleet'.

    The two scenarios are distinguishable in the data itself: a density sweep holds one
    fleet size at five densities, a fleet sweep holds every fleet size at the training
    density. Results without a 'density' column predate it and can only be the latter.
    """
    per_fleet_densities = defaultdict(set)
    for row in rows:
        if not row.get("density"):
            return "fleet"
        per_fleet_densities[int(row["eval_fleet_size"])].add(row["density"])
    return "density" if any(len(v) > 1 for v in per_fleet_densities.values()) else "fleet"


def display_path(path: Path) -> str:
    """Repo-relative when possible; a relative --output-dir is not under PROJECT_ROOT."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_rows(results_path: Path) -> list[dict[str, str]]:
    if results_path.exists():
        rows = list(csv.DictReader(results_path.open()))
        if rows:
            return rows
    # Fall back to the per-policy CSVs when the merged file is missing or empty.
    rows = []
    for csv_path in sorted(results_path.parent.glob("*_n??.csv")):
        rows.extend(csv.DictReader(csv_path.open()))
    if not rows:
        raise SystemExit(f"No results found in {results_path.parent}")
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - stays inside [0, 1] where the normal approximation does not.

    Matters here because several cells sit at 0.02, where a normal interval would
    dip below zero and imply precision the 50 episodes cannot support.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def titled(label: str, title: str) -> str:
    """Prefix a figure title with what it shows, e.g. 'flow · antipodal ring'.

    Titles describe rather than conclude: the same functions now plot two policies
    and two scenarios, and a conclusion that held for the first MLP data (encoders
    identical, training fleet size irrelevant) is false for flow on the ring.
    """
    return f"{label} — {title}" if label else title


def plot_matrix(rows, metric: str, output_path: Path, axis: Axis, label: str = "") -> None:
    train_sizes = sorted({int(r["train_fleet_size"]) for r in rows})
    eval_sizes = sorted({axis.value(r) for r in rows})
    values = {
        (r["encoder_type"], int(r["train_fleet_size"]), axis.value(r)): float(r[metric])
        for r in rows
    }
    encoders = [e for e in ENCODER_ORDER if any(k[0] == e for k in values)]

    all_values = [v for v in values.values()]
    if metric in RATE_METRICS:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = min(all_values), max(all_values)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_STEPS)

    fig, axes = plt.subplots(
        1, len(encoders),
        figsize=(3.5 * len(encoders) + 1.4, 4.6),
        sharey=True,
    )
    fig.patch.set_facecolor(SURFACE)
    if len(encoders) == 1:
        axes = [axes]

    for ax, encoder in zip(axes, encoders):
        grid = np.array([
            [values.get((encoder, t, e), np.nan) for t in train_sizes]
            for e in eval_sizes
        ])
        ax.set_facecolor(SURFACE)
        mesh = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        for row_idx, eval_size in enumerate(eval_sizes):
            for col_idx, train_size in enumerate(train_sizes):
                value = grid[row_idx, col_idx]
                if np.isnan(value):
                    continue
                # Flip ink to stay legible as the cell darkens.
                shade = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                ink = "#ffffff" if shade > 0.55 else TEXT_PRIMARY
                text = f"{value:.2f}" if metric in RATE_METRICS else f"{value:.1f}"
                ax.text(col_idx, row_idx, text, ha="center", va="center",
                        fontsize=10, color=ink)
                if axis.in_distribution(train_size, eval_size):
                    ax.add_patch(mpatches.Rectangle(
                        (col_idx - 0.5, row_idx - 0.5), 1, 1,
                        fill=False, edgecolor=TEXT_PRIMARY, linewidth=2.0, zorder=3))

        ax.set_xticks(range(len(train_sizes)), [str(t) for t in train_sizes])
        ax.set_yticks(range(len(eval_sizes)), [axis.tick(e) for e in eval_sizes])
        ax.set_xlabel("Trained on (robots)", fontsize=10, color=TEXT_SECONDARY)
        ax.set_title(ENCODER_LABELS.get(encoder, encoder), fontsize=12,
                     color=TEXT_PRIMARY, pad=10)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # 2px surface gap between cells, per mark specs.
        ax.set_xticks(np.arange(-0.5, len(train_sizes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(eval_sizes), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)

    axes[0].set_ylabel(axis.axis_label, fontsize=10, color=TEXT_SECONDARY)

    colorbar = fig.colorbar(mesh, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label(METRIC_LABELS.get(metric, metric), fontsize=10, color=TEXT_SECONDARY)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, length=0)
    colorbar.outline.set_visible(False)

    episodes = int(rows[0]["episodes"])
    # Titles sit above the panel row; panel titles own the band just under them.
    fig.suptitle(
        titled(label, f"{METRIC_LABELS.get(metric, metric)} by training fleet size and {axis.title}"),
        fontsize=13, color=TEXT_PRIMARY, x=0.02, ha="left", y=1.10)
    fig.text(0.02, 1.045,
             f"{episodes} episodes per cell; outlined cells are in-distribution ({axis.note})",
             fontsize=9, color=TEXT_MUTED, ha="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {display_path(output_path)}")


def plot_by_axis(rows, output_path: Path, axis: Axis, label: str = "") -> None:
    """Success rate along the axis, pooled over training fleet size.

    Pooling is what makes the comparison readable: per cell there are only 50
    episodes, so any single row of the matrix is dominated by sampling noise.
    """
    eval_sizes = sorted({axis.value(r) for r in rows})
    pooled = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (row["encoder_type"], axis.value(row))
        episodes = int(row["episodes"])
        pooled[key][0] += round(float(row["success_rate"]) * episodes)
        pooled[key][1] += episodes

    encoders = [e for e in ENCODER_ORDER if any(k[0] == e for k in pooled)]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(eval_sizes))
    for encoder in encoders:
        rates, lows, highs = [], [], []
        for eval_size in eval_sizes:
            successes, total = pooled[(encoder, eval_size)]
            rate = successes / total if total else np.nan
            low, high = wilson_interval(successes, total)
            rates.append(rate)
            lows.append(rate - low)
            highs.append(high - rate)
        color = SERIES_COLORS[encoder]
        ax.errorbar(x, rates, yerr=[lows, highs], color=color, linewidth=2.0,
                    marker="o", markersize=8, capsize=4, elinewidth=1.5,
                    markeredgecolor=SURFACE, markeredgewidth=2,
                    label=ENCODER_LABELS.get(encoder, encoder), zorder=3)
        # No direct end-labels: the three series converge at both ends, so labels
        # there land on top of each other. The legend carries identity instead --
        # marker + text, never color alone, which is what aqua's sub-3:1 contrast
        # against this surface requires.

    total_per_point = pooled[(encoders[0], eval_sizes[0])][1]
    ax.set_xticks(x, [axis.tick(e) for e in eval_sizes])
    ax.set_xlabel(axis.axis_label, fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel("Success rate", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-0.4, len(eval_sizes) - 0.1)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.legend(frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY, loc="upper right")

    fig.suptitle(titled(label, f"Success rate by {axis.title}"),
                 fontsize=13, color=TEXT_PRIMARY, x=0.02, ha="left", y=1.06)
    fig.text(0.02, 1.0,
             f"Pooled over all training fleet sizes ({total_per_point} episodes per point); bars are 95% Wilson intervals",
             fontsize=9, color=TEXT_MUTED, ha="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {display_path(output_path)}")


def plot_by_axis_facets(rows, output_path: Path, axis: Axis, label: str = "") -> None:
    """One panel per training fleet size -- the un-pooled view of plot_by_fleet.

    Each point is a single matrix cell, so the intervals are the honest per-cell
    ones (n=50, roughly +-0.13 at mid-range) rather than the pooled +-0.06.
    """
    eval_sizes = sorted({axis.value(r) for r in rows})
    train_sizes = sorted({int(r["train_fleet_size"]) for r in rows})
    cells = {
        (r["encoder_type"], int(r["train_fleet_size"]), axis.value(r)):
            (round(float(r["success_rate"]) * int(r["episodes"])), int(r["episodes"]))
        for r in rows
    }
    encoders = [e for e in ENCODER_ORDER if any(k[0] == e for k in cells)]

    fig, axes = plt.subplots(1, len(train_sizes), figsize=(3.3 * len(train_sizes) + 0.6, 3.9),
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    if len(train_sizes) == 1:
        axes = [axes]

    x = np.arange(len(eval_sizes))
    handles = []
    for ax, train_size in zip(axes, train_sizes):
        ax.set_facecolor(SURFACE)
        for encoder in encoders:
            rates, lows, highs = [], [], []
            for eval_size in eval_sizes:
                successes, total = cells.get((encoder, train_size, eval_size), (0, 0))
                rate = successes / total if total else np.nan
                low, high = wilson_interval(successes, total)
                rates.append(rate)
                lows.append(max(0.0, rate - low))
                highs.append(max(0.0, high - rate))
            line = ax.errorbar(
                x, rates, yerr=[lows, highs], color=SERIES_COLORS[encoder], linewidth=2.0,
                marker="o", markersize=8, capsize=3, elinewidth=1.2,
                markeredgecolor=SURFACE, markeredgewidth=2,
                label=ENCODER_LABELS.get(encoder, encoder), zorder=3)
            if ax is axes[0]:
                handles.append(line)

        ax.set_xticks(x, [axis.tick(e) for e in eval_sizes])
        ax.set_xlabel(axis.axis_label, fontsize=10, color=TEXT_SECONDARY)
        ax.set_title(f"Trained on {train_size} robots", fontsize=11, color=TEXT_PRIMARY, pad=8)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-0.4, len(eval_sizes) - 0.6)
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)

    axes[0].set_ylabel("Success rate", fontsize=10, color=TEXT_SECONDARY)

    episodes = int(rows[0]["episodes"])
    fig.suptitle(titled(label, f"Success rate by {axis.title}, per training fleet size"),
                 fontsize=13, color=TEXT_PRIMARY, x=0.02, ha="left", y=1.14)
    fig.text(0.02, 1.07,
             f"One point per matrix cell ({episodes} episodes); bars are 95% Wilson intervals",
             fontsize=9, color=TEXT_MUTED, ha="left")
    fig.legend(handles=handles, labels=[ENCODER_LABELS.get(e, e) for e in encoders],
               frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY,
               ncol=len(encoders), loc="upper right", bbox_to_anchor=(0.99, 1.11))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {display_path(output_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metric", default="success_rate", choices=sorted(METRIC_LABELS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    parser.add_argument(
        "--policy", choices=["mlp", "flow"], default=None,
        help="plot only this policy's rows; required when the results hold more than one",
    )
    parser.add_argument("--label", default="", help="scenario name shown in the titles, e.g. 'antipodal ring'")
    parser.add_argument(
        "--axis", choices=["auto", "fleet", "density"], default="auto",
        help="what varies along the plotted axis; 'auto' reads it off the results",
    )
    args = parser.parse_args()

    rows = load_rows(args.results)
    # Every series is keyed by encoder alone, so a file holding both policies would
    # silently merge each encoder's mlp and flow rows into one line.
    policies = sorted({row.get("policy_type") or "mlp" for row in rows})
    if args.policy is None and len(policies) > 1:
        raise SystemExit(f"{args.results} holds policies {policies}; pass --policy to pick one")
    prefix = "encoder_study"
    if args.policy is not None:
        rows = [row for row in rows if (row.get("policy_type") or "mlp") == args.policy]
        if not rows:
            raise SystemExit(f"no {args.policy} rows in {args.results}")
        prefix = f"encoder_study_{args.policy}"

    label = " · ".join(part for part in (args.policy, args.label) if part)

    axis_name = args.axis if args.axis != "auto" else detect_axis(rows)
    axis = density_axis() if axis_name == "density" else fleet_axis()
    if axis_name == "density" and not all(row.get("density") for row in rows):
        raise SystemExit(
            f"{args.results} has no 'density' column; it predates the density axis. "
            "Re-run test/evaluate_scaling.py, or pass --axis fleet."
        )

    # On the density axis the rows of several fleet sizes would land in the same cell,
    # so each fleet size gets its own set of figures.
    groups = (
        [("", rows)]
        if axis_name == "fleet"
        else [
            (f"_n{size:02d}", [r for r in rows if int(r["eval_fleet_size"]) == size])
            for size in sorted({int(r["eval_fleet_size"]) for r in rows})
        ]
    )
    for suffix, group in groups:
        group_label = label
        if suffix:
            group_label = " · ".join(part for part in (label, f"N={int(suffix[2:])}") if part)
        plot_matrix(group, args.metric,
                    args.output_dir / f"{prefix}_{args.metric}_matrix{suffix}.{args.format}",
                    axis, group_label)
        plot_by_axis(group,
                     args.output_dir / f"{prefix}_by_{axis.name}{suffix}.{args.format}",
                     axis, group_label)
        plot_by_axis_facets(group,
                            args.output_dir / f"{prefix}_by_{axis.name}_facets{suffix}.{args.format}",
                            axis, group_label)


if __name__ == "__main__":
    main()
