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
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

# The 1x level of both evaluation axes: the density the policies train at.
REFERENCE_FLEET_CONFIG = "test/config/study/fleet/unicycle2_n02.yaml"
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

# Status palette: reserved for the outcome of a cell, never reused as a series colour.
# Used by --color-by-outcome, where the number in a cell is a continuous metric but the
# fill says which of the three outcomes produced it.
#
# These are the status steps (#0ca30c good, #fab219 warning) at 60% over the surface: a
# full-strength fill behind a whole grid of cells reads louder than the numbers it is
# meant to support. Near-black ink sits at 9.4:1 on the green and 13.5:1 on the yellow.
#
# The cost is colour-blind separation. The pair measures OKLab dE 18.6 in normal vision
# and 13.5 deutan, both comfortable, but 6.7 protan -- inside the 6-8 band, so a protan
# reader may not separate a success cell from a timeout cell by fill. The legend names
# all three states, and the paler the tint the worse this gets: at 50% protan falls to
# 5.5 and normal vision to 15.7, which is the hard floor. Raise the factor toward 1.0
# (protan 10.6) if the distinction has to survive protanopia unaided.
STATUS_GOOD = "#6cc76c"       # the scenario succeeded
STATUS_WARNING = "#fbd073"    # the scenario ran out of steps

SURFACE = "#fcfcfb"
# Raster output resolution. 300 is the print/thesis standard: at these figure sizes it
# gives ~3500 px across, which still reads when a matrix is scaled down into a column.
# PDF is vector, so this only affects the PNG companions.
PNG_DPI = 300
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8983"
GRID = "#e3e2de"

METRIC_LABELS = {
    "success_rate": "Success rate (episodes)",
    "collision_rate": "Collision rate (episodes)",
    "timeout_rate": "Timeout rate (episodes)",
    "robot_success_rate": "Success rate (robots)",
    "robot_collision_rate": "Collision rate (robots)",
    "robot_timeout_rate": "Timeout rate (robots)",
    "mean_steps": "Mean steps",
    "mean_goal_position_error": "Mean goal position error (m)",
    "mean_goal_heading_error": "Mean goal heading error (rad)",
    "mean_min_pair_distance": "Mean min pair distance",
    "mean_action_ms": "Mean policy call (ms/step)",
    "mean_action_ms_per_robot": "Mean policy call (ms/step/robot)",
}
RATE_METRICS = {
    "success_rate", "collision_rate", "timeout_rate",
    "robot_success_rate", "robot_collision_rate", "robot_timeout_rate",
}
# The two failure modes belonging to a success metric, shown under the value in each
# matrix cell. Episode rates and per-robot rates must not be mixed: they have different
# denominators (episodes against robots), so a cell would not add up.
FAILURE_COLUMNS = {
    "success_rate": ("collision_rate", "timeout_rate"),
    "robot_success_rate": ("robot_collision_rate", "robot_timeout_rate"),
}


def training_density() -> float:
    """Robots per m^2 the policies train at, and the 1x level of the density axis.

    Read off the reference fleet config rather than written down here, so the unit the
    density axis is expressed in cannot drift from the scenarios themselves. The 1x
    density level of test/config/study/density/ is generated from the same file.
    """
    config = yaml.safe_load((PROJECT_ROOT / REFERENCE_FLEET_CONFIG).read_text())
    num_robots = len(config["robots"])
    half_width = float(config["robots"][0]["config"]["workspace_bounds"][1])
    return num_robots / (2.0 * half_width) ** 2


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


def density_axis(train_density_factor: float = 1.0) -> Axis:
    """`train_density_factor` in units of the reference density: the row to outline.

    Which row is in-distribution depends on the runs the results came from, not on the
    axis, so it has to be passed in. data_mid and data_large train at 0.167 robots/m^2,
    which is the reference density itself, hence the default of 1; an earlier grid trained
    at a third of it and needed 3.
    """
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
        in_distribution=lambda train_size, value: abs(value - train_density_factor) < 0.005,
        note=f"evaluated at the density the policies trained at ({train_density_factor:g}x)",
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
    # Snapped to p as well as to [0, 1]: the interval contains the observed rate by
    # construction, but at 0 or 1 successes the two sides cancel only up to rounding,
    # leaving a bound off by ~1e-18. Callers subtract these to get error bars, and
    # matplotlib rejects a yerr of -7e-18.
    return (min(p, max(0.0, center - margin)), max(p, min(1.0, center + margin)))


def titled(label: str, title: str) -> str:
    """Prefix a figure title with what it shows, e.g. 'flow · antipodal ring'.

    Titles describe rather than conclude: the same functions now plot two policies
    and two scenarios, and a conclusion that held for the first MLP data (encoders
    identical, training fleet size irrelevant) is false for flow on the ring.
    """
    return f"{label} — {title}" if label else title


def save_figure(fig, output_path: Path, *header) -> None:
    """Write the figure, then the same figure again without its header text, as a PNG.

    A document that carries its own caption would otherwise print the title and its note
    twice, and cropping them off by hand re-renders at a different size. The second file
    drops every artist in `header` -- the suptitle and the note under it -- and keeps
    everything inside the axes, including the legend, which carries meaning rather than
    description. It is always a PNG, whatever --format the primary is, since that is what
    a slide or a document wants to embed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=SURFACE, dpi=PNG_DPI)
    print(f"wrote {display_path(output_path)}")

    dropped = [artist for artist in header if artist is not None]
    if dropped:
        for artist in dropped:
            artist.set_visible(False)
        untitled = output_path.with_name(f"{output_path.stem}_notitle.png")
        # bbox_inches="tight" re-crops, so the hidden header leaves no band behind.
        fig.savefig(untitled, bbox_inches="tight", facecolor=SURFACE, dpi=PNG_DPI)
        print(f"wrote {display_path(untitled)}")
    plt.close(fig)


def plot_matrix(rows, metric: str, output_path: Path, axis: Axis, label: str = "",
                show_failures: bool = True, outcome_colors: bool = False) -> None:
    train_sizes = sorted({int(r["train_fleet_size"]) for r in rows})
    eval_sizes = sorted({axis.value(r) for r in rows})
    values = {
        (r["encoder_type"], int(r["train_fleet_size"]), axis.value(r)): float(r[metric])
        for r in rows
    }
    failures = None
    if show_failures and metric in FAILURE_COLUMNS:
        collision_column, timeout_column = FAILURE_COLUMNS[metric]
        if all(r.get(collision_column) not in (None, "") and r.get(timeout_column) not in (None, "")
               for r in rows):
            failures = {
                (r["encoder_type"], int(r["train_fleet_size"]), axis.value(r)):
                    (float(r[collision_column]), float(r[timeout_column]))
                for r in rows
            }
    encoders = [e for e in ENCODER_ORDER if any(k[0] == e for k in values)]

    # Which outcome produced each cell: whichever of the three rates is largest. With
    # one episode per cell that is the episode's own outcome; with many it is the modal
    # one, and the fill should then be read as "mostly", which the note says.
    outcomes = {}
    if outcome_colors:
        for r in rows:
            triple = (
                ("success", float(r["success_rate"])),
                ("collision", float(r["collision_rate"])),
                ("timeout", float(r["timeout_rate"])),
            )
            outcomes[(r["encoder_type"], int(r["train_fleet_size"]), axis.value(r))] = (
                max(triple, key=lambda pair: pair[1])[0]
            )

    all_values = [v for v in values.values()]
    if metric in RATE_METRICS:
        vmin, vmax = 0.0, 1.0
    elif outcomes:
        # The ramp now describes the collision cells alone, so it is scaled to them. Over
        # the whole range the successes -- which are far apart by definition -- would push
        # vmax up and flatten the failures into the palest steps, which is the opposite of
        # what this figure is for.
        collided = [v for key, v in values.items() if outcomes.get(key) == "collision"]
        vmin, vmax = (min(collided), max(collided)) if collided else (min(all_values), max(all_values))
    else:
        vmin, vmax = min(all_values), max(all_values)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_STEPS)
    if outcomes:
        # Non-collision cells are masked out of the mesh and painted as status fills.
        cmap = cmap.copy()
        cmap.set_bad(SURFACE)

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
        shaded = grid
        if outcomes:
            shaded = np.array([
                [grid[r, c] if outcomes.get((encoder, t, e)) == "collision" else np.nan
                 for c, t in enumerate(train_sizes)]
                for r, e in enumerate(eval_sizes)
            ])
            shaded = np.ma.masked_invalid(shaded)
        mesh = ax.imshow(shaded, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        for row_idx, eval_size in enumerate(eval_sizes):
            for col_idx, train_size in enumerate(train_sizes):
                value = grid[row_idx, col_idx]
                if np.isnan(value):
                    continue
                outcome = outcomes.get((encoder, train_size, eval_size))
                if outcome in ("success", "timeout"):
                    # Painted here rather than through the mesh: these two are states,
                    # not magnitudes, so they get a flat status fill under the value.
                    ax.add_patch(mpatches.Rectangle(
                        (col_idx - 0.5, row_idx - 0.5), 1, 1, zorder=1,
                        facecolor=STATUS_GOOD if outcome == "success" else STATUS_WARNING,
                        edgecolor="none"))
                    ink = TEXT_PRIMARY
                else:
                    # Flip ink to stay legible as the cell darkens.
                    shade = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                    ink = "#ffffff" if shade > 0.55 else TEXT_PRIMARY
                # Two decimals on a small scale: mean_min_pair_distance is read against
                # d_collision = 1.0 m, and at one decimal 0.94 and 1.04 both print as
                # ~1.0, hiding the only boundary that matters. Large scales (mean_steps,
                # in the hundreds) do not need the extra digit.
                if metric in RATE_METRICS or vmax < 10.0:
                    text = f"{value:.2f}"
                else:
                    text = f"{value:.1f}"
                cell = failures.get((encoder, train_size, eval_size)) if failures else None
                if cell is None:
                    ax.text(col_idx, row_idx, text, ha="center", va="center",
                            fontsize=10, color=ink, zorder=4)
                else:
                    # Value and failure split on two lines: the split is the secondary
                    # reading, so it sits smaller and below, and the pair stays centred
                    # in the cell rather than the value alone.
                    collision, timeout = cell
                    ax.text(col_idx, row_idx - 0.13, text, ha="center", va="center",
                            fontsize=10, color=ink, zorder=4)
                    ax.text(col_idx, row_idx + 0.17,
                            f"C{collision:.2f}".replace("0.", ".")
                            + " " + f"T{timeout:.2f}".replace("0.", "."),
                            ha="center", va="center", fontsize=7, color=ink, alpha=0.85,
                            zorder=4)
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
    colorbar_label = METRIC_LABELS.get(metric, metric)
    if outcomes:
        colorbar_label += " (collisions only)"
    colorbar.set_label(colorbar_label, fontsize=10, color=TEXT_SECONDARY)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, length=0)
    colorbar.outline.set_visible(False)

    episodes = int(rows[0]["episodes"])
    # Titles sit above the panel row; panel titles own the band just under them.
    title = fig.suptitle(
        titled(label, f"{METRIC_LABELS.get(metric, metric)} by training fleet size and {axis.title}"),
        fontsize=13, color=TEXT_PRIMARY, x=0.02, ha="left", y=1.10)
    note = f"{episodes} episodes per cell; outlined cells are in-distribution ({axis.note})"
    if failures:
        note += "; C = collision rate, T = timeout rate"
    if outcomes:
        note += "; fill is the outcome" + ("" if episodes == 1 else " of most episodes")
        # Below the panels, not in the header: the note already spans most of the width
        # there, and a right-aligned legend lands on top of its tail.
        fig.legend(
            handles=[
                mpatches.Patch(facecolor=STATUS_GOOD, edgecolor="none", label="success"),
                mpatches.Patch(facecolor=STATUS_WARNING, edgecolor="none", label="timeout"),
                mpatches.Patch(facecolor=SEQUENTIAL_STEPS[6], edgecolor="none",
                               label="collision — shade is the value"),
            ],
            frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
            ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.02),
        )
    subtitle = fig.text(0.02, 1.045, note, fontsize=9, color=TEXT_MUTED, ha="left")

    save_figure(fig, output_path, title, subtitle)


def plot_by_axis(rows, output_path: Path, axis: Axis, label: str = "",
                 metric: str = "success_rate") -> None:
    """Success rate along the axis, pooled over training fleet size.

    Pooling is what makes the comparison readable: per cell there are only 50
    episodes, so any single row of the matrix is dominated by sampling noise.
    """
    per_robot = metric.startswith("robot_")
    eval_sizes = sorted({axis.value(r) for r in rows})
    pooled = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (row["encoder_type"], axis.value(row))
        # Wilson intervals need counts, and the denominator differs per metric: an
        # episode rate is over episodes, a per-robot rate over episodes x robots.
        trials = int(row["episodes"]) * (int(row["eval_fleet_size"]) if per_robot else 1)
        pooled[key][0] += round(float(row[metric]) * trials)
        pooled[key][1] += trials

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
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(-0.4, len(eval_sizes) - 0.1)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.legend(frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY, loc="upper right")

    title = fig.suptitle(titled(label, f"{METRIC_LABELS.get(metric, metric)} by {axis.title}"),
                 fontsize=13, color=TEXT_PRIMARY, x=0.02, ha="left", y=1.06)
    subtitle = fig.text(0.02, 1.0,
             f"Pooled over all training fleet sizes ({total_per_point} "
             f"{'robot-episodes' if per_robot else 'episodes'} per point); bars are 95% Wilson intervals",
             fontsize=9, color=TEXT_MUTED, ha="left")

    save_figure(fig, output_path, title, subtitle)


def plot_by_axis_facets(rows, output_path: Path, axis: Axis, label: str = "",
                        metric: str = "success_rate") -> None:
    """One panel per training fleet size -- the un-pooled view of plot_by_fleet.

    Each point is a single matrix cell, so the intervals are the honest per-cell
    ones rather than the pooled ones.

    `metric` must be one of the two success rates. They need different denominators --
    an episode rate is over episodes, a per-robot rate over episodes x robots -- and the
    per-robot one is what keeps saying something once fleet success has fallen to p^N.
    """
    per_robot = metric.startswith("robot_")
    eval_sizes = sorted({axis.value(r) for r in rows})
    train_sizes = sorted({int(r["train_fleet_size"]) for r in rows})
    def trials(row) -> int:
        return int(row["episodes"]) * (int(row["eval_fleet_size"]) if per_robot else 1)

    cells = {
        (r["encoder_type"], int(r["train_fleet_size"]), axis.value(r)):
            (round(float(r[metric]) * trials(r)), trials(r))
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

    axes[0].set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10, color=TEXT_SECONDARY)

    episodes = int(rows[0]["episodes"])
    unit = "robot-episodes" if per_robot else "episodes"
    title = fig.suptitle(
        titled(label, f"{METRIC_LABELS.get(metric, metric)} by {axis.title}, per training fleet size"),
        fontsize=13, color=TEXT_PRIMARY, x=0.02, ha="left", y=1.14)
    subtitle = fig.text(0.02, 1.07,
             f"One point per matrix cell ({episodes} {unit} before the per-robot factor); "
             "bars are 95% Wilson intervals",
             fontsize=9, color=TEXT_MUTED, ha="left")
    fig.legend(handles=handles, labels=[ENCODER_LABELS.get(e, e) for e in encoders],
               frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY,
               ncol=len(encoders), loc="upper right", bbox_to_anchor=(0.99, 1.11))

    save_figure(fig, output_path, title, subtitle)


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
    parser.add_argument("--train-density", type=float, default=1.0,
                        help="density the evaluated policies trained at, in multiples of the "
                             "reference density (data_mid/data_large train at it, so 1); "
                             "decides which row of the density matrix is in-distribution")
    parser.add_argument("--color-by-outcome", action="store_true",
                        help="colour each matrix cell by what happened -- green success, "
                             "yellow timeout, the blue ramp for collisions shaded by the "
                             "metric. For a continuous metric such as "
                             "mean_min_pair_distance, which says how badly a cell failed "
                             "but not whether it failed at all")
    parser.add_argument("--no-failure-modes", action="store_true",
                        help="only the metric per cell, without the collision/timeout split")
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
    axis = density_axis(args.train_density) if axis_name == "density" else fleet_axis()
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
                    axis, group_label, show_failures=not args.no_failure_modes,
                    outcome_colors=args.color_by_outcome)
        # The line plots can only show a success rate. When --metric is something else
        # (a distance, a latency) they fall back to episode success -- so the filename is
        # built from the metric actually plotted, not from the one requested.
        line_metric = args.metric if args.metric in FAILURE_COLUMNS else "success_rate"
        plot_by_axis(group,
                     args.output_dir / f"{prefix}_{line_metric}_by_{axis.name}{suffix}.{args.format}",
                     axis, group_label, metric=line_metric)
        plot_by_axis_facets(group,
                            args.output_dir / f"{prefix}_{line_metric}_by_{axis.name}_facets{suffix}.{args.format}",
                            axis, group_label, metric=line_metric)


if __name__ == "__main__":
    main()
