"""Plot encoder-scaling study results produced by test/evaluate_scaling.py.

The study evaluates on two axes, each of which varies exactly one quantity: evaluation
**fleet size** at the training density, and evaluation **density** at a fixed fleet
size. ``--axis`` picks which one is plotted;
``auto`` reads it off the results, since a density sweep holds several densities per
fleet size and a fleet sweep exactly one.

Four figures per axis:

* ``*_matrix.pdf`` - one panel per encoder, training fleet size across, the axis down,
  one metric per cell. The in-distribution cells are outlined: trained and evaluated on
  the same fleet size, or evaluated at the training density.
* ``*_by_<axis>.pdf`` - the same success rates as lines along the axis with 95% Wilson
  intervals, pooled over training fleet size. The matrix shows the cells; this shows
  whether the differences between them survive the sample.
* ``*_by_<axis>_facets.pdf`` - one panel per training fleet size, so the pooled figure's
  assumption is visible rather than implied. Pooling buys a tighter interval but would
  hide a real training-fleet effect if one existed.
* ``*_by_train_fleet.pdf`` - the transpose of ``*_by_<axis>.pdf``: training fleet size
  across, pooled over the axis instead of over the training fleet sizes. Where the curve
  peaks is the fleet size that was worth training on, which is the question the other
  three cannot answer. ``--train-fleet-pooling`` picks how the axis is pooled.

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
import re
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

REFERENCE_FLEET_CONFIG = "test/config/study/fleet/unicycle2_n02.yaml"
DEFAULT_RESULTS = PROJECT_ROOT / "outputs/study2/eval/random/encoder_scaling.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/study2/plots/random"

ENCODER_ORDER = ("deepset", "transformer", "gnn")
ENCODER_LABELS = {"deepset": "DeepSet", "transformer": "Transformer", "gnn": "GNN"}

SEQUENTIAL_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SERIES_COLORS = {"deepset": "#2a78d6", "transformer": "#eb6834", "gnn": "#1baf7a"}

STATUS_GOOD = "#6cc76c"
STATUS_WARNING = "#fbd073"

SURFACE = "#fcfcfb"
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
FAILURE_COLUMNS = {
    "success_rate": ("collision_rate", "timeout_rate"),
    "robot_success_rate": ("robot_collision_rate", "robot_timeout_rate"),
}


def training_density() -> float:
    """Robots per m^2 the policies train at, and the 1x level of the density axis."""
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


V0_FROM_CONFIG_RE = re.compile(r"v(\d+)")


def v0_from_config(config_path: str) -> float:
    """Initial speed a crash config was generated with, in m/s."""
    match = V0_FROM_CONFIG_RE.search(os.path.basename(config_path))
    if not match:
        raise SystemExit(f"crash config filename '{config_path}' has no v<NNNN> tag")
    return int(match.group(1)) / 1000.0


def v0_axis() -> Axis:
    """Crash-ladder axis: initial speed the robots already carry at t=0."""
    return Axis(
        name="v0",
        value=lambda row: v0_from_config(row["config"]),
        tick=lambda value: f"{value:g}",
        axis_label="Initial speed v0 (m/s)",
        title="crash ladder",
        in_distribution=lambda train_size, value: False,
        note="two-robot head-on ladder at rising initial speeds",
    )


def density_axis(train_density_factor: float = 1.0) -> Axis:
    """`train_density_factor` in units of the reference density: the row to outline."""
    reference = training_density()
    return Axis(
        name="density",
        value=lambda row: round(float(row["density"]) / reference, 2),
        tick=lambda value: f"{value:g}x",
        axis_label="Evaluated at (x training density)",
        title="evaluation density",
        in_distribution=lambda train_size, value: abs(value - train_density_factor) < 0.005,
        note=f"evaluated at the density the policies trained at ({train_density_factor:g}x)",
    )


def detect_axis(rows: list[dict[str, str]]) -> str:
    """'density' when a fleet size appears at several densities, else 'fleet'."""
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
    rows = []
    for csv_path in sorted(results_path.parent.glob("*_n??.csv")):
        rows.extend(csv.DictReader(csv_path.open()))
    if not rows:
        raise SystemExit(f"No results found in {results_path.parent}")
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - stays inside [0, 1] where the normal approximation does not."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    # Snap to p: at 0 or 1 successes rounding can give a negative yerr, which matplotlib rejects.
    return (min(p, max(0.0, center - margin)), max(p, min(1.0, center + margin)))


def titled(label: str, title: str) -> str:
    """Prefix a figure title with its label, e.g. 'flow · antipodal ring'."""
    return f"{label} — {title}" if label else title


NO_TITLE = False


def save_figure(fig, output_path: Path, *header) -> None:
    """Write the figure, then the same figure again without its header text, as a PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if NO_TITLE:
        for artist in header:
            if artist is not None:
                artist.set_visible(False)
        fig.savefig(output_path, bbox_inches="tight", facecolor=SURFACE, dpi=PNG_DPI)
        print(f"wrote {display_path(output_path)}")
        plt.close(fig)
        return

    fig.savefig(output_path, bbox_inches="tight", facecolor=SURFACE, dpi=PNG_DPI)
    print(f"wrote {display_path(output_path)}")

    dropped = [artist for artist in header if artist is not None]
    if dropped:
        for artist in dropped:
            artist.set_visible(False)
        untitled = output_path.with_name(f"{output_path.stem}_notitle.png")
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

    outcomes = {}
    has_rates = all(
        row.get(column) not in (None, "")
        for row in rows for column in ("success_rate", "collision_rate", "timeout_rate")
    )
    if outcome_colors and has_rates:
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
        collided = [v for key, v in values.items() if outcomes.get(key) == "collision"]
        vmin, vmax = (min(collided), max(collided)) if collided else (min(all_values), max(all_values))
    else:
        vmin, vmax = min(all_values), max(all_values)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_STEPS)
    if outcomes:
        cmap = cmap.copy()
        cmap.set_bad(SURFACE)

    fig, axes = plt.subplots(
        1, len(encoders),
        figsize=(3.5 * len(encoders) + 1.4, 1.6 + 0.5 * len(eval_sizes)),
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
                    ax.add_patch(mpatches.Rectangle(
                        (col_idx - 0.5, row_idx - 0.5), 1, 1, zorder=1,
                        facecolor=STATUS_GOOD if outcome == "success" else STATUS_WARNING,
                        edgecolor="none"))
                    ink = TEXT_PRIMARY
                else:
                    shade = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                    ink = "#ffffff" if shade > 0.55 else TEXT_PRIMARY
                if metric in RATE_METRICS or vmax < 10.0:
                    text = f"{value:.2f}"
                else:
                    text = f"{value:.1f}"
                cell = failures.get((encoder, train_size, eval_size)) if failures else None
                if cell is None:
                    ax.text(col_idx, row_idx, text, ha="center", va="center",
                            fontsize=10, color=ink, zorder=4)
                else:
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
    title = fig.suptitle(
        titled(label, f"{METRIC_LABELS.get(metric, metric)} by training fleet size and {axis.title}"),
        fontsize=13, color=TEXT_PRIMARY, x=0.02, ha="left", y=1.10)
    note = f"{episodes} episodes per cell; outlined cells are in-distribution ({axis.note})"
    if failures:
        note += "; C = collision rate, T = timeout rate"
    if outcomes:
        note += "; fill is the outcome" + ("" if episodes == 1 else " of most episodes")
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
    """Success rate along the axis, pooled over training fleet size."""
    per_robot = metric.startswith("robot_")
    eval_sizes = sorted({axis.value(r) for r in rows})
    pooled = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (row["encoder_type"], axis.value(row))
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
    """One panel per training fleet size -- the un-pooled view of plot_by_fleet."""
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


def plot_by_train_fleet(rows, output_path: Path, axis: Axis, label: str = "",
                        metric: str = "success_rate", pooling: str = "macro") -> None:
    """The transpose of ``plot_by_axis``: training fleet size across, pooled over the axis."""
    if pooling not in {"macro", "micro"}:
        raise ValueError("'pooling' must be 'macro' or 'micro'.")
    per_robot = metric.startswith("robot_")
    train_sizes = sorted({int(r["train_fleet_size"]) for r in rows})
    axis_values = sorted({axis.value(r) for r in rows})

    cells: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (row["encoder_type"], int(row["train_fleet_size"]), axis.value(row))
        trials = int(row["episodes"]) * (int(row["eval_fleet_size"]) if per_robot else 1)
        cells[key][0] += round(float(row[metric]) * trials)
        cells[key][1] += trials

    def pooled(encoder: str | None, train_size: int) -> tuple[float, float]:
        """(rate, half-width of the 95% interval) over every axis value, for one column."""
        rates, variances, successes, trials = [], [], 0, 0
        for value in axis_values:
            keys = ([(encoder, train_size, value)] if encoder is not None
                    else [(e, train_size, value) for e in ENCODER_ORDER])
            k = sum(cells[key][0] for key in keys if key in cells)
            n = sum(cells[key][1] for key in keys if key in cells)
            if not n:
                continue
            successes += k
            trials += n
            rate = k / n
            rates.append(rate)
            variances.append(rate * (1.0 - rate) / n)
        if not rates:
            return (np.nan, 0.0)
        if pooling == "micro":
            rate = successes / trials
            low, high = wilson_interval(successes, trials)
            return (rate, max(rate - low, high - rate))
        mean = float(np.mean(rates))
        return (mean, 1.96 * math.sqrt(sum(variances)) / len(rates))

    encoders = [e for e in ENCODER_ORDER if any(k[0] == e for k in cells)]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(train_sizes))

    combined = [pooled(None, t) for t in train_sizes]
    ax.plot(x, [c[0] for c in combined], color=TEXT_SECONDARY, linewidth=2.6,
            linestyle=(0, (5, 2)), marker="D", markersize=7, markerfacecolor=SURFACE,
            markeredgewidth=2, label="All encoders", zorder=2)

    for encoder in encoders:
        points = [pooled(encoder, t) for t in train_sizes]
        ax.errorbar(x, [p[0] for p in points], yerr=[p[1] for p in points],
                    color=SERIES_COLORS[encoder], linewidth=2.0, marker="o", markersize=8,
                    capsize=4, elinewidth=1.5, markeredgecolor=SURFACE, markeredgewidth=2,
                    label=ENCODER_LABELS.get(encoder, encoder), zorder=3)

    ax.set_xticks(x, [str(t) for t in train_sizes])
    ax.set_xlabel("Trained on (robots)", fontsize=10, color=TEXT_SECONDARY)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10, color=TEXT_SECONDARY)
    top = max(c[0] + c[1] for c in combined if not np.isnan(c[0]))
    ax.set_ylim(-0.03, max(0.35, min(1.03, top * 1.35)))
    ax.set_xlim(-0.4, len(train_sizes) - 0.6 + 0.5)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.legend(frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY, loc="upper right")

    unit = "robot-episodes" if per_robot else "episodes"
    how = (f"mean of the {len(axis_values)} per-{axis.name} rates, each weighted equally"
           if pooling == "macro" else f"weighted by {unit}")
    title = fig.suptitle(titled(label, f"{METRIC_LABELS.get(metric, metric)} by training fleet size"),
                         fontsize=13, color=TEXT_PRIMARY, x=0.02, ha="left", y=1.06)
    subtitle = fig.text(0.02, 1.0,
             f"Pooled over all {len(axis_values)} {axis.name} conditions ({how}); "
             "bars are 95% sampling intervals",
             fontsize=9, color=TEXT_MUTED, ha="left")
    save_figure(fig, output_path, title, subtitle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, nargs="+", default=[DEFAULT_RESULTS])
    parser.add_argument("--metric", default="success_rate", choices=sorted(METRIC_LABELS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    parser.add_argument(
        "--policy", choices=["mlp", "flow", "both"], default=None,
        help="plot only this policy's rows; required when the results hold more than one. "
             "'both' deliberately pools the heads into one set of series -- each point then "
             "averages the two, weighted by their episode counts",
    )
    parser.add_argument("--label", default="", help="scenario name shown in the titles, e.g. 'antipodal ring'")
    parser.add_argument("--train-density", type=float, default=1.0,
                        help="density the evaluated policies trained at, in multiples of the "
                             "reference density (data_mid/data_large train at it, so 1); "
                             "decides which row of the density matrix is in-distribution")
    parser.add_argument("--color-by-outcome", action=argparse.BooleanOptionalAction, default=True,
                        help="colour each matrix cell by what happened -- green success, "
                             "yellow timeout, the blue ramp for collisions shaded by the "
                             "metric (default). --no-color-by-outcome restores a plain "
                             "sequential ramp over the metric alone")
    parser.add_argument("--no-failure-modes", action="store_true",
                        help="only the metric per cell, without the collision/timeout split")
    parser.add_argument(
        "--axis", choices=["auto", "fleet", "density", "v0"], default="auto",
        help="what varies along the plotted axis; 'auto' reads it off the results. "
             "'v0' is the crash-ladder axis and must be passed explicitly",
    )
    parser.add_argument(
        "--figures", default="all",
        help="comma-separated subset of {matrix, by_axis, facets, by_train_fleet}, "
             "or 'all' (default)",
    )
    parser.add_argument(
        "--no-title", action="store_true",
        help="write the figures with no title or note, for a document that captions them "
             "itself; suppresses the separate _notitle.png companion",
    )
    parser.add_argument(
        "--train-fleet-pooling", choices=["macro", "micro"], default="macro",
        help="how the by-training-fleet figure pools the evaluation conditions: 'macro' "
             "weights each condition equally (default), 'micro' weights by robot-episodes, "
             "which hands most of the weight to the largest fleet",
    )
    args = parser.parse_args()

    known = {"matrix", "by_axis", "facets", "by_train_fleet"}
    figures = known if args.figures == "all" else {f.strip() for f in args.figures.split(",")}
    if unknown := figures - known:
        raise SystemExit(f"unknown figure(s) {sorted(unknown)}; choose from {sorted(known)}")

    global NO_TITLE
    NO_TITLE = args.no_title

    rows = [row for path in args.results for row in load_rows(path)]
    policies = sorted({row.get("policy_type") or "mlp" for row in rows})
    if args.policy is None and len(policies) > 1:
        raise SystemExit(
            f"{[str(p) for p in args.results]} holds policies {policies}; pass --policy to "
            "pick one, or --policy both to pool them"
        )
    prefix = "encoder_study"
    if args.policy is not None:
        if args.policy != "both":
            rows = [row for row in rows if (row.get("policy_type") or "mlp") == args.policy]
        if not rows:
            raise SystemExit(f"no {args.policy} rows in {[str(p) for p in args.results]}")
        prefix = f"encoder_study_{args.policy}"

    label = " · ".join(part for part in (args.policy, args.label) if part)

    axis_name = args.axis if args.axis != "auto" else detect_axis(rows)
    if axis_name == "density":
        axis = density_axis(args.train_density)
    elif axis_name == "v0":
        axis = v0_axis()
    else:
        axis = fleet_axis()
    if axis_name == "density" and not all(row.get("density") for row in rows):
        raise SystemExit(
            f"{args.results} has no 'density' column; it predates the density axis. "
            "Re-run test/evaluate_scaling.py, or pass --axis fleet."
        )

    if axis_name == "density":
        groups = [
            (f"_n{size:02d}", [r for r in rows if int(r["eval_fleet_size"]) == size])
            for size in sorted({int(r["eval_fleet_size"]) for r in rows})
        ]
    else:
        groups = [("", rows)]

    if axis_name == "v0":
        figures = figures & {"by_axis", "facets"}
    for suffix, group in groups:
        group_label = label
        if suffix:
            group_label = " · ".join(part for part in (label, f"N={int(suffix[2:])}") if part)
        if "matrix" in figures:
            plot_matrix(group, args.metric,
                        args.output_dir / f"{prefix}_{args.metric}_matrix{suffix}.{args.format}",
                        axis, group_label, show_failures=not args.no_failure_modes,
                        outcome_colors=args.color_by_outcome)
        line_metric = args.metric if args.metric in FAILURE_COLUMNS else "success_rate"
        if "by_axis" in figures:
            plot_by_axis(group,
                         args.output_dir / f"{prefix}_{line_metric}_by_{axis.name}{suffix}.{args.format}",
                         axis, group_label, metric=line_metric)
        if "facets" in figures:
            plot_by_axis_facets(group,
                                args.output_dir / f"{prefix}_{line_metric}_by_{axis.name}_facets{suffix}.{args.format}",
                                axis, group_label, metric=line_metric)
        if "by_train_fleet" in figures:
            plot_by_train_fleet(group,
                                args.output_dir / f"{prefix}_{line_metric}_by_train_fleet{suffix}.{args.format}",
                                axis, group_label, metric=line_metric,
                                pooling=args.train_fleet_pooling)


if __name__ == "__main__":
    main()
