"""Markdown tables for the encoder study, from the CSVs ``test/evaluate_scaling.py`` writes.

``test/plot_study_results.py`` turns the same rows into figures. A figure answers "which
line is higher"; a table answers "what exactly was the number", which is what a write-up
has to quote and what a reader checks. Generating both from the same CSV keeps them from
drifting, and means no number in the text is typed by hand.

Per scenario it writes:

* **success per cell** -- the matrix the figures show, episode success with the per-robot
  rate in brackets. The two are different questions: fleet success needs all N robots at
  once and so falls as p^N, while the per-robot rate (GLAS eq. 6) keeps saying something
  at 32 robots. A cell that reads ``0.00 (0.62)`` is a policy that works and a fleet metric
  that cannot see it.
* **failure split per cell** -- collision against timeout, the two ways an episode fails:
  driving into a neighbour, or avoiding without ever arriving.
* **pooled by encoder** with 95% Wilson intervals, the same pooling the line plots use --
  per cell there are only ``episodes`` episodes, so a single row of the matrix is mostly
  sampling noise.
* **secondary metrics** -- closest pair distance, steps taken, policy time per control
  step.

The axis (evaluation fleet size or density) is detected per file exactly as the plots
detect it, and a density file is split per evaluation fleet size for the same reason: the
fleet sizes would otherwise share a cell.

Usage:
    python test/summarize_study_results.py --results outputs/data_mid_best/eval/*.csv \
        --output-dir outputs/data_mid_best/tables
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "test"))

from plot_study_results import (  # noqa: E402
    ENCODER_LABELS, ENCODER_ORDER, Axis, density_axis, detect_axis, display_path,
    fleet_axis, load_rows, wilson_interval,
)

Row = dict[str, str]


def markdown_table(header: list[str], body: list[list[str]]) -> str:
    """A GitHub-flavoured table. First column left-aligned, the value columns right."""
    align = [":--"] + ["--:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines)


def rate(value: str | None) -> float:
    """A rate column as a float; blank or missing becomes NaN rather than 0.0."""
    return float(value) if value not in (None, "") else float("nan")


def dot(value: float) -> str:
    """'.31' rather than '0.31' -- the leading zero carries no information in a grid."""
    if value != value:
        return "--"
    return f"{value:.2f}".replace("0.", ".", 1) if value < 1.0 else f"{value:.2f}"


def encoders_in(rows: list[Row]) -> list[str]:
    present = {row["encoder_type"] for row in rows}
    return [encoder for encoder in ENCODER_ORDER if encoder in present]


def trials(row: Row, per_robot: bool) -> int:
    """The denominator of a rate: episodes, or episodes x robots for a per-robot rate."""
    return int(row["episodes"]) * (int(row["eval_fleet_size"]) if per_robot else 1)


def success_cells(rows: list[Row], axis: Axis) -> str:
    """Episode success with the per-robot rate in brackets, one row per encoder x train N."""
    levels = sorted({axis.value(row) for row in rows})
    by_cell = {
        (row["encoder_type"], int(row["train_fleet_size"]), axis.value(row)): row
        for row in rows
    }
    body = []
    for encoder in encoders_in(rows):
        for train_size in sorted({int(row["train_fleet_size"]) for row in rows}):
            cells = []
            for level in levels:
                row = by_cell.get((encoder, train_size, level))
                if row is None:
                    cells.append("--")
                    continue
                text = f"{dot(rate(row['success_rate']))} ({dot(rate(row.get('robot_success_rate')))})"
                # In-distribution: scored under the condition it trained on. Bold rather
                # than the figures' outline, which markdown has no equivalent for.
                cells.append(f"**{text}**" if axis.in_distribution(train_size, level) else text)
            body.append([f"{ENCODER_LABELS[encoder]}, trained N={train_size}", *cells])
    return markdown_table(["Policy", *(axis.tick(level) for level in levels)], body)


def failure_cells(rows: list[Row], axis: Axis) -> str:
    """Collision / timeout per cell, episode-level."""
    levels = sorted({axis.value(row) for row in rows})
    by_cell = {
        (row["encoder_type"], int(row["train_fleet_size"]), axis.value(row)): row
        for row in rows
    }
    body = []
    for encoder in encoders_in(rows):
        for train_size in sorted({int(row["train_fleet_size"]) for row in rows}):
            cells = []
            for level in levels:
                row = by_cell.get((encoder, train_size, level))
                cells.append(
                    "--" if row is None
                    else f"{dot(rate(row['collision_rate']))} / {dot(rate(row['timeout_rate']))}"
                )
            body.append([f"{ENCODER_LABELS[encoder]}, trained N={train_size}", *cells])
    return markdown_table(["Policy (collision / timeout)",
                           *(axis.tick(level) for level in levels)], body)


def pooled(rows: list[Row], axis: Axis, metric: str) -> str:
    """One row per encoder, pooled over training fleet size, with 95% Wilson intervals.

    Pooling is what makes the comparison readable, and it is the assumption the facet
    figures exist to expose: it buys a tighter interval but would hide a real
    training-fleet effect. The per-cell tables above are the un-pooled view.
    """
    per_robot = metric.startswith("robot_")
    levels = sorted({axis.value(row) for row in rows})
    counts: dict[tuple[str, float], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (row["encoder_type"], axis.value(row))
        total = trials(row, per_robot)
        counts[key][0] += round(rate(row[metric]) * total)
        counts[key][1] += total

    body = []
    for encoder in encoders_in(rows):
        cells = []
        for level in levels:
            successes, total = counts[(encoder, level)]
            if total == 0:
                cells.append("--")
                continue
            low, high = wilson_interval(successes, total)
            cells.append(f"{dot(successes / total)} [{dot(low)}–{dot(high)}]")
        body.append([ENCODER_LABELS[encoder], *cells])
    denominator = counts[(encoders_in(rows)[0], levels[0])][1]
    unit = "robot-episodes" if per_robot else "episodes"
    table = markdown_table(["Encoder", *(axis.tick(level) for level in levels)], body)
    return f"{table}\n\n{denominator} {unit} per cell, pooled over the training fleet sizes."


def secondary(rows: list[Row], axis: Axis) -> str:
    """Closest approach, episode length and inference cost, per axis level and encoder."""
    levels = sorted({axis.value(row) for row in rows})
    encoders = encoders_in(rows)
    grouped: dict[tuple[str, float], list[Row]] = defaultdict(list)
    for row in rows:
        grouped[(row["encoder_type"], axis.value(row))].append(row)

    def mean(cell_rows: list[Row], column: str) -> float:
        values = [rate(row.get(column)) for row in cell_rows]
        values = [value for value in values if value == value]
        return sum(values) / len(values) if values else float("nan")

    blocks = []
    for column, title, fmt in (
        ("mean_min_pair_distance", "Mean closest pair distance (m), 1.0 m is a collision",
         lambda v: f"{v:.2f}"),
        ("mean_steps", "Mean steps per episode", lambda v: f"{v:.0f}"),
        ("mean_action_ms", "Mean policy call (ms per control step, per robot in brackets)",
         None),
    ):
        body = []
        for level in levels:
            cells = []
            for encoder in encoders:
                cell_rows = grouped.get((encoder, level), [])
                if not cell_rows:
                    cells.append("--")
                elif fmt is None:
                    cells.append(f"{mean(cell_rows, 'mean_action_ms'):.2f} "
                                 f"({mean(cell_rows, 'mean_action_ms_per_robot'):.2f})")
                else:
                    cells.append(fmt(mean(cell_rows, column)))
            body.append([axis.tick(level), *cells])
        header = [axis.axis_label, *(ENCODER_LABELS[encoder] for encoder in encoders)]
        blocks.append(f"**{title}**\n\n{markdown_table(header, body)}")
    return "\n\n".join(blocks)


def scenario_groups(rows: list[Row], axis_name: str) -> list[tuple[str, list[Row]]]:
    """A density file splits per evaluation fleet size; a fleet file is one group.

    The same split the figures make, and for the same reason: at one density level the
    rows of N=2 and N=6 would land in the same cell.
    """
    if axis_name == "fleet":
        return [("", rows)]
    return [
        (f"N={size}", [row for row in rows if int(row["eval_fleet_size"]) == size])
        for size in sorted({int(row["eval_fleet_size"]) for row in rows})
    ]


def scenario_section(name: str, rows: list[Row], train_density: float) -> tuple[str, str]:
    """(full section, pooled-only section) for one scenario CSV."""
    axis_name = detect_axis(rows)
    axis = density_axis(train_density) if axis_name == "density" else fleet_axis()
    episodes = sorted({int(row["episodes"]) for row in rows})
    checkpoints = len({row["checkpoint"] for row in rows})

    preamble = (
        f"{checkpoints} checkpoints x {len({row['config'] for row in rows})} configs, "
        f"{'/'.join(str(count) for count in episodes)} episodes per cell, "
        f"step budget {'/'.join(sorted({row['steps'] for row in rows}, key=int))} steps, "
        f"action noise {'/'.join(sorted({row['action_noise_std'] for row in rows}))}. "
        f"Bold marks the in-distribution cells ({axis.note})."
    )
    if episodes == [1]:
        preamble += (
            "\n\n> The ring collapsed to a single episode per cell: from a fixed start "
            "with no action noise a deterministic MLP has only one trajectory, so these "
            "are pass/fail, not rates."
        )

    full = [f"## {name}\n\n{preamble}"]
    pooled_only = [f"## {name}\n\n{preamble}"]
    for suffix, group in scenario_groups(rows, axis_name):
        heading = f" ({suffix})" if suffix else ""
        pooled_block = (
            f"### Pooled by encoder{heading}\n\n"
            f"Episode success:\n\n{pooled(group, axis, 'success_rate')}\n\n"
            f"Per-robot success:\n\n{pooled(group, axis, 'robot_success_rate')}"
        )
        full.append(
            f"### Success per cell{heading}\n\n"
            "Episode success, per-robot success in brackets.\n\n"
            f"{success_cells(group, axis)}\n\n"
            f"### Failure split per cell{heading}\n\n{failure_cells(group, axis)}\n\n"
            f"{pooled_block}\n\n"
            f"### Secondary metrics{heading}\n\n{secondary(group, axis)}"
        )
        pooled_only.append(pooled_block)
    return "\n\n".join(full), "\n\n".join(pooled_only)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="one or more scenario CSVs, e.g. outputs/<exp>/eval/*.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=["mlp", "flow"], default=None,
                        help="tabulate only this policy's rows; required when a file holds "
                             "more than one, since a cell is keyed by encoder alone")
    parser.add_argument("--train-density", type=float, default=1.0,
                        help="density the evaluated policies trained at, in multiples of "
                             "the reference density (data_mid/data_large train at it, so 1)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = ["# Encoder study — results summary",
               "",
               "Pooled success per axis, generated by `test/summarize_study_results.py`. "
               "The per-scenario files beside this one hold the full matrices, the failure "
               "split and the secondary metrics."]
    for results_path in args.results:
        rows = load_rows(results_path)
        policies = sorted({row.get("policy_type") or "mlp" for row in rows})
        if args.policy is None and len(policies) > 1:
            raise SystemExit(f"{results_path} holds policies {policies}; pass --policy")
        if args.policy is not None:
            rows = [row for row in rows if (row.get("policy_type") or "mlp") == args.policy]
            if not rows:
                raise SystemExit(f"no {args.policy} rows in {results_path}")

        name = results_path.stem
        full, pooled_only = scenario_section(name, rows, args.train_density)
        output_path = args.output_dir / f"{name}.md"
        output_path.write_text(
            f"# Encoder study — {name}\n\n"
            f"Generated by `test/summarize_study_results.py` from "
            f"`{display_path(results_path)}`.\n\n{full}\n"
        )
        print(f"wrote {display_path(output_path)}")
        summary.append(pooled_only)

    summary_path = args.output_dir / "summary.md"
    summary_path.write_text("\n\n".join(summary) + "\n")
    print(f"wrote {display_path(summary_path)}")


if __name__ == "__main__":
    main()
