"""Why the best training fleet size reverses between the arena and the ring.

``test/plot_study_results.py --metric robot_success_rate`` produces a by-training-fleet
curve per scenario, and the two disagree: on ``arena`` the curve falls with the training
fleet size, on ``circle`` it rises. This script draws the explanation as one figure.

The mechanism is a single policy property with two opposite consequences:

**Training on more robots produces a slower, more hesitant policy.** That is scenario
independent -- the fraction of the step budget a policy consumes rises monotonically with
its training fleet size in both scenarios (panel B). What differs is whether hesitancy is
rewarded:

* ``arena`` draws starts and goals at random, so most robots never conflict. The collision
  rate barely moves with the training fleet size (panel C) -- caution buys nothing -- while
  the timeout rate climbs. Hesitancy is pure cost, so the smallest training fleet wins.
* ``circle`` starts every robot on a ring aimed at its antipode, so all N paths cross the
  centre at once: a symmetric head-on conflict that cannot be driven straight through.
  Yielding is the solution, the collision rate falls steeply with the training fleet size
  (panel D) and timeouts stay near zero, so the largest training fleet wins.

The competing hypothesis -- that a policy simply needs to be *evaluated* at the neighbour
count it *trained* at -- is refuted by the density sweep and is not drawn here: at N=6 and
1.25x the training density the visible-neighbour count is 3.96, which matches what the N=6
and N=8 policies trained on (3.50 and 4.33), yet the N=2 policy still wins by better than
2x (0.247 against 0.111 and 0.114). Matching the neighbour count is not the lever; caution
is, and the training fleet size is what sets it.

Usage:
    python test/plot_reversal_explanation.py                      # both heads
    python test/plot_reversal_explanation.py --policy flow
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

from test.plot_study_results import (  # noqa: E402
    GRID, PNG_DPI, SURFACE, TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY, display_path,
)

# Scenario colours from the study's validated categorical slots. Collision and timeout
# reuse the same two so a reader carries one mapping through the whole figure: the left
# column is always the arena hue, the right always the ring hue, and within the failure
# panels the solid line is the mode that binds.
ARENA = "#2a78d6"
RING = "#eb6834"
COLLISION = "#0d366b"
TIMEOUT = "#d99100"

# Where each head keeps the two scenarios. The MLP ring is circle_det (one deterministic
# episode per cell); the flow ring is circle (ten, since its head samples).
SOURCES = {
    "mlp": {"arena": "outputs/data_mid_best/eval/arena.csv",
            "ring": "outputs/data_mid_best/eval/circle_det.csv"},
    "flow": {"arena": "outputs/data_mid_flow/eval/arena.csv",
             "ring": "outputs/data_mid_flow/eval/circle.csv"},
}


def pooled(path: Path) -> dict[int, dict[str, float]]:
    """Per training fleet size, pooled over encoders and evaluation fleet sizes.

    Rates are pooled by their own denominator -- per-robot rates over robot-episodes,
    episode rates over episodes -- so a large evaluation fleet carries the weight it
    actually contributes rather than counting once. ``step_frac`` is the share of the step
    budget consumed, averaged over cells, which is the figure's measure of hesitancy.
    """
    counts: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            train = int(row["train_fleet_size"])
            episodes = int(row["episodes"])
            robots = episodes * int(row["eval_fleet_size"])
            bucket = counts[train]
            bucket["robot_success"] += float(row["robot_success_rate"]) * robots
            bucket["robot_n"] += robots
            for name in ("collision_rate", "timeout_rate"):
                bucket[name] += float(row[name]) * episodes
            bucket["episode_n"] += episodes
            bucket["step_frac"] += float(row["mean_steps"]) / float(row["steps"])
            bucket["cells"] += 1

    out = {}
    for train, bucket in counts.items():
        out[train] = {
            "robot_success": bucket["robot_success"] / bucket["robot_n"],
            "collision_rate": bucket["collision_rate"] / bucket["episode_n"],
            "timeout_rate": bucket["timeout_rate"] / bucket["episode_n"],
            "step_frac": bucket["step_frac"] / bucket["cells"],
        }
    return out


def style(ax, xs, xlabel, ylabel, top=None) -> None:
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)
    ax.set_ylim(-0.03, top if top is not None else 1.03)
    ax.set_xlim(-0.3, len(xs) - 0.7)


def panel_title(ax, text: str, note: str) -> None:
    ax.set_title(text, fontsize=10.5, color=TEXT_PRIMARY, loc="left", pad=14)
    ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=8.5, color=TEXT_MUTED,
            ha="left", va="bottom")


def plot(policy: str, output_path: Path) -> None:
    arena = pooled(PROJECT_ROOT / SOURCES[policy]["arena"])
    ring = pooled(PROJECT_ROOT / SOURCES[policy]["ring"])
    trains = sorted(set(arena) & set(ring))
    x = np.arange(len(trains))

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6))
    fig.patch.set_facecolor(SURFACE)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # A -- the thing to be explained. The two series end far apart in both of the top
    # panels, so they take direct end-labels rather than a legend box that would have to
    # share the corner with the "best:" annotations.
    for data, color, name in ((arena, ARENA, "arena"), (ring, RING, "ring")):
        ys = [data[t]["robot_success"] for t in trains]
        ax_a.plot(x, ys, color=color, linewidth=2.4, marker="o", markersize=7,
                  markeredgecolor=SURFACE, markeredgewidth=2)
        ax_a.annotate(name, (x[-1], ys[-1]), textcoords="offset points", xytext=(9, 0),
                      ha="left", va="center", fontsize=9.5, color=color)
        best = int(np.argmax(ys))
        ax_a.annotate(f"best: {trains[best]}", (best, ys[best]), textcoords="offset points",
                      xytext=(0, 11), ha="center", fontsize=8.5, color=color)
    style(ax_a, trains, "Trained on (robots)", "Success rate (robots)",
          top=max(max(d[t]["robot_success"] for t in trains) for d in (arena, ring)) * 1.3)
    ax_a.set_xlim(-0.3, len(trains) - 0.4)
    panel_title(ax_a, "A · The reversal", "smaller fleets win in the arena, larger on the ring")

    # B -- the one property that moves the same way in both scenarios.
    for data, color, name in ((arena, ARENA, "arena"), (ring, RING, "ring")):
        ys = [data[t]["step_frac"] for t in trains]
        ax_b.plot(x, ys, color=color, linewidth=2.4, marker="s", markersize=7,
                  markeredgecolor=SURFACE, markeredgewidth=2)
        ax_b.annotate(name, (x[-1], ys[-1]), textcoords="offset points", xytext=(9, 0),
                      ha="left", va="center", fontsize=9.5, color=color)
    style(ax_b, trains, "Trained on (robots)", "Share of step budget used")
    ax_b.set_xlim(-0.3, len(trains) - 0.4)
    # Described from the data, not asserted: the MLP head gets monotonically slower with
    # the training fleet size, which is the mechanism -- but the flow head does not, because
    # its N=2 policies stall and burn the whole budget without moving, which lifts the left
    # end of this curve for a reason that is not caution. A hardcoded "hesitancy rises"
    # would be false there.
    deltas = {name: data[trains[-1]]["step_frac"] - data[trains[0]]["step_frac"]
              for data, name in ((arena, "arena"), (ring, "ring"))}
    # Rising end to end with no dip worth reading. The tolerance is there because the mlp
    # arena curve eases off by 0.005 on its last step, which is not a reversal of anything.
    monotone = all(
        deltas[name] > 0 and all(data[a]["step_frac"] - data[b]["step_frac"] < 0.03
                                 for a, b in zip(trains, trains[1:]))
        for data, name in ((arena, "arena"), (ring, "ring")))
    panel_title(ax_b, "B · Policy speed against training fleet size",
                ("slower with every step up in training fleet size, in both scenarios"
                 if monotone else
                 f"not monotone here (arena {deltas['arena']:+.2f}, ring {deltas['ring']:+.2f} "
                 "end to end)"))

    # C and D -- the same hesitancy meeting two different binding constraints.
    for ax, data, title in (
        (ax_c, arena, "C · Arena: which failure mode moves"),
        (ax_d, ring, "D · Ring: which failure mode moves"),
    ):
        collisions = [data[t]["collision_rate"] for t in trains]
        timeouts = [data[t]["timeout_rate"] for t in trains]
        # The note states the two measured end-to-end changes and names the larger one,
        # rather than asserting a direction that only holds for one head.
        d_collision = collisions[-1] - collisions[0]
        d_timeout = timeouts[-1] - timeouts[0]
        larger = "collisions" if abs(d_collision) > abs(d_timeout) else "timeouts"
        note = (f"across training fleet sizes: collisions {d_collision:+.2f}, "
                f"timeouts {d_timeout:+.2f} — {larger} move most")
        # The binding mode is drawn solid, the other dashed: which line moves is the point.
        binding_is_collision = (max(collisions) - min(collisions)) > (max(timeouts) - min(timeouts))
        ax.plot(x, collisions, color=COLLISION, linewidth=2.4, marker="o", markersize=7,
                linestyle="-" if binding_is_collision else (0, (4, 2)),
                markeredgecolor=SURFACE, markeredgewidth=2, label="collision")
        ax.plot(x, timeouts, color=TIMEOUT, linewidth=2.4, marker="^", markersize=7,
                linestyle="-" if not binding_is_collision else (0, (4, 2)),
                markeredgecolor=SURFACE, markeredgewidth=2, label="timeout")
        for ys, color in ((collisions, COLLISION), (timeouts, TIMEOUT)):
            ax.annotate(f"{ys[-1] - ys[0]:+.2f}", (x[-1], ys[-1]), textcoords="offset points",
                        xytext=(8, 0), ha="left", va="center", fontsize=8.5, color=color)
        style(ax, trains, "Trained on (robots)", "Rate (episodes)")
        ax.set_xlim(-0.3, len(trains) - 0.35)
        panel_title(ax, title, note)
        # Upper right in both: the collision line is the highest series in C and descends
        # away from that corner in D, so it is the one spot free in both panels.
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper right",
                  bbox_to_anchor=(1.0, 1.02))

    title = fig.suptitle(
        f"{policy} — why the best training fleet size reverses between scenarios",
        fontsize=13.5, color=TEXT_PRIMARY, x=0.02, ha="left", y=1.035)
    caption = ("Pooled over the three encoders and every evaluation fleet size. "
               "A larger training fleet yields a slower policy (B); that caution is wasted "
               "where collisions do not depend on it (C) and decisive where they do (D).")
    if policy == "flow":
        caption = ("Pooled over the three encoders and every evaluation fleet size. "
                   "The mlp figure's mechanism does not carry over: this head's N=2 policies "
                   "stall — burning the whole step budget without moving — which lifts the "
                   "left end of B for a reason other than caution, and inverts C and D.")
    subtitle = fig.text(0.02, 0.995, caption, fontsize=9, color=TEXT_MUTED, ha="left")

    fig.tight_layout(h_pad=3.2, w_pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=SURFACE, dpi=PNG_DPI)
    print(f"wrote {display_path(output_path)}")
    for artist in (title, subtitle):
        artist.set_visible(False)
    untitled = output_path.with_name(f"{output_path.stem}_notitle.png")
    fig.savefig(untitled, bbox_inches="tight", facecolor=SURFACE, dpi=PNG_DPI)
    print(f"wrote {display_path(untitled)}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["mlp", "flow"], default=None,
                        help="which head to draw; both when omitted")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/plots/reversal")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = parser.parse_args()

    for policy in ([args.policy] if args.policy else ["mlp", "flow"]):
        plot(policy, args.output_dir / f"reversal_{policy}.{args.format}")


if __name__ == "__main__":
    main()
