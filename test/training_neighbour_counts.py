"""Average visible neighbours in the distribution each policy was actually TRAINED on.

The evaluation CSVs carry `mean_visible_neighbours`, but that is what a policy saw at
*evaluation* time. For "how many neighbours did this encoder ever have to aggregate during
training?" there is no recorded number, and two plausible shortcuts both give the wrong
answer:

* **The study config's own header.** `test/config/study/unicycle2_fleet_06.yaml` says
  "mean visible neighbours ~1.45", but that describes the config as written, at its own
  +-5.196 bounds. Every `data_mid` run overrode `workspace_bounds` to a 3x denser box
  (+-3.0 for N=6, see `learning/config/study/data_mid_n*/`), so the header does not
  describe what was trained.
* **`density * pi * R^2`.** The usual infinite-domain estimate gives 8.4 neighbours at
  0.1667 robots/m^2 with R = 4 m. It does not apply: a disk of radius 4 covers 50 m^2 while
  the N=8 training box is 48 m^2, so the visibility radius spans the entire workspace and
  the count is set by the fleet size and the box geometry, not by the density.

So this measures it, by sampling from the same sampler the trainer used
(`sample_initial_state` on a simulator built with the training override applied), and by
reading the explicit ring layouts the training config pins down. A third of every training
round starts from those rings, so the reported mix weights the two accordingly.

The figure is the **initial-state** count, which is policy independent. A rollout average
would depend on the policy driving it, and goals are drawn from the same box as starts, so
the spatial distribution does not drift much over an episode.

Usage:
    python test/training_neighbour_counts.py
    python test/training_neighbour_counts.py --episodes 20000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_and_validate_system_config, validate_system_config  # noqa: E402
from core.factory import DynamicsFactory  # noqa: E402
from learning.dagger import sample_initial_state  # noqa: E402
from learning.dagger.utils import apply_config_overrides  # noqa: E402
from test.evaluate_scaling import visible_neighbours  # noqa: E402

FLEET_SIZES = (2, 4, 6, 8)


def build(base_config, bounds):
    """The fleet simulator as the trainer built it: base config plus the bounds override."""
    merged = apply_config_overrides(base_config, {"workspace_bounds": list(bounds)})
    return DynamicsFactory.create("multi_robot", validate_system_config("multi_robot", merged))


def measure(fleet_size: int, episodes: int) -> dict[str, float]:
    train_config = yaml.safe_load(
        (PROJECT_ROOT / f"learning/config/study/data_mid_n{fleet_size:02d}/gnn_mlp.yaml").read_text()
    )["training"]
    bounds = train_config["workspace_bounds"]
    ring_layouts = train_config.get("initial_states") or []
    per_round = train_config["trajectories_per_iteration"][0]

    base = load_and_validate_system_config(
        "multi_robot", str(PROJECT_ROOT / f"test/config/study/unicycle2_fleet_{fleet_size:02d}.yaml")
    )
    simulator = build(base, bounds)

    uniform = float(np.mean([
        visible_neighbours(simulator, sample_initial_state(simulator, episode))
        for episode in range(episodes)
    ]))

    ring = float("nan")
    if ring_layouts:
        ring = float(np.mean([
            visible_neighbours(simulator, np.concatenate([np.asarray(r, float) for r in layout]))
            for layout in ring_layouts
        ]))
    ring_fraction = len(ring_layouts) / per_round if per_round else 0.0
    mix = (1 - ring_fraction) * uniform + ring_fraction * ring if ring_layouts else uniform

    return {
        "bound": bounds[1],
        "density": fleet_size / (bounds[1] - bounds[0]) ** 2,
        "uniform": uniform,
        "ring": ring,
        "ring_fraction": ring_fraction,
        "mix": mix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=4000,
                        help="uniform-start samples per fleet size (default 4000)")
    args = parser.parse_args()

    print(f"Visible neighbours at R = 4 m, over {args.episodes} sampled starts per fleet size.\n")
    print(f"{'train N':>7} {'box':>9} {'density':>8} {'ceiling':>8} "
          f"{'uniform':>8} {'ring':>7} {'ring %':>7} {'TRAIN MIX':>10}")
    print("-" * 72)
    for fleet_size in FLEET_SIZES:
        m = measure(fleet_size, args.episodes)
        box = "±{:g}".format(m["bound"])
        print(f"{fleet_size:>7} {box:>9} {m['density']:>8.4f} {fleet_size - 1:>8} "
              f"{m['uniform']:>8.2f} {m['ring']:>7.2f} {m['ring_fraction']:>6.0%} {m['mix']:>10.2f}")


if __name__ == "__main__":
    main()
