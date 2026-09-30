"""Generate the evaluation scenarios of the encoder study's two randomized-goal axes.

What a policy sees is the set of neighbours inside its sensing radius R. Its size is

    visible ~ min(N - 1, density * pi * R^2 * boundary factor)

and how close those neighbours come scales with 1 / sqrt(density). The two axes each
move exactly one of the two, so a curve along one of them has a single cause:

- **fleet size** (``fleet/``): N = 2..32 at the training density. Spacing is then the
  same at every N (mean nearest neighbour ~2 m), while the neighbour set grows -- 1.0
  visible at N=2, 3.5 at N=8, 5.4 at N=32 -- until it saturates at the density's
  ceiling of 8.4. It also takes the policy from seeing the whole arena (R/L = 1.15 at
  N=2) to a local window (0.29 at N=32), i.e. from effectively global observation into
  the decentralized regime.
- **density** (``density/``): a fixed N at 0.25x .. 1.25x the training density. The
  ceiling N-1 is fixed, and what changes is proximity: at N=6 the mean nearest
  neighbour goes from 5.6 m to 1.7 m, so conflicts get tighter and more frequent.
- **arena** (``arena/``): N = 2..32 in one fixed workspace, after GLAS (Riviere et al.
  2020), whose evaluation axis is robots per fixed 64 m^2. This is the only one of the
  three that keeps the *task* identical -- same box, same goal distribution, ~7 m to
  drive at every fleet size -- and varies nothing but how many robots share it. Density
  then rises with N, from 0.07x the training density at N=2 to 1.13x at N=32, so the
  sweep brackets the condition the policies trained under.

Both are expressed in multiples of the **training** density, so 1x is the condition the
policies were trained under and the in-distribution row of the density matrix.

The density axis stops at 1.25x: the sampler has to place 2N points (starts and goals)
at least d_safe apart, and at 1.33x one N=6 episode in 200 has no valid placement. Every
config is drawn here over the same 200 evaluation episodes the runs use, so a level that
cannot be placed fails now rather than aborting an evaluation run hours in.

The training configs themselves (test/config/study/unicycle2_fleet_NN.yaml) are not
touched: they define the task the expert labels, and the policy configs override the
workspace to the training density.

Usage:
    python test/config/generate_eval_configs.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import validate_system_config  # noqa: E402
from core.factory import DynamicsFactory  # noqa: E402
from learning.dagger import evaluation_seed_specs, sample_initial_state  # noqa: E402

# Loaded by path: test/config is not a package, and "test" would shadow the stdlib module.
_spec = importlib.util.spec_from_file_location(
    "_generate_fleet_configs", Path(__file__).resolve().parent / "generate_fleet_configs.py"
)
_fleet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fleet)

TEMPLATE_PATH = PROJECT_ROOT / "test/config/study/unicycle2_fleet_02.yaml"
TRAINING_DENSITY = 0.1667
TRAINING_DENSITY_REFERENCE = "learning/config/study/data_mid_n02/deepset_mlp.yaml"

FLEET_AXIS_SIZES = (2, 4, 6, 8, 16, 32)
ARENA_AXIS_SIZES = (2, 4, 6, 8, 16, 32)
ARENA_HALF_WIDTH = 6.5
DENSITY_AXIS_SIZES = (2, 6)
DENSITY_FACTORS = (0.25, 0.5, 1.0, 1.25)
CHECK_EPISODES = 200
CHECK_SEED_START = 50000


def half_width_for(num_robots: int, density: float) -> float:
    return round(math.sqrt(num_robots / density) / 2.0, 3)


def factor_label(factor: float) -> str:
    return f"{factor:g}".replace(".", "")


def check_training_density() -> None:
    """Fail if TRAINING_DENSITY no longer matches what the policy configs train at."""
    path = PROJECT_ROOT / TRAINING_DENSITY_REFERENCE
    if not path.exists():
        print(f"note: {TRAINING_DENSITY_REFERENCE} missing, cannot verify TRAINING_DENSITY")
        return
    training = yaml.safe_load(path.read_text())["training"]
    bounds = training.get("workspace_bounds")
    if bounds is None:
        raise SystemExit(f"{TRAINING_DENSITY_REFERENCE} has no workspace_bounds to check against.")
    num_robots = len(training["initial_states"][0])
    density = num_robots / (2.0 * float(bounds[1])) ** 2
    if abs(density - TRAINING_DENSITY) > 0.002:
        raise SystemExit(
            f"TRAINING_DENSITY is {TRAINING_DENSITY} but {TRAINING_DENSITY_REFERENCE} trains at "
            f"{density:.4f} robots/m^2; update one of the two."
        )


def assert_evaluation_episodes_place(validated: dict, num_robots: int, label: str) -> float:
    """Draw every evaluation episode, and return the mean visible-neighbour count."""
    simulator = DynamicsFactory.create(system_name="multi_robot", config=validated)
    visible = []
    for episode, seed_spec in enumerate(
        evaluation_seed_specs(simulator, CHECK_EPISODES, CHECK_SEED_START)
    ):
        try:
            state = simulator.reset(sample_initial_state(simulator, seed_spec))
        except RuntimeError as exc:
            raise SystemExit(
                f"{label}: evaluation episode {episode} cannot be placed ({exc}).\n"
                "Lower the density, or drop this level from DENSITY_FACTORS."
            ) from exc
        observation = simulator.observe(state, validate=False)
        visible.append(
            sum(
                observation_of(simulator, observation, robot_id)
                for robot_id in range(num_robots)
            ) / num_robots
        )
    return float(sum(visible) / len(visible))


def observation_of(simulator, observation, robot_id: int) -> float:
    return float(
        simulator.decentralized_policy_observation(observation, robot_id)[
            "observation.neighbor_mask"
        ].sum()
    )


def write_config(num_robots: int, density: float, out_path: Path, scenario: str,
                 template: dict) -> None:
    half_width = half_width_for(num_robots, density)
    config = _fleet.build_config(template, num_robots, half_width=half_width)
    validated = validate_system_config(system_name="multi_robot", raw_config=config)
    mean_visible = assert_evaluation_episodes_place(validated, num_robots, out_path.name)

    header = (
        f"# {num_robots}x unicycle2 {scenario} scenario, +-{half_width} m box, "
        f"{density / TRAINING_DENSITY:.2f}x training density, ~{mean_visible:.2f} visible neighbours.\n"
        f"# Generated by test/config/generate_eval_configs.py -- do not edit by hand.\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + _fleet.render_config(config, num_robots))
    print(f"wrote {out_path.relative_to(PROJECT_ROOT)} ({num_robots} robots, +-{half_width}, "
          f"{density / TRAINING_DENSITY:.2f}x training density, {mean_visible:.2f} visible)")


def clear_stale(directory: Path) -> None:
    """evaluate_grid.py scores every YAML in a scenario directory, so leftovers would join in."""
    for stale in directory.glob("*.yaml"):
        stale.unlink()


def main() -> None:
    check_training_density()
    template = yaml.safe_load(TEMPLATE_PATH.read_text())

    fleet_dir = PROJECT_ROOT / "test/config/study/fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    clear_stale(fleet_dir)
    print(f"fleet-size axis at the training density ({TRAINING_DENSITY} robots/m^2):")
    for num_robots in FLEET_AXIS_SIZES:
        write_config(
            num_robots, TRAINING_DENSITY,
            fleet_dir / f"unicycle2_n{num_robots:02d}.yaml",
            "fleet",
            template,
        )

    arena_dir = PROJECT_ROOT / "test/config/study/arena"
    arena_dir.mkdir(parents=True, exist_ok=True)
    clear_stale(arena_dir)
    print(f"\narena axis in one fixed +-{ARENA_HALF_WIDTH} m workspace:")
    for num_robots in ARENA_AXIS_SIZES:
        write_config(
            num_robots, num_robots / (2.0 * ARENA_HALF_WIDTH) ** 2,
            arena_dir / f"unicycle2_n{num_robots:02d}.yaml",
            "arena",
            template,
        )

    density_dir = PROJECT_ROOT / "test/config/study/density"
    density_dir.mkdir(parents=True, exist_ok=True)
    clear_stale(density_dir)
    print(f"\ndensity axis at fixed fleet sizes {DENSITY_AXIS_SIZES}:")
    for num_robots in DENSITY_AXIS_SIZES:
        for factor in DENSITY_FACTORS:
            write_config(
                num_robots, factor * TRAINING_DENSITY,
                density_dir / f"unicycle2_n{num_robots:02d}_d{factor_label(factor)}.yaml",
                "density",
                template,
            )


if __name__ == "__main__":
    main()
