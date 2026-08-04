import os
import sys
import ast
import argparse
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config, validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from core.seed_utils import default_seed_argument_for_simulator
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol
from utils import plot_xy_trajectories, save_xy_rollout_video


def default_plot_output_path(system: str, planner_name: str) -> str:
    return os.path.join("outputs", "plots", f"{system}_{planner_name}_expert_rollouts.pdf")


def pairwise_distance_report(
    simulator: DynamicsProtocol,
    trajectories: list[np.ndarray],
    d_safe: float,
) -> str | None:
    state_slices = simulator.robot_state_slices
    if len(state_slices) < 2 or len(trajectories) == 0:
        return None

    min_distance = float("inf")
    worst_violation = 0.0
    violation_steps = 0
    near_boundary_steps = 0
    total_checked = 0
    tolerance = 1e-9

    for trajectory in trajectories:
        for k in range(len(trajectory)):
            for i in range(len(state_slices)):
                p_i = trajectory[k, state_slices[i]][:2]
                for j in range(i + 1, len(state_slices)):
                    p_j = trajectory[k, state_slices[j]][:2]
                    dist = float(np.linalg.norm(p_i - p_j))
                    min_distance = min(min_distance, dist)
                    total_checked += 1
                    if dist < d_safe - tolerance:
                        violation_steps += 1
                        worst_violation = max(worst_violation, d_safe - dist)
                    elif abs(dist - d_safe) <= tolerance:
                        near_boundary_steps += 1

    return (
        "Pairwise distance check: "
        f"min_dist={min_distance:.4f}, d_safe={d_safe:.4f}, "
        f"violations={violation_steps}/{total_checked}, "
        f"near_boundary={near_boundary_steps}/{total_checked}, "
        f"max_shortfall={worst_violation:.4f}"
    )


def parse_seed_argument(raw_seeds: str | None) -> list[int] | list[list[int]]:
    if raw_seeds is None:
        return []

    text = raw_seeds.strip()
    if text == "":
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parts = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
        if len(parts) > 1:
            return [int(part) for part in parts]
        return [int(text)]

    if isinstance(parsed, int):
        return [parsed]
    if isinstance(parsed, list):
        if len(parsed) == 0:
            return []
        if all(isinstance(x, int) for x in parsed):
            return [int(x) for x in parsed]
        if all(isinstance(x, list) for x in parsed):
            nested: list[list[int]] = []
            for robot_idx, robot_seeds in enumerate(parsed):
                if not all(isinstance(seed, int) for seed in robot_seeds):
                    raise ValueError(
                        f"All nested seed entries must be integers. Invalid entry in robot {robot_idx}."
                    )
                nested.append([int(seed) for seed in robot_seeds])
            return nested

    raise ValueError(
        "Unable to parse seeds. Use an int (e.g. '42'), a list (e.g. '[42, 7]'), "
        "or nested per-robot lists (e.g. '[[10, 4], [21, 0]]')."
    )


def normalize_seed_specs(
    simulator: DynamicsProtocol,
    seeds: list[int] | list[list[int]],
) -> list[int | list[int]]:
    if len(seeds) == 0:
        seeds = default_seed_argument_for_simulator(simulator)

    if isinstance(seeds[0], list):
        seed_lists = seeds  # type: ignore[assignment]
        expected_len = len(seed_lists[0])
        for robot_idx, robot_seeds in enumerate(seed_lists[1:], start=1):
            if len(robot_seeds) != expected_len:
                raise ValueError(
                    "Seed list length mismatch across robots. "
                    f"Robot 0 has {expected_len} seeds, but Robot {robot_idx} has {len(robot_seeds)} seeds. "
                    "The number of seeds per robot must match to pair initial conditions into joint rollout scenarios."
                )
        return [list(seed_tuple) for seed_tuple in zip(*seed_lists)]

    return [int(seed) for seed in seeds]  # type: ignore[arg-type]


def sample_initial_state(simulator: DynamicsProtocol, seed_spec: int | list[int]) -> np.ndarray:
    if isinstance(seed_spec, int):
        rng = np.random.default_rng(seed_spec)
        simulator.randomize_goal_for_reset(rng)
        return simulator.random_initial_state(rng)

    sub_states = []
    sub_simulators = simulator.simulators
    if len(seed_spec) != len(sub_simulators):
        raise ValueError(
            "Per-robot seed specification length must match robot count. "
            f"Got {len(seed_spec)} seeds for {len(sub_simulators)} robots."
        )

    for robot_seed, sub_sim in zip(seed_spec, sub_simulators):
        rng = np.random.default_rng(int(robot_seed))
        sub_sim.randomize_goal_for_reset(rng)
        sub_states.append(sub_sim.random_initial_state(rng))

    return np.concatenate(sub_states)


def rollout_trajectory(
    simulator: DynamicsProtocol,
    planner: PlannerProtocol,
    initial_state: np.ndarray,
    num_steps: int,
    action_noise_std: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """
    simulates one trajectory controlled by a planner

    args:
        simulator: dynamics simulator used to generate states and observations
        planner: motion planner that selects an action from each observation
        num_steps: maximum number of simulation steps.

    returns:
        trajectory: array of visited states
        reached_goal: whether simulator reached its goal condition
    """

    state = simulator.reset(initial_state)
    planner.reset()

    trajectory = [state.copy()]
    reached_goal = False

    for _ in range(num_steps):
        observation = simulator.observe(state)
        action = planner(observation)

        if action_noise_std > 0.0:
            noise = np.random.normal(
                loc=0.0,
                scale=action_noise_std,
                size=action.shape,
            ).astype(action.dtype, copy=False)
            executed_action = np.clip(
                action + noise,
                -simulator.max_action,
                simulator.max_action,
            )
        else:
            executed_action = action

        state = simulator.step(state, executed_action)
        trajectory.append(state.copy())
        if simulator.should_terminate_rollout(state):
            reached_goal = True
            break

    return np.asarray(trajectory), reached_goal


def run_plotting(
    system: str,
    planner_name: str,
    config: Mapping[str, Any],
    seeds: list[int] | list[list[int]],
    num_steps: int,
    action_noise_std: float = 0.0,
    output_path: str | None = None,
) -> str:
    validated_config = validate_system_config(system_name=system, raw_config=config)

    simulator = DynamicsFactory.create(system_name=system, config=validated_config)
    seed_specs = normalize_seed_specs(simulator=simulator, seeds=seeds)
    planner = PlannerFactory.create(
        planner_name=planner_name,
        simulator=simulator,
        config=validated_config,
    )

    output_path = output_path or os.path.join(
        default_plot_output_path(system=system, planner_name=planner_name),
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_traj = len(seed_specs)
    print(f"simulating {num_traj} randomized trajectories...")

    goals_reached = 0
    trajectories: list[np.ndarray] = []
    for seed_spec in seed_specs:
        initial_state = sample_initial_state(simulator=simulator, seed_spec=seed_spec)
        trajectory, reached_goal = rollout_trajectory(
            simulator=simulator,
            planner=planner,
            initial_state=initial_state,
            num_steps=num_steps,
            action_noise_std=action_noise_std,
        )

        trajectories.append(trajectory)

        if reached_goal:
            goals_reached += 1

    system_title = system.replace("_", " ").title()

    show_heading = simulator.has_heading

    plot_xy_trajectories(
        simulator=simulator,
        trajectories=trajectories,
        path_to_output=output_path,
        title=f"{planner_name.replace('_', ' ').title()} optimal control paths ({system_title})",
        show_heading=show_heading,
        marker="o",
    )

    video_path = save_xy_rollout_video(
        simulator=simulator,
        trajectories=trajectories,
        path_to_output=output_path,
        title=f"{planner_name.replace('_', ' ').title()} rollout ({system_title})",
        show_heading=show_heading,
        fps=12,
    )

    d_safe = float(validated_config.get("d_safe", 0.0))
    distance_report = pairwise_distance_report(
        simulator=simulator,
        trajectories=trajectories,
        d_safe=d_safe,
    )

    print(f"goal reached in {goals_reached}/{num_traj} trajectories")
    print(f"plot saved to {output_path}")
    if video_path is not None:
        print(f"video saved to {video_path}")
    if distance_report is not None:
        print(distance_report)
    return output_path


def main():
    """
    plots planner-controlled trajectories for a selected dynamics system
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        type=str.lower,
        choices=DynamicsFactory.names(),
        help="name of system class in lower case, e.g. single_integrator, unicycle2, ...",
    )
    parser.add_argument(
        "--planner",
        type=str.lower,
        choices=PlannerFactory.names(),
        help="name of planner class in lower case, e.g. casadi",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to yaml config file for experiment",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "seed specification: int ('1'), list ('[1, 7]'), or nested per-robot lists "
            "('[[10, 4, 2], [21, 0, 9]]')."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=150,
        help="maximum number of simulation steps per trajectory",
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.0,
        help=(
            "std-dev of Gaussian action noise applied during rollout execution; "
            "default 0.0 keeps expert plotting deterministic"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="path to generated PDF plot",
    )

    args = parser.parse_args()
    config = load_and_validate_system_config(system_name=args.system, config_path=args.config)

    run_plotting(
        system=args.system,
        planner_name=args.planner,
        config=config,
        seeds=parse_seed_argument(args.seeds),
        num_steps=args.num_steps,
        action_noise_std=args.action_noise_std,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
