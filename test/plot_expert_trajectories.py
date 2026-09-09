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

from core.config import load_and_validate_system_config, load_yaml_config, validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from learning.dagger import apply_config_overrides
from systems.initial_state_utils import (
    normalize_goal_state_specs,
    normalize_initial_state_specs,
    parse_goal_states_argument,
    parse_initial_states_argument,
)
from systems.seed_utils import (
    action_noise_rng_for_rollout,
    default_action_noise_seed_for_config,
    default_seed_argument_for_simulator,
)
from planning.casadi_planner import PlannerSolveError
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol
from utils import plot_xy_trajectories, save_xy_rollout_video


def _extract_config_start_state(system: str, raw_config: Mapping[str, Any]) -> np.ndarray:
    """Build an initial state from config-defined starts when available."""
    if system == "multi_robot":
        robots = raw_config.get("robots")
        if not isinstance(robots, list) or len(robots) == 0:
            raise ValueError("'multi_robot' requires a non-empty 'robots' list in config.")

        starts: list[np.ndarray] = []
        for idx, robot_entry in enumerate(robots):
            if not isinstance(robot_entry, Mapping):
                raise ValueError(f"robots[{idx}] must be a mapping.")
            if "start" in robot_entry:
                starts.append(np.asarray(robot_entry["start"], dtype=np.float32))
                continue

            robot_cfg = robot_entry.get("config")
            if not isinstance(robot_cfg, Mapping) or "start" not in robot_cfg:
                raise ValueError(
                    "Config-start rollout requested, but a robot entry is missing 'start' or 'config.start'."
                )
            starts.append(np.asarray(robot_cfg["start"], dtype=np.float32))

        return np.concatenate(starts)

    if "start" not in raw_config:
        raise ValueError("Config-start rollout requested, but system config has no 'start' entry.")

    return np.asarray(raw_config["start"], dtype=np.float32)


def default_plot_output_path(system: str, planner_name: str) -> str:
    return os.path.join("outputs", "plots", f"{system}_{planner_name}_expert_rollouts.pdf")


def pairwise_distance_report(
    simulator: DynamicsProtocol,
    trajectories: list[np.ndarray],
    d_collision: float,
) -> str | None:
    state_slices = simulator.robot_state_slices
    if len(state_slices) < 2 or len(trajectories) == 0:
        return None

    # check_homogeneous_fleet_collisions/is_collision() apply one shared
    # position_indices (from the first sub-simulator) across the whole fleet,
    # so mirror that here rather than assuming a 2D position.
    position_indices = list(simulator.simulators[0].position_indices)

    min_distance = float("inf")
    worst_violation = 0.0
    violation_steps = 0
    near_boundary_steps = 0
    total_checked = 0
    tolerance = 1e-9

    for trajectory in trajectories:
        for k in range(len(trajectory)):
            for i in range(len(state_slices)):
                p_i = trajectory[k, state_slices[i]][position_indices]
                for j in range(i + 1, len(state_slices)):
                    p_j = trajectory[k, state_slices[j]][position_indices]
                    dist = float(np.linalg.norm(p_i - p_j))
                    min_distance = min(min_distance, dist)
                    total_checked += 1
                    if dist < d_collision - tolerance:
                        violation_steps += 1
                        worst_violation = max(worst_violation, d_collision - dist)
                    elif abs(dist - d_collision) <= tolerance:
                        near_boundary_steps += 1

    return (
        "Pairwise distance check: "
        f"min_dist={min_distance:.4f}, d_collision={d_collision:.4f}, "
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
    seeds: list[int] | list[list[int]] | None,
) -> list[int | list[int]]:
    if seeds is None:
        seeds = []

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


def _rng_for_seed_spec(simulator: DynamicsProtocol, seed_spec: int | list[int]) -> np.random.Generator:
    if isinstance(seed_spec, int):
        return np.random.default_rng(seed_spec)

    sub_simulators = simulator.simulators
    if len(seed_spec) != len(sub_simulators):
        raise ValueError(
            "Per-robot seed specification length must match robot count. "
            f"Got {len(seed_spec)} seeds for {len(sub_simulators)} robots."
        )

    joint_seed_seq = np.random.SeedSequence([int(robot_seed) for robot_seed in seed_spec])
    return np.random.default_rng(joint_seed_seq)


def sample_initial_state(simulator: DynamicsProtocol, seed_spec: int | list[int]) -> np.ndarray:
    rng = _rng_for_seed_spec(simulator, seed_spec)
    simulator.randomize_goal_for_reset(rng)
    return simulator.random_initial_state(rng)


def rollout_trajectory(
    simulator: DynamicsProtocol,
    planner: PlannerProtocol,
    initial_state: np.ndarray,
    num_steps: int,
    action_noise_std: float = 0.0,
    rollout_id: int | None = None,
    seed_value: Any | None = None,
    initial_state_source: str | None = None,
    action_noise_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, bool, bool]:
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
    reached_goal = simulator.should_terminate_rollout(state)
    if reached_goal:
        return np.asarray(trajectory), reached_goal, False

    planner_failed = False

    for _ in range(num_steps):
        observation = simulator.observe(state)
        try:
            action = planner(observation)
        except PlannerSolveError as exc:
            rollout_label = "?" if rollout_id is None else str(rollout_id)
            print(
                "Skipping trajectory due to planner failure "
                f"(rollout={rollout_label}, source={initial_state_source}, seed={seed_value}, "
                f"action_noise_std={action_noise_std:.6f})."
            )
            print(
                "Planner failure context: "
                f"initial_state={np.array2string(np.asarray(initial_state), precision=6)}, "
                f"current_state={np.array2string(np.asarray(state), precision=6)}, "
                f"goal_state={np.array2string(np.asarray(simulator.goal_state), precision=6)}"
            )
            print(f"Underlying solver error: {exc}")
            planner_failed = True
            break

        if action_noise_std > 0.0:
            if action_noise_rng is None:
                action_noise_rng = np.random.default_rng()
            noise = action_noise_rng.normal(
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

    return np.asarray(trajectory), reached_goal, planner_failed


def run_plotting(
    system: str,
    planner_name: str,
    config: Mapping[str, Any],
    seeds: list[int] | list[list[int]] | None,
    num_steps: int,
    initial_states: Any | None = None,
    goal_states: Any | None = None,
    tolerance_overrides: Mapping[str, float] | None = None,
    action_noise_std: float = 0.0,
    num_traj: int | None = None,
    use_config_start: bool = False,
    video: bool = True,
    video_fps: int = 12,
    output_path: str | None = None,
    raw_config: Mapping[str, Any] | None = None,
) -> str:
    if tolerance_overrides:
        config = apply_config_overrides(config, tolerance_overrides)
    validated_config = validate_system_config(system_name=system, raw_config=config)
    action_noise_seed = default_action_noise_seed_for_config(validated_config)

    simulator = DynamicsFactory.create(system_name=system, config=validated_config)
    seed_specs = normalize_seed_specs(simulator=simulator, seeds=seeds)
    initial_state_specs = normalize_initial_state_specs(
        simulator=simulator,
        initial_states=initial_states,
    )
    goal_state_specs = normalize_goal_state_specs(
        simulator=simulator,
        goal_states=goal_states,
    )
    planner = PlannerFactory.create(
        planner_name=planner_name,
        simulator=simulator,
        config=validated_config,
    )
    print(f"action noise seed: {action_noise_seed}")

    output_path = output_path or default_plot_output_path(
        system=system,
        planner_name=planner_name,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    goals_reached = 0
    failed_trajectories = 0
    trajectories: list[np.ndarray] = []
    trajectory_goal_states: list[np.ndarray] = []
    per_robot_goals_reached: np.ndarray | None = None

    if num_traj is not None and num_traj <= 0:
        raise ValueError("'num_traj' must be positive when provided.")
    if video_fps <= 0:
        raise ValueError("'video_fps' must be positive.")

    config_start_state: np.ndarray | None = None
    if use_config_start:
        if raw_config is None:
            raise ValueError("'raw_config' is required when 'use_config_start' is enabled.")
        config_start_state = simulator.validate_state(
            _extract_config_start_state(system=system, raw_config=raw_config)
        ).copy()

    explicit_initial_states: list[np.ndarray] = []
    if config_start_state is not None:
        explicit_initial_states.append(config_start_state.copy())
    explicit_initial_states.extend(
        simulator.validate_state(initial_state_spec).copy() for initial_state_spec in initial_state_specs
    )

    if num_traj is not None:
        planned_rollouts = int(num_traj)
    elif len(explicit_initial_states) > 0:
        planned_rollouts = max(len(seed_specs), len(explicit_initial_states), len(goal_state_specs))
    else:
        planned_rollouts = max(len(seed_specs), len(goal_state_specs))

    if len(explicit_initial_states) > 0:
        print(
            "simulating "
            f"{planned_rollouts} trajectories "
            f"({len(explicit_initial_states)} explicit initial states + seeded/RNG fallback)..."
        )
        initial_state_plan: list[tuple[Any, str, int | list[int] | None]] = []
        for idx in range(planned_rollouts):
            if idx < len(explicit_initial_states):
                initial_state_source = "config_start" if idx == 0 and config_start_state is not None else "provided"
                initial_state_plan.append(
                    (
                        explicit_initial_states[idx].copy(),
                        initial_state_source,
                        seed_specs[idx] if idx < len(seed_specs) else None,
                    )
                )
            elif idx < len(seed_specs):
                initial_state_plan.append((seed_specs[idx], "seeded", seed_specs[idx]))
            else:
                initial_state_plan.append((None, "rng_fallback", None))
    else:
        print(f"simulating {planned_rollouts} trajectories...")
        initial_state_plan = []
        for idx in range(planned_rollouts):
            if idx < len(seed_specs):
                initial_state_plan.append((seed_specs[idx], "seeded", seed_specs[idx]))
            else:
                initial_state_plan.append((None, "rng_fallback", None))

    baseline_goal = simulator.goal.copy()
    for rollout_idx, (initial_state_spec, initial_state_source, noise_seed_spec) in enumerate(initial_state_plan, start=1):
        explicit_goal = rollout_idx - 1 < len(goal_state_specs)
        if explicit_goal:
            simulator.set_goal(goal_state_specs[rollout_idx - 1])
        else:
            # A prior rollout's explicit goal mutates the simulator; restore the
            # config's baseline goal before any fallback sampling, since
            # randomize_goal_for_reset() is a no-op under `randomize_goal: false`
            # and would otherwise silently leak that leftover explicit goal into
            # this rollout instead of the configured/default one.
            simulator.set_goal(baseline_goal)

        if initial_state_source == "seeded":
            if explicit_goal:
                initial_state = simulator.random_initial_state(_rng_for_seed_spec(simulator, initial_state_spec))
            else:
                initial_state = sample_initial_state(simulator=simulator, seed_spec=initial_state_spec)
            seed_value = initial_state_spec
        elif initial_state_source in {"provided", "config_start"}:
            if not explicit_goal:
                goal_rng = (
                    _rng_for_seed_spec(simulator, noise_seed_spec)
                    if noise_seed_spec is not None
                    else np.random.default_rng(rollout_idx)
                )
                simulator.randomize_goal_for_reset(goal_rng)
            initial_state = simulator.validate_state(initial_state_spec).copy()
            seed_value = None
        else:
            if explicit_goal:
                initial_state = simulator.reset_random_state_only().copy()
            else:
                initial_state = simulator.reset_random().copy()
            seed_value = None

        action_noise_rng = action_noise_rng_for_rollout(
            action_noise_seed,
            seed_spec=noise_seed_spec,
            rollout_index=None if noise_seed_spec is not None else rollout_idx,
        )

        trajectory, reached_goal, planner_failed = rollout_trajectory(
            simulator=simulator,
            planner=planner,
            initial_state=initial_state,
            num_steps=num_steps,
            action_noise_std=action_noise_std,
            rollout_id=rollout_idx,
            seed_value=seed_value,
            initial_state_source=initial_state_source,
            action_noise_rng=action_noise_rng,
        )

        if planner_failed:
            failed_trajectories += 1
            continue

        trajectories.append(trajectory)
        trajectory_goal_states.append(simulator.goal_state.copy())

        if reached_goal:
            goals_reached += 1

        if system == "multi_robot":
            final_state = trajectory[-1]
            robot_reached_goals = np.asarray(
                [
                    robot.is_done(final_state[state_slice])
                    for robot, state_slice in zip(simulator.simulators, simulator.robot_state_slices)
                ],
                dtype=int,
            )
            if per_robot_goals_reached is None:
                per_robot_goals_reached = np.zeros_like(robot_reached_goals)
            per_robot_goals_reached += robot_reached_goals

    system_title = system.replace("_", " ").title()

    if len(trajectories) == 0:
        raise RuntimeError(
            "All trajectories failed due to planner infeasibility. "
            "Try lowering action noise, relaxing d_safe, or providing less adversarial initial states."
        )

    show_heading = not simulator.is_euclidean

    plot_xy_trajectories(
        simulator=simulator,
        trajectories=trajectories,
        path_to_output=output_path,
        title=f"{planner_name.replace('_', ' ').title()} optimal control paths ({system_title})",
        show_heading=show_heading,
        marker="o",
        goal_states=trajectory_goal_states,
    )

    video_path = None
    if video:
        video_path = save_xy_rollout_video(
            simulator=simulator,
            trajectories=trajectories,
            path_to_output=output_path,
            title=f"{planner_name.replace('_', ' ').title()} rollout ({system_title})",
            show_heading=show_heading,
            fps=video_fps,
            goal_states=trajectory_goal_states,
        )

    d_collision = float(validated_config.get("d_collision", validated_config.get("d_safe", 0.0)))
    distance_report = pairwise_distance_report(
        simulator=simulator,
        trajectories=trajectories,
        d_collision=d_collision,
    )

    print(f"goal reached in {goals_reached}/{len(trajectories)} successful trajectories")
    if per_robot_goals_reached is not None:
        for robot_idx, robot_count in enumerate(per_robot_goals_reached):
            print(f"robot {robot_idx}: goal reached in {int(robot_count)}/{len(trajectories)} successful trajectories")
    if failed_trajectories > 0:
        print(f"skipped {failed_trajectories}/{planned_rollouts} trajectories due to planner failure")
    print(f"plot saved to {output_path}")
    if video:
        if video_path is None:
            print("video export skipped (missing ffmpeg backend).")
        else:
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
        required=True,
        help="name of system class in lower case, e.g. single_integrator, unicycle2, ...",
    )
    parser.add_argument(
        "--planner",
        type=str.lower,
        choices=PlannerFactory.names(),
        required=True,
        help="name of planner class in lower case, e.g. casadi",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
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
        "--initial-states",
        type=str,
        default=None,
        help=(
            "explicit initial state specs. Examples: '[x, y, ...]' for one rollout, "
            "'[[...], [...]]' for multiple global states, or "
            "'[[[robot1...], [robot2...]], ...]' for multi-robot rollouts. "
            "When exhausted, plotting falls back to simulator RNG sampling."
        ),
    )
    parser.add_argument(
        "--goal-states",
        type=str,
        default=None,
        help=(
            "explicit goal state specs, independently indexed from --initial-states. "
            "Examples: '[x, y, ...]' for one rollout, '[[...], [...]]' for multiple global goals, or "
            "'[[[robot1...], [robot2...]], ...]' for multi-robot rollouts. "
            "When exhausted, plotting falls back to simulator RNG sampling."
        ),
    )
    parser.add_argument(
        "--tolerance-overrides",
        type=str,
        default=None,
        help=(
            "per-run override for the expert config's convergence tolerances, as a Python-literal "
            "dict matching the target system's tolerance keys, e.g. "
            "'{\"pos_tol\": 0.2, \"theta_tol\": 1.1, \"vel_tol\": 0.05, \"omega_tol\": 0.05}' for unicycle2, "
            "or '{\"error_tolerance\": 0.05}' for single_integrator/double_integrator/unicycle1."
        ),
    )
    parser.add_argument(
        "--num-traj",
        type=int,
        default=None,
        help=(
            "number of trajectories to attempt. When omitted, the count is inferred from --initial-states "
            "or --seeds/default seeds."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=150,
        help="maximum number of simulation steps per trajectory",
    )
    parser.add_argument(
        "--use-config-start",
        action="store_true",
        help="use config-defined start state for the first trajectory",
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
        "--video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="export MP4 alongside the PDF plot (default: enabled)",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=12,
        help="video export FPS",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="path to generated PDF plot",
    )

    args = parser.parse_args()
    if args.num_traj is not None and args.num_traj <= 0:
        parser.error("--num-traj must be positive when provided.")
    if args.num_steps <= 0:
        parser.error("--num-steps must be positive.")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive.")

    raw_config = load_yaml_config(args.config)
    config = load_and_validate_system_config(system_name=args.system, config_path=args.config)

    tolerance_overrides = None
    if args.tolerance_overrides:
        try:
            tolerance_overrides = ast.literal_eval(args.tolerance_overrides)
        except (SyntaxError, ValueError) as exc:
            parser.error(f"Unable to parse --tolerance-overrides: {exc}")
        if not isinstance(tolerance_overrides, dict):
            parser.error("--tolerance-overrides must evaluate to a dict.")

    run_plotting(
        system=args.system,
        planner_name=args.planner,
        config=config,
        seeds=parse_seed_argument(args.seeds),
        initial_states=parse_initial_states_argument(args.initial_states),
        goal_states=parse_goal_states_argument(args.goal_states),
        tolerance_overrides=tolerance_overrides,
        num_steps=args.num_steps,
        action_noise_std=args.action_noise_std,
        num_traj=args.num_traj,
        use_config_start=args.use_config_start,
        video=args.video,
        video_fps=args.video_fps,
        output_path=args.output_path,
        raw_config=raw_config,
    )


if __name__ == "__main__":
    main()
