import os
import sys
import torch
import ast
import argparse
import numpy as np
from typing import Any, Mapping

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from core.config import load_and_validate_system_config, validate_system_config
from core.factory import DynamicsFactory, PlannerFactory
from systems.initial_state_utils import (
    normalize_goal_state_specs,
    normalize_initial_state_specs,
    parse_goal_states_argument,
    parse_initial_states_argument,
)
from systems.seed_utils import (
    action_noise_seed_for_rollout,
    default_action_noise_seed_for_config,
    default_seed_argument_for_simulator,
)
from planning.casadi_planner import PlannerSolveError
from learning.dagger import ObservationHistoryBuffer, apply_config_overrides, build_decentralized_joint_action
from learning.models.encoder import (
    DEFAULT_ENCODER_TYPE,
    EncoderFactory,
    ObservationEncoder,
)
from learning.models.policy import ActionPolicy, PolicyFactory
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol

# Import both policy types
from utils import plot_xy_trajectories, save_xy_rollout_video


def default_evaluation_output_path(system: str, policy_type: str) -> str:
    return os.path.join("outputs", "plots", f"{system}_casadi_{policy_type}_evaluation.pdf")


def evaluation_rollout_title(system: str, policy_display_name: str) -> str:
    system_name = system.replace("_", " ").title()
    return f"{system_name} Expert Rollout vs {policy_display_name} Policy"


def is_observation_feature(feature_name: str) -> bool:
    return feature_name.startswith("observation.") or ".observation." in feature_name


def detect_collision(simulator: DynamicsProtocol, state: np.ndarray) -> tuple[bool, str]:
    """Single collision-distance pass; returns (collided, human-readable summary)."""
    details_fn = getattr(simulator, "collision_details", None)
    if not callable(details_fn):
        return bool(simulator.is_collision(state)), "collision details unavailable"
    details = details_fn(state)
    if details is None:
        return False, "collision details unavailable"
    return True, (
        f"robots=({details['robot_i']}, {details['robot_j']}), "
        f"distance={float(details['distance']):.6f}, "
        f"d_collision={float(details['threshold']):.6f}"
    )


def resolve_checkpoint_observation_dimensions(
    checkpoint: Mapping[str, Any],
    simulator: DynamicsProtocol,
    requested_policy_type: str,
) -> tuple[int, int, int, int]:
    """Resolve and validate checkpoint dimensions against the evaluation simulator."""
    features = simulator.get_dataset_features()
    runtime_state_dim = sum(
        int(feature_info["shape"][0])
        for feature_name, feature_info in features.items()
        if is_observation_feature(feature_name)
    )
    neighbor_slots = max(0, int(simulator.num_robots) - 1)
    neighbor_state_dim = int(features["observation.neighbor_state"]["shape"][0])
    runtime_neighbor_feature_dim = (
        neighbor_state_dim // neighbor_slots if neighbor_slots > 0 else 1
    )

    checkpoint_policy_type = checkpoint.get("policy_type")
    if checkpoint_policy_type is not None and str(checkpoint_policy_type).lower() != requested_policy_type:
        raise ValueError(
            f"Checkpoint was trained as '{checkpoint_policy_type}', but evaluation requested "
            f"'{requested_policy_type}'."
        )

    raw_horizon = checkpoint.get("observation_horizon", 1)
    observation_horizon = 1 if raw_horizon is None else int(raw_horizon)
    if observation_horizon <= 0:
        raise ValueError("Checkpoint 'observation_horizon' must be positive.")

    ego_base_dim = sum(
        int(features[name]["shape"][0])
        for name in ("observation.environment_state", "observation.state")
    )
    checkpoint_neighbor_slots = int(checkpoint.get("neighbor_slots", neighbor_slots))
    neighbor_feature_dim = int(
        checkpoint.get("neighbor_feature_dim", runtime_neighbor_feature_dim * observation_horizon)
    )
    expected_neighbor_feature_dim = (
        runtime_neighbor_feature_dim * observation_horizon
        if neighbor_slots > 0
        else neighbor_feature_dim
    )
    if checkpoint_neighbor_slots < 0:
        raise ValueError("Checkpoint 'neighbor_slots' must be non-negative.")
    if neighbor_feature_dim != expected_neighbor_feature_dim:
        raise ValueError(
            "Checkpoint neighbor feature schema is incompatible with the evaluation simulator: "
            f"checkpoint=(neighbor_feature_dim={neighbor_feature_dim}, observation_horizon={observation_horizon}), "
            f"expected=(neighbor_feature_dim={expected_neighbor_feature_dim}, observation_horizon={observation_horizon}). "
            "The policy can be evaluated with a different number of robots, but the per-neighbor feature schema "
            "and observation horizon must match."
        )
    expected_state_dim = (
        ego_base_dim
        + checkpoint_neighbor_slots * neighbor_feature_dim
        + checkpoint_neighbor_slots * observation_horizon
    )
    state_dim = int(checkpoint.get("state_dim", expected_state_dim))
    if state_dim != expected_state_dim:
        raise ValueError(
            "Checkpoint ego observation schema is incompatible with the evaluation simulator: "
            f"checkpoint=(state_dim={state_dim}, neighbor_slots={checkpoint_neighbor_slots}, "
            f"neighbor_feature_dim={neighbor_feature_dim}, observation_horizon={observation_horizon}), "
            f"expected_state_dim={expected_state_dim}. "
            "Use a checkpoint trained with the same ego observation schema and horizon."
        )
    return state_dim, neighbor_feature_dim, checkpoint_neighbor_slots, observation_horizon


def infer_mlp_hidden_dims_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
    linear_layers: list[tuple[int, int]] = []
    for prefix in ("network", "net"):
        linear_layers = [
            (int(key.split(".")[1]), int(value.shape[0]))
            for key, value in state_dict.items()
            if key.startswith(f"{prefix}.")
            and key.endswith(".weight")
            and value.ndim == 2
            and key.split(".")[1].isdigit()
        ]
        if linear_layers:
            break

    linear_layers.sort(key=lambda item: item[0])
    if len(linear_layers) < 2:
        return (256, 256, 128)
    return tuple(out_dim for _, out_dim in linear_layers[:-1])


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
    if seeds is None or len(seeds) == 0:
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


def _rng_for_seed_spec(
    simulator: DynamicsProtocol,
    seed_spec: int | list[int],
) -> np.random.Generator:
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


def sample_initial_state(
    simulator: DynamicsProtocol,
    seed_spec: int | list[int],
) -> np.ndarray:
    rng = _rng_for_seed_spec(simulator, seed_spec)
    simulator.randomize_goal_for_reset(rng)
    return simulator.random_initial_state(rng)


def get_inference_device():
    """
    returns the best available device for policy inference
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def apply_execution_noise(
    simulator: DynamicsProtocol,
    action: np.ndarray,
    action_noise_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if action_noise_std <= 0.0:
        return action

    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(
        loc=0.0,
        scale=action_noise_std,
        size=action.shape,
    ).astype(action.dtype, copy=False)
    return np.clip(
        action + noise,
        -simulator.max_action,
        simulator.max_action,
    )


def rollout_planner(
    simulator: DynamicsProtocol,
    planner: PlannerProtocol,
    initial_state: np.ndarray,
    num_steps: int,
    action_noise_std: float = 0.0,
    rollout_id: int | None = None,
    seed_value: Any | None = None,
    initial_state_source: str | None = None,
    action_noise_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Rolls out the expert planner from a given initial state.
    """
    state = simulator.reset(initial_state)
    planner.reset()
    trajectory = [state.copy()]

    collided, summary = detect_collision(simulator, state)
    if collided:
        print(
            "Expert rollout starts in collision "
            f"(rollout={rollout_id}, {summary})."
        )
        return np.asarray(trajectory)
    if simulator.should_terminate_rollout(state):
        return np.asarray(trajectory)

    for _ in range(num_steps):
        obs = simulator.observe(state)
        
        try:
            action = planner(obs)
        except PlannerSolveError as exc:
            rollout_label = "?" if rollout_id is None else str(rollout_id)
            print(
                "Expert planner failed during evaluation "
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
            break

        executed_action = apply_execution_noise(
            simulator=simulator,
            action=action,
            action_noise_std=action_noise_std,
            rng=action_noise_rng,
        )
        state = simulator.step(state, executed_action)
        trajectory.append(state.copy())

        collided, summary = detect_collision(simulator, state)
        if collided:
            print(
                "Expert rollout collision "
                f"(rollout={rollout_id}, step={len(trajectory) - 1}, "
                f"{summary})."
            )
            break

        if simulator.should_terminate_rollout(state):
            break

    return np.asarray(trajectory)


def rollout_policy(
    simulator: DynamicsProtocol,
    policy: ActionPolicy,
    device: torch.device,
    initial_state: np.ndarray,
    num_steps: int,
    action_noise_std: float = 0.0,
    action_noise_rng: np.random.Generator | None = None,
    observation_horizon: int = 1,
) -> tuple[np.ndarray, bool, int, bool]:
    """
    Rolls out the neural policy from a given initial state.
    
    returns:
        trajectory: array containing visited simulator states
        reached_goal: whether simulator reached goal state
        steps_taken: number of executed simulation steps
        collided: whether robots collided during the rollout
    """
    state = simulator.reset(initial_state)
    trajectory = [state.copy()]
    policy.reset()
    history_buffer = ObservationHistoryBuffer(observation_horizon, int(simulator.num_robots))
    collided, summary = detect_collision(simulator, state)

    if collided:
        print(
            "Policy rollout starts in collision "
            f"({summary})."
        )
        return np.asarray(trajectory), False, 0, True
    if simulator.should_terminate_rollout(state):
        return np.asarray(trajectory), True, 0, False

    for step in range(1, num_steps + 1):
        observation = simulator.observe(state)
        action = build_decentralized_joint_action(
            simulator=simulator,
            policy=policy,
            observation=observation,
            device=device,
            observation_horizon=observation_horizon,
            history_buffer=history_buffer,
        )
        executed_action = apply_execution_noise(
            simulator=simulator,
            action=action,
            action_noise_std=action_noise_std,
            rng=action_noise_rng,
        )

        state = simulator.step(state, executed_action)
        trajectory.append(state.copy())

        collision_now, summary = detect_collision(simulator, state)
        if collision_now:
            collided = True
            print(
                "Policy rollout collision "
                f"(step={step}, {summary})."
            )
            break

        if simulator.should_terminate_rollout(state):
            return np.asarray(trajectory), True, step, collided

    return np.asarray(trajectory), False, len(trajectory) - 1, collided


def _load_checkpoint_policy_components(
    model_dir: str,
    simulator: DynamicsProtocol,
    policy_type: str,
    device: torch.device,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, torch.Tensor],
    ObservationEncoder,
    int,
    tuple[int, ...],
    int,
    int,
]:
    """Load a metadata checkpoint and resolve its encoder/action/hidden-dim schema.

    Returns (checkpoint, state_dict, obs_encoder, action_dim, hidden_dims,
    prediction_horizon, observation_horizon).
    """
    checkpoint = torch.load(model_dir, map_location=device)
    if not (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
        raise ValueError(
            f"'{policy_type}' models require a metadata checkpoint to infer the prediction_horizon "
            "and cannot be loaded from raw state dictionaries."
        )
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"'{policy_type}' checkpoint must contain a state dictionary.")

    hidden_dims = infer_mlp_hidden_dims_from_state_dict(state_dict)
    prediction_horizon = int(checkpoint.get("prediction_horizon", 1))

    state_dim, neighbor_feature_dim, neighbor_slots, observation_horizon = resolve_checkpoint_observation_dimensions(
        checkpoint, simulator, policy_type
    )
    action_dim = int(checkpoint.get("action_dim", int(simulator.nu)))
    hidden_dims_raw = checkpoint.get("hidden_dims")
    if isinstance(hidden_dims_raw, list) and hidden_dims_raw:
        hidden_dims = tuple(int(width) for width in hidden_dims_raw)

    encoder_type = str(checkpoint.get("encoder_type", DEFAULT_ENCODER_TYPE))
    encoder_kwargs_raw = checkpoint.get("encoder_kwargs")
    encoder_kwargs = dict(encoder_kwargs_raw) if isinstance(encoder_kwargs_raw, Mapping) else {}
    obs_encoder: ObservationEncoder = EncoderFactory.create(
        encoder_type=encoder_type,
        state_dim=state_dim,
        neighbor_feature_dim=neighbor_feature_dim,
        neighbor_slots=neighbor_slots,
        observation_horizon=observation_horizon,
        **encoder_kwargs,
    )
    return (
        checkpoint,
        state_dict,
        obs_encoder,
        action_dim,
        hidden_dims,
        prediction_horizon,
        observation_horizon,
    )


def run_evaluation(
    system: str,
    policy_type: str,
    config: Mapping[str, Any],
    model_dir: str,
    num_steps: int = 150,
    seeds: list[int] | list[list[int]] | None = None,
    initial_states: Any | None = None,
    goal_states: Any | None = None,
    tolerance_overrides: Mapping[str, float] | None = None,
    action_noise_std: float = 0.0,
    output_path: str | None = None,
):
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

    if not os.path.exists(model_dir):
        print(f"assuming '{model_dir}' is a Hugging Face Hub ID")

    device = get_inference_device()
    print(f"running inference on {device}")
    print(f"action noise seed: {action_noise_seed}")

    checkpoint, state_dict, obs_encoder, action_dim, hidden_dims, prediction_horizon, observation_horizon = (
        _load_checkpoint_policy_components(model_dir, simulator, policy_type, device)
    )
    policy_kwargs: dict[str, object] = {
        "action_dim": action_dim,
        "obs_encoder": obs_encoder,
        "hidden_dims": hidden_dims,
        "prediction_horizon": prediction_horizon,
    }
    if policy_type == "flow":
        num_inference_steps = 10
        flow_config_raw = checkpoint.get("flow_config")
        if isinstance(flow_config_raw, Mapping):
            num_inference_steps = int(flow_config_raw.get("num_inference_steps", 10))
        policy_kwargs["num_inference_steps"] = num_inference_steps

    policy = PolicyFactory.create(policy_type, **policy_kwargs)
    policy.load_state_dict(state_dict)
    policy_display_name = policy_type.title()

    policy.eval()
    policy.to(device)

    # Instantiate expert planner once and reset it for each rollout.
    expert_planner = PlannerFactory.create(planner_name="casadi", simulator=simulator, config=validated_config)
    expert_trajectories: list[np.ndarray] = []
    policy_trajectories: list[np.ndarray] = []
    per_seed_metrics: list[dict[str, Any]] = []

    if len(initial_state_specs) > 0:
        total_rollouts = max(len(seed_specs), len(initial_state_specs), len(goal_state_specs))
        print(
            "evaluating "
            f"{total_rollouts} trajectories "
            f"({len(initial_state_specs)} explicit initial states + seeded/RNG fallback)"
        )
        rollout_plan: list[tuple[Any, str, int, Any, int | list[int] | None]] = []
        for rollout_idx in range(total_rollouts):
            if rollout_idx < len(initial_state_specs):
                initial_state_spec: Any = simulator.validate_state(initial_state_specs[rollout_idx]).copy()
                initial_state_source = "provided"
                seed_value: Any = None
                noise_seed_spec: int | list[int] | None = None
                if rollout_idx < len(seed_specs):
                    seed_spec = seed_specs[rollout_idx]
                    torch_seed = int(seed_spec) if isinstance(seed_spec, int) else int(seed_spec[0])
                    noise_seed_spec = seed_spec
                else:
                    torch_seed = rollout_idx + 1
            elif rollout_idx < len(seed_specs):
                initial_state_spec = seed_specs[rollout_idx]
                initial_state_source = "seeded"
                seed_spec = seed_specs[rollout_idx]
                torch_seed = int(seed_spec) if isinstance(seed_spec, int) else int(seed_spec[0])
                seed_value = seed_spec
                noise_seed_spec = seed_spec
            else:
                initial_state_spec = None
                initial_state_source = "rng_fallback"
                torch_seed = rollout_idx + 1
                seed_value = None
                noise_seed_spec = None

            rollout_plan.append((initial_state_spec, initial_state_source, torch_seed, seed_value, noise_seed_spec))
    else:
        total_rollouts = max(len(seed_specs), len(goal_state_specs))
        print(f"evaluating {total_rollouts} trajectories ({len(seed_specs)} seeded + goal/RNG fallback)")
        rollout_plan = []
        for rollout_idx in range(total_rollouts):
            if rollout_idx < len(seed_specs):
                seed_spec = seed_specs[rollout_idx]
                torch_seed = int(seed_spec) if isinstance(seed_spec, int) else int(seed_spec[0])
                rollout_plan.append((seed_spec, "seeded", torch_seed, seed_spec, seed_spec))
            else:
                rollout_plan.append((None, "rng_fallback", rollout_idx + 1, None, None))

    baseline_goal = simulator.goal.copy()
    for rollout_idx, (rollout_spec, initial_state_source, torch_seed, seed_value, noise_seed_spec) in enumerate(rollout_plan, start=1):
        torch.manual_seed(torch_seed)

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
            seed_spec = rollout_spec
            if explicit_goal:
                initial_state = simulator.random_initial_state(_rng_for_seed_spec(simulator, seed_spec))
            else:
                initial_state = sample_initial_state(simulator=simulator, seed_spec=seed_spec)
        elif initial_state_source == "provided":
            if not explicit_goal:
                simulator.randomize_goal_for_reset(np.random.default_rng(torch_seed))
            initial_state = simulator.validate_state(rollout_spec).copy()
        else:
            if explicit_goal:
                initial_state = simulator.reset_random_state_only().copy()
            else:
                initial_state = simulator.reset_random().copy()

        if noise_seed_spec is not None:
            rollout_noise_seed = action_noise_seed_for_rollout(
                action_noise_seed,
                seed_spec=noise_seed_spec,
            )
        else:
            rollout_noise_seed = action_noise_seed_for_rollout(
                action_noise_seed,
                rollout_index=rollout_idx,
            )

        expert_action_noise_rng = np.random.default_rng(rollout_noise_seed)
        policy_action_noise_rng = np.random.default_rng(rollout_noise_seed)

        goal_state = simulator.goal_state.copy()

        expert_trajectory = rollout_planner(
            simulator=simulator,
            planner=expert_planner,
            initial_state=initial_state,
            num_steps=num_steps,
            action_noise_std=action_noise_std,
            rollout_id=rollout_idx,
            seed_value=seed_value,
            initial_state_source=initial_state_source,
            action_noise_rng=expert_action_noise_rng,
        )

        policy_trajectory, reached_goal, steps_taken, policy_collided = rollout_policy(
            simulator=simulator,
            policy=policy,
            device=device,
            initial_state=initial_state,
            num_steps=num_steps,
            action_noise_std=action_noise_std,
            action_noise_rng=policy_action_noise_rng,
            observation_horizon=observation_horizon,
        )
        expert_collided = simulator.is_collision(expert_trajectory[-1])

        policy_final_state = policy_trajectory[-1]
        expert_final_state = expert_trajectory[-1]
        policy_goal_error = float(np.linalg.norm(policy_final_state - goal_state))
        expert_goal_error = float(np.linalg.norm(expert_final_state - goal_state))

        expert_trajectories.append(expert_trajectory)
        policy_trajectories.append(policy_trajectory)
        per_seed_metrics.append(
            {
                "seed": seed_value,
                "initial_state_source": initial_state_source,
                "goal_state": goal_state,
                "initial_state": initial_state,
                "policy_reached_goal": reached_goal,
                "policy_collided": policy_collided,
                "expert_collided": expert_collided,
                "policy_steps": max(len(policy_trajectory) - 1, 0),
                "expert_steps": max(len(expert_trajectory) - 1, 0),
                "policy_goal_error_l2": policy_goal_error,
                "expert_goal_error_l2": expert_goal_error,
            }
        )

    total_runs = len(per_seed_metrics)
    total_successes = sum(
        1
        for metric in per_seed_metrics
        if metric["policy_reached_goal"] and not metric["policy_collided"]
    )
    success_rate = (total_successes / total_runs) if total_runs > 0 else 0.0
    mean_policy_error = float(
        np.mean([metric["policy_goal_error_l2"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0
    mean_expert_error = float(
        np.mean([metric["expert_goal_error_l2"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0
    mean_policy_steps = float(
        np.mean([metric["policy_steps"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0
    mean_expert_steps = float(
        np.mean([metric["expert_steps"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0

    print("\n--- Evaluation Summary ---")
    print(f"system: {system}")
    print(f"policy_type: {policy_type}")
    print(f"seeds: {seed_specs}")
    print(f"device: {device}")
    print(f"action_noise_std: {action_noise_std:.6f}")
    unique_goal_states = {
        tuple(np.asarray(metric["goal_state"], dtype=float).tolist())
        for metric in per_seed_metrics
    }
    summary_goal_state: list[float] | None = None
    if len(unique_goal_states) == 1:
        only_goal_state = np.asarray(next(iter(unique_goal_states)), dtype=float)
        summary_goal_state = only_goal_state.tolist()
        print(f"goal_state: {np.array2string(only_goal_state, precision=4)}")
    else:
        print("goal_state: varies per seed (goal randomization enabled)")
    print(f"num_trajectories: {total_runs}")
    print(f"policy_successes: {total_successes}/{total_runs}")
    print(f"success_rate: {success_rate:.4f}")
    policy_collision_rate = (
        np.mean([metric["policy_collided"] for metric in per_seed_metrics])
        if total_runs > 0 else 0.0
    )
    expert_collision_rate = (
        np.mean([metric["expert_collided"] for metric in per_seed_metrics])
        if total_runs > 0 else 0.0
    )
    print(f"policy_collision_rate: {policy_collision_rate:.4f}")
    print(f"expert_collision_rate: {expert_collision_rate:.4f}")
    print(f"mean_policy_steps: {mean_policy_steps:.3f}")
    print(f"mean_expert_steps: {mean_expert_steps:.3f}")
    print(f"mean_policy_goal_error_l2: {mean_policy_error:.6f}")
    print(f"mean_expert_goal_error_l2: {mean_expert_error:.6f}")

    # Dynamically set output names
    output_path = output_path or default_evaluation_output_path(
        system=system,
        policy_type=policy_type,
    )

    comparison_title = evaluation_rollout_title(system, policy_display_name)

    all_trajectories = expert_trajectories + policy_trajectories
    num_expert = len(expert_trajectories)
    num_policy = len(policy_trajectories)
    path_labels = [None] * (num_expert + num_policy)
    if num_expert > 0:
        path_labels[0] = "Expert"
    if num_policy > 0:
        path_labels[num_expert] = f"{policy_display_name} Policy"
    trajectory_colors = ["tab:blue"] * num_expert + ["tab:orange"] * num_policy
    trajectory_line_styles = ["--"] * num_expert + ["-"] * num_policy

    show_heading = not simulator.is_euclidean

    plot_xy_trajectories(
        simulator=simulator,
        trajectories=all_trajectories,
        path_to_output=output_path,
        title=comparison_title,
        path_labels=path_labels,
        show_heading=show_heading,
        marker="o",
        trajectory_colors=trajectory_colors,
        trajectory_line_styles=trajectory_line_styles,
    )
    print(f"Plot saved to {output_path}")

    video_path = save_xy_rollout_video(
        simulator=simulator,
        trajectories=all_trajectories,
        path_to_output=output_path,
        title=comparison_title,
        show_heading=show_heading,
        fps=12,
        path_labels=path_labels,
        trajectory_colors=trajectory_colors,
        trajectory_line_styles=trajectory_line_styles,
        phase_lengths=[num_expert, num_policy],
    )
    if video_path is not None:
        print(f"Video saved to {video_path}")

    metrics = {
        "system": system,
        "policy_type": policy_type,
        "seeds": seed_specs,
        "device": str(device),
        "action_noise_std": action_noise_std,
        "goal_state": summary_goal_state,
        "goal_state_varies_by_seed": len(unique_goal_states) > 1,
        "num_trajectories": total_runs,
        "policy_successes": total_successes,
        "success_rate": success_rate,
        "policy_collision_rate": float(policy_collision_rate),
        "expert_collision_rate": float(expert_collision_rate),
        "mean_policy_steps": mean_policy_steps,
        "mean_expert_steps": mean_expert_steps,
        "mean_policy_goal_error_l2": mean_policy_error,
        "mean_expert_goal_error_l2": mean_expert_error,
        "per_seed": per_seed_metrics,
        "plot_path": output_path,
        "video_path": video_path,
    }

    return metrics


def main():
    """
    evaluates a trained policy for a selected dynamics system
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        type=str.lower,
        choices=DynamicsFactory.names(),
        required=True,
        help="name of system class, e.g. single_integrator, unicycle2, ...",
    )
    parser.add_argument(
        "--policy-type",
        type=str.lower,
        choices=["mlp", "flow"],
        required=True,
        help="the type of policy architecture to evaluate",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to yaml config file for experiment",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="path to a local checkpoint or Hugging Face Hub model ID",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=150,
        help="maximum number of simulation steps",
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
            "When exhausted, evaluation falls back to simulator RNG sampling."
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
            "When exhausted, evaluation falls back to simulator RNG sampling."
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
        "--action-noise-std",
        type=float,
        default=0.0,
        help=(
            "std-dev of Gaussian action noise applied during expert and policy rollout execution; "
            "default 0.0 keeps evaluation deterministic and clean"
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

    tolerance_overrides = None
    if args.tolerance_overrides:
        try:
            tolerance_overrides = ast.literal_eval(args.tolerance_overrides)
        except (SyntaxError, ValueError) as exc:
            parser.error(f"Unable to parse --tolerance-overrides: {exc}")
        if not isinstance(tolerance_overrides, dict):
            parser.error("--tolerance-overrides must evaluate to a dict.")

    run_evaluation(
        system=args.system,
        policy_type=args.policy_type,
        config=config,
        model_dir=args.model_dir,
        num_steps=args.num_steps,
        seeds=parse_seed_argument(args.seeds),
        initial_states=parse_initial_states_argument(args.initial_states),
        goal_states=parse_goal_states_argument(args.goal_states),
        tolerance_overrides=tolerance_overrides,
        action_noise_std=args.action_noise_std,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()