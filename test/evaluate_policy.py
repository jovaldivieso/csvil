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
    normalize_initial_state_specs,
    parse_initial_states_argument,
)
from systems.seed_utils import (
    action_noise_seed_for_rollout,
    default_action_noise_seed_for_config,
    default_seed_argument_for_simulator,
)
from planning.casadi_planner import PlannerSolveError
from learning.dagger import (
    build_decentralized_joint_action,
    build_observation_feature_pack_cache,
    uses_decentralized_policy,
)
from learning.models.flow_policy import FlowPolicy
from learning.models.mlp_policy import MLPPolicy
from learning.models.encoder import (
    DEFAULT_ENCODER_TYPE,
    EncoderFactory,
    ObservationEncoder,
)
from planning.planner import PlannerProtocol
from systems.dynamics import DynamicsProtocol

# Import both policy types
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

from utils import plot_xy_trajectories, save_xy_rollout_video


def default_evaluation_output_path(system: str, policy_type: str) -> str:
    return os.path.join("outputs", "plots", f"{system}_casadi_{policy_type}_evaluation.pdf")


def evaluation_rollout_title(system: str, policy_display_name: str) -> str:
    system_name = system.replace("_", " ").title()
    return f"{system_name} Expert Rollout vs {policy_display_name} Policy"


def is_observation_feature(feature_name: str) -> bool:
    return feature_name.startswith("observation.") or ".observation." in feature_name


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


def infer_mlp_dimensions_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    linear_weights: list[torch.Tensor] = []
    for prefix in ("network", "net"):
        linear_weights = [
            value
            for key, value in state_dict.items()
            if key.startswith(f"{prefix}.") and key.endswith(".weight") and value.ndim == 2
        ]
        if linear_weights:
            break

    if not linear_weights:
        raise ValueError("Checkpoint does not contain expected policy linear weights.")
    return int(linear_weights[0].shape[1]), int(linear_weights[-1].shape[0])


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


def sample_initial_state(
    simulator: DynamicsProtocol,
    seed_spec: int | list[int],
) -> np.ndarray:
    if isinstance(seed_spec, int):
        rng = np.random.default_rng(seed_spec)
        simulator.randomize_goal_for_reset(rng)
        return simulator.random_initial_state(rng)

    sub_simulators = simulator.simulators
    if len(seed_spec) != len(sub_simulators):
        raise ValueError(
            "Per-robot seed specification length must match robot count. "
            f"Got {len(seed_spec)} seeds for {len(sub_simulators)} robots."
        )

    joint_seed_seq = np.random.SeedSequence([int(robot_seed) for robot_seed in seed_spec])
    rng = np.random.default_rng(joint_seed_seq)
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


def create_policy_input(
    simulator: DynamicsProtocol,
    observation: np.ndarray,
    device: torch.device,
    add_batch_dim: bool = True,
    observation_feature_cache=None,
) -> dict[str, torch.Tensor]:
    """
    Dynamically slices the flat observation array based on the
    feature shapes defined by the simulator.
    """
    from learning.dagger import (  # local import keeps this helper self-contained
        build_observation_feature_pack_cache,
        pack_observation_features_from_cache,
    )

    if observation_feature_cache is None:
        feature_names = [
            feature_name
            for feature_name in simulator.get_dataset_features().keys()
            if is_observation_feature(feature_name)
        ]
        observation_feature_cache = build_observation_feature_pack_cache(simulator, feature_names)

    packed_features = pack_observation_features_from_cache(observation, observation_feature_cache)
    policy_input: dict[str, torch.Tensor] = {}
    for feature_name in observation_feature_cache.feature_names:
        feature_tensor = torch.as_tensor(packed_features[feature_name], dtype=torch.float32)
        if add_batch_dim:
            feature_tensor = feature_tensor.view(1, -1)
        policy_input[feature_name] = feature_tensor

    if device.type != "cpu":
        policy_input = {feature_name: tensor.to(device) for feature_name, tensor in policy_input.items()}

    return policy_input


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

    if simulator.should_terminate_rollout(state):
        return np.asarray(trajectory)

    for _ in range(num_steps):
        obs = simulator.observe(state, validate=False)
        
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
        state = simulator.step(state, executed_action, validate=False)
        trajectory.append(state.copy())

        if simulator.is_collision(state):
            break

        if simulator.should_terminate_rollout(state):
            break

    return np.asarray(trajectory)


def rollout_policy(
    simulator: DynamicsProtocol,
    policy: DiffusionPolicy | ACTPolicy | MLPPolicy | FlowPolicy,
    device: torch.device,
    initial_state: np.ndarray,
    num_steps: int,
    action_noise_std: float = 0.0,
    action_noise_rng: np.random.Generator | None = None,
    policy_preprocessor=None,
    policy_postprocessor=None,
    observation_feature_cache=None,
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
    collided = False

    decentralized_inference = uses_decentralized_policy(simulator, policy)

    for step in range(1, num_steps + 1):
        observation = simulator.observe(state, validate=False)

        if decentralized_inference:
            action = build_decentralized_joint_action(
                simulator=simulator,
                policy=policy,
                observation=observation,
                device=device,
            )
        else:
            policy_input = create_policy_input(
                simulator=simulator,
                observation=observation,
                device=device,
                add_batch_dim=policy_preprocessor is None,
                observation_feature_cache=observation_feature_cache,
            )
            if policy_preprocessor is not None:
                policy_input = policy_preprocessor(policy_input)

            with torch.inference_mode():
                action_tensor = policy.select_action(policy_input)
            if policy_postprocessor is not None:
                action_tensor = policy_postprocessor(action_tensor)

            if isinstance(policy, (MLPPolicy, FlowPolicy)):
                action = action_tensor.squeeze(0)[0].cpu().numpy()
            else:
                action = action_tensor.squeeze(0).cpu().numpy()

        executed_action = apply_execution_noise(
            simulator=simulator,
            action=action,
            action_noise_std=action_noise_std,
            rng=action_noise_rng,
        )

        state = simulator.step(state, executed_action, validate=False)
        trajectory.append(state.copy())

        if simulator.is_collision(state):
            collided = True
            break

        if simulator.should_terminate_rollout(state):
            return np.asarray(trajectory), True, step, collided

    return np.asarray(trajectory), False, len(trajectory) - 1, collided


def run_evaluation(
    system: str,
    policy_type: str,
    config: Mapping[str, Any],
    model_dir: str,
    num_steps: int = 150,
    seeds: list[int] | list[list[int]] | None = None,
    initial_states: Any | None = None,
    action_noise_std: float = 0.0,
    output_path: str | None = None,
):
    validated_config = validate_system_config(system_name=system, raw_config=config)
    action_noise_seed = default_action_noise_seed_for_config(validated_config)

    simulator = DynamicsFactory.create(system_name=system, config=validated_config)
    seed_specs = normalize_seed_specs(simulator=simulator, seeds=seeds)
    initial_state_specs = normalize_initial_state_specs(
        simulator=simulator,
        initial_states=initial_states,
    )

    if not os.path.exists(model_dir):
        print(f"assuming '{model_dir}' is a Hugging Face Hub ID")

    device = get_inference_device()
    print(f"running inference on {device}")
    print(f"action noise seed: {action_noise_seed}")

    # Dynamically load the requested policy
    policy_preprocessor = None
    policy_postprocessor = None
    if policy_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(model_dir)
        policy_display_name = "Diffusion"
    elif policy_type == "act":
        policy = ACTPolicy.from_pretrained(model_dir)
        policy_display_name = "ACT"
    elif policy_type == "flow":
        state_dim = sum(
            int(feature_info["shape"][0])
            for feature_name, feature_info in simulator.get_dataset_features().items()
            if is_observation_feature(feature_name)
        )
        action_dim = int(simulator.nu)

        checkpoint = torch.load(model_dir, map_location=device)
        checkpoint_metadata = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        state_dict = checkpoint["model_state_dict"] if checkpoint_metadata else checkpoint
        if not isinstance(state_dict, Mapping):
            raise ValueError("Flow checkpoint must contain a state dictionary.")

        state_dim, action_dim = infer_mlp_dimensions_from_state_dict(state_dict)
        hidden_dims = infer_mlp_hidden_dims_from_state_dict(state_dict)
        prediction_horizon = int(checkpoint.get("prediction_horizon", 1)) if checkpoint_metadata else 1
        num_inference_steps = 10
        neighbor_feature_dim = 2
        neighbor_slots = max(0, int(simulator.num_robots) - 1)
        
        if checkpoint_metadata:
            state_dim = int(checkpoint.get("state_dim", state_dim))
            action_dim = int(checkpoint.get("action_dim", action_dim))
            hidden_dims_raw = checkpoint.get("hidden_dims")
            if isinstance(hidden_dims_raw, list) and hidden_dims_raw:
                hidden_dims = tuple(int(width) for width in hidden_dims_raw)
            flow_config_raw = checkpoint.get("flow_config")
            if isinstance(flow_config_raw, Mapping):
                num_inference_steps = int(flow_config_raw.get("num_inference_steps", 10))
        
        neighbor_feature_dim = int(checkpoint.get("neighbor_feature_dim", neighbor_feature_dim)) if checkpoint_metadata else neighbor_feature_dim
        neighbor_slots = int(checkpoint.get("neighbor_slots", neighbor_slots)) if checkpoint_metadata else neighbor_slots
        encoder_type = str(checkpoint.get("encoder_type", DEFAULT_ENCODER_TYPE)) if checkpoint_metadata else DEFAULT_ENCODER_TYPE
        encoder_kwargs_raw = checkpoint.get("encoder_kwargs") if checkpoint_metadata else {}
        encoder_kwargs = dict(encoder_kwargs_raw) if isinstance(encoder_kwargs_raw, Mapping) else {}
        obs_encoder: ObservationEncoder = EncoderFactory.create(
            encoder_type=encoder_type,
            state_dim=state_dim,
            neighbor_feature_dim=neighbor_feature_dim,
            neighbor_slots=neighbor_slots,
            **encoder_kwargs,
        )

        policy = FlowPolicy(
            action_dim=action_dim,
            obs_encoder=obs_encoder,
            hidden_dims=hidden_dims,
            prediction_horizon=prediction_horizon,
            num_inference_steps=num_inference_steps,
        )
        state_dict = {
            key.replace("neighbor_encoder.", "obs_encoder.", 1): value
            for key, value in state_dict.items()
        }
        policy.load_state_dict(state_dict)

        policy_display_name = "Flow"
    elif policy_type == "mlp":
        state_dim = sum(
            int(feature_info["shape"][0])
            for feature_name, feature_info in simulator.get_dataset_features().items()
            if is_observation_feature(feature_name)
        )
        action_dim = int(simulator.nu)

        checkpoint = torch.load(model_dir, map_location=device)
        checkpoint_metadata = isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        state_dict = checkpoint["model_state_dict"] if checkpoint_metadata else checkpoint
        if not isinstance(state_dict, Mapping):
            raise ValueError("MLP checkpoint must contain a state dictionary.")

        state_dim, action_dim = infer_mlp_dimensions_from_state_dict(state_dict)
        hidden_dims = infer_mlp_hidden_dims_from_state_dict(state_dict)
        prediction_horizon = int(checkpoint.get("prediction_horizon", 1)) if checkpoint_metadata else 1
        neighbor_feature_dim = 2
        neighbor_slots = max(0, int(simulator.num_robots) - 1)
        
        if checkpoint_metadata:
            state_dim = int(checkpoint.get("state_dim", state_dim))
            action_dim = int(checkpoint.get("action_dim", action_dim))
            hidden_dims_raw = checkpoint.get("hidden_dims")
            if isinstance(hidden_dims_raw, list) and hidden_dims_raw:
                hidden_dims = tuple(int(width) for width in hidden_dims_raw)
        
        neighbor_feature_dim = int(checkpoint.get("neighbor_feature_dim", neighbor_feature_dim)) if checkpoint_metadata else neighbor_feature_dim
        neighbor_slots = int(checkpoint.get("neighbor_slots", neighbor_slots)) if checkpoint_metadata else neighbor_slots
        encoder_type = str(checkpoint.get("encoder_type", DEFAULT_ENCODER_TYPE)) if checkpoint_metadata else DEFAULT_ENCODER_TYPE
        encoder_kwargs_raw = checkpoint.get("encoder_kwargs") if checkpoint_metadata else {}
        encoder_kwargs = dict(encoder_kwargs_raw) if isinstance(encoder_kwargs_raw, Mapping) else {}
        obs_encoder: ObservationEncoder = EncoderFactory.create(
            encoder_type=encoder_type,
            state_dim=state_dim,
            neighbor_feature_dim=neighbor_feature_dim,
            neighbor_slots=neighbor_slots,
            **encoder_kwargs,
        )

        policy = MLPPolicy(
            action_dim=action_dim,
            obs_encoder=obs_encoder,
            hidden_dims=hidden_dims,
            prediction_horizon=prediction_horizon,
        )
        state_dict = {
            key.replace("neighbor_encoder.", "obs_encoder.", 1): value
            for key, value in state_dict.items()
        }
        policy.load_state_dict(state_dict)

        policy_display_name = "MLP"
    else:
        raise ValueError("'policy_type' must be one of {'diffusion', 'act', 'mlp', 'flow'}.")

    policy.eval()
    policy.to(device)

    mlp_observation_feature_cache = None
    if policy_type in {"mlp", "flow"} and simulator.num_robots <= 1:
        dataset_features = simulator.get_dataset_features()
        current_feature_names = [
            name for name in dataset_features if is_observation_feature(name)
        ]
        saved_feature_names = None
        if checkpoint_metadata:
            raw_names = checkpoint.get("obs_feature_names")
            if isinstance(raw_names, list) and all(isinstance(name, str) for name in raw_names):
                saved_feature_names = raw_names

        if saved_feature_names is None:
            saved_feature_names = [
                name for name in current_feature_names if name != "observation.neighbor_mask"
            ]

        mlp_observation_feature_cache = build_observation_feature_pack_cache(
            simulator,
            saved_feature_names,
            allow_schema_subset=True,
        )

    if policy_type in {"diffusion", "act"}:
        policy_cfg = PreTrainedConfig.from_pretrained(model_dir)
        policy_cfg.device = str(device)
        policy_preprocessor, policy_postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=model_dir,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
            },
        )

    # Instantiate expert planner once and reset it for each rollout.
    expert_planner = PlannerFactory.create(planner_name="casadi", simulator=simulator, config=validated_config)
    expert_trajectories: list[np.ndarray] = []
    policy_trajectories: list[np.ndarray] = []
    per_seed_metrics: list[dict[str, Any]] = []

    if len(initial_state_specs) > 0:
        total_rollouts = max(len(seed_specs), len(initial_state_specs))
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
        print(f"evaluating {len(seed_specs)} seeded trajectories")
        rollout_plan = []
        for seed_spec in seed_specs:
            torch_seed = int(seed_spec) if isinstance(seed_spec, int) else int(seed_spec[0])
            rollout_plan.append(
                (
                    seed_spec,
                    "seeded",
                    torch_seed,
                    seed_spec,
                    seed_spec,
                )
            )

    for rollout_idx, (rollout_spec, initial_state_source, torch_seed, seed_value, noise_seed_spec) in enumerate(rollout_plan, start=1):
        torch.manual_seed(torch_seed)

        if initial_state_source == "seeded":
            seed_spec = rollout_spec
            initial_state = sample_initial_state(simulator=simulator, seed_spec=seed_spec)
        elif initial_state_source == "provided":
            initial_state = simulator.validate_state(rollout_spec).copy()
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
            policy_preprocessor=policy_preprocessor,
            policy_postprocessor=policy_postprocessor,
            observation_feature_cache=mlp_observation_feature_cache,
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
        choices=["diffusion", "act", "mlp", "flow"],
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

    run_evaluation(
        system=args.system,
        policy_type=args.policy_type,
        config=config,
        model_dir=args.model_dir,
        num_steps=args.num_steps,
        seeds=parse_seed_argument(args.seeds),
        initial_states=parse_initial_states_argument(args.initial_states),
        action_noise_std=args.action_noise_std,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()