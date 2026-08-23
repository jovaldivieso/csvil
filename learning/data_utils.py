from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from systems.dynamics import DynamicsProtocol

StructuredObservation = dict[str, torch.Tensor]


def _tensor_field(sample: Mapping[str, Any], name: str) -> torch.Tensor:
    value = sample.get(name)
    if value is None:
        raise ValueError(f"Dataset sample must contain '{name}'.")
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def _build_action_sequence(
    sample: Mapping[str, Any],
    subsequent_samples: list[Mapping[str, Any]],
    prediction_horizon: int,
) -> torch.Tensor:
    """Build action sequence from current and subsequent samples with episode boundary padding.
    
    Args:
        sample: The current sample containing action[t]
        subsequent_samples: List of subsequent samples containing action[t+1], action[t+2], etc.
        prediction_horizon: Number of actions to predict (including current)
    
    Returns:
        Tensor of shape (prediction_horizon, action_dim) containing the action sequence
    """
    actions_list = [_tensor_field(sample, "action")]
    
    # Add subsequent actions, padding with the last action if we run out
    for i in range(1, prediction_horizon):
        if i <= len(subsequent_samples):
            actions_list.append(_tensor_field(subsequent_samples[i - 1], "action"))
        else:
            # Episode boundary: repeat the last available action
            actions_list.append(actions_list[-1].clone())
    
    return torch.stack(actions_list, dim=0)


def format_sample_for_policy(
    sample: Mapping[str, Any],
    simulator: DynamicsProtocol,
    prediction_horizon: int = 1,
    subsequent_samples: list[Mapping[str, Any]] | None = None,
) -> tuple[StructuredObservation, torch.Tensor]:
    """Convert one native decentralized LeRobot frame into policy tensors.
    
    Args:
        sample: A single frame dictionary from LeRobotDataset
        simulator: The dynamics simulator for validation
        prediction_horizon: Number of future action steps to predict
        subsequent_samples: List of subsequent frame samples for multi-step predictions
    """
    if prediction_horizon <= 0:
        raise ValueError("'prediction_horizon' must be positive.")

    if simulator.num_robots > 1 and "observation.neighbor_state" not in sample:
        raise RuntimeError(
            "Legacy centralized multi-robot datasets are no longer supported. "
            "Multi-robot training requires a decentralized dataset format. "
            "Please regenerate your dataset."
        )

    environment_state = _tensor_field(sample, "observation.environment_state")
    state = _tensor_field(sample, "observation.state")
    neighbor_state = _tensor_field(sample, "observation.neighbor_state")
    neighbor_mask = _tensor_field(sample, "observation.neighbor_mask")
    action = _tensor_field(sample, "action")

    if neighbor_mask.numel() == 0 and neighbor_state.numel() != 0:
        raise ValueError(
            "Neighbor state and mask feature dimensions must agree: "
            f"got {neighbor_state.numel()} and {neighbor_mask.numel()}."
        )
    if neighbor_mask.numel() > 0 and neighbor_state.numel() % neighbor_mask.numel() != 0:
        raise ValueError(
            "Neighbor state and mask feature dimensions must agree: "
            f"got {neighbor_state.numel()} and {neighbor_mask.numel()}."
        )
    expected_action_dim = int(simulator.nu // simulator.num_robots)
    if simulator.num_robots > 1 and action.numel() != expected_action_dim:
        raise ValueError(
            "Decentralized action dimension does not match the simulator's local action dimension: "
            f"got {action.numel()}, expected {expected_action_dim}."
        )

    observation = {
        "observation.environment_state": environment_state,
        "observation.state": state,
        "observation.neighbor_state": neighbor_state,
        "observation.neighbor_mask": neighbor_mask,
    }
    
    # Build action sequence with proper horizon handling
    if subsequent_samples is None:
        subsequent_samples = []
    actions = _build_action_sequence(sample, subsequent_samples, prediction_horizon)
    
    return observation, actions


def collate_batch_for_policy(
    batch: Sequence[Mapping[str, Any]],
    simulator: DynamicsProtocol,
    prediction_horizon: int = 1,
    dataset: object | None = None,
) -> tuple[StructuredObservation, torch.Tensor]:
    """Collate native LeRobot frames into batched policy observations and actions.
    
    Args:
        batch: Sequence of sample dictionaries from LeRobotDataset
        simulator: The dynamics simulator for validation
        prediction_horizon: Number of future action steps to predict per frame
        dataset: Optional LeRobotDataset instance to fetch subsequent actions for horizons > 1
    
    Returns:
        Tuple of (batched_observations_dict, batched_actions_tensor)
    """
    if not batch:
        raise ValueError("Cannot collate an empty dataset batch.")

    subsequent_samples_by_item = _bulk_future_samples(
        batch=batch,
        dataset=dataset,
        prediction_horizon=prediction_horizon,
    )
    formatted = []
    for sample, subsequent_samples in zip(batch, subsequent_samples_by_item):
        formatted_obs, formatted_actions = format_sample_for_policy(
            sample=sample,
            simulator=simulator,
            prediction_horizon=prediction_horizon,
            subsequent_samples=subsequent_samples,
        )
        formatted.append((formatted_obs, formatted_actions))
    
    observations, actions = zip(*formatted)
    return (
        {
            name: torch.stack([observation[name] for observation in observations])
            for name in observations[0]
        },
        torch.stack(actions),
    )


def _bulk_future_samples(
    batch: Sequence[Mapping[str, Any]],
    dataset: object | None,
    prediction_horizon: int,
) -> list[list[Mapping[str, Any]]]:
    """Fetch all same-episode future frames for a batch with one dataset query."""
    future_indices_by_item: list[list[int]] = [[] for _ in batch]
    all_future_indices: list[int] = []

    if dataset is None or prediction_horizon <= 1:
        return future_indices_by_item

    for item_idx, sample in enumerate(batch):
        if "index" not in sample or "episode_index" not in sample:
            continue
        current_index = int(torch.as_tensor(sample["index"]).item())
        future_indices = [
            current_index + horizon_step
            for horizon_step in range(1, prediction_horizon)
        ]
        future_indices_by_item[item_idx] = future_indices
        all_future_indices.extend(future_indices)

    if not all_future_indices:
        return [[] for _ in batch]

    try:
        bulk_future_samples = dataset[all_future_indices]
    except (IndexError, KeyError, TypeError, AttributeError):
        return [[] for _ in batch]

    fetched_by_index = _samples_by_index(bulk_future_samples, all_future_indices)
    subsequent_samples_by_item: list[list[Mapping[str, Any]]] = []
    for sample, future_indices in zip(batch, future_indices_by_item):
        if "episode_index" not in sample:
            subsequent_samples_by_item.append([])
            continue
        current_episode = int(torch.as_tensor(sample["episode_index"]).item())
        same_episode_samples: list[Mapping[str, Any]] = []
        for future_index in future_indices:
            future_sample = fetched_by_index.get(future_index)
            if future_sample is None:
                break
            future_episode = future_sample.get("episode_index")
            if future_episode is None:
                break
            if int(torch.as_tensor(future_episode).item()) != current_episode:
                break
            same_episode_samples.append(future_sample)
        subsequent_samples_by_item.append(same_episode_samples)
    return subsequent_samples_by_item


def _samples_by_index(
    bulk_samples: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    requested_indices: Sequence[int],
) -> dict[int, Mapping[str, Any]]:
    """Normalize bulk dataset output into an absolute-indexed sample mapping."""
    if isinstance(bulk_samples, Mapping):
        sample_count = len(requested_indices)
        samples = [
            {
                name: _bulk_value_at_index(values, row_idx, sample_count)
                for name, values in bulk_samples.items()
            }
            for row_idx in range(sample_count)
        ]
    else:
        samples = list(bulk_samples)

    return {
        int(torch.as_tensor(sample["index"]).item()): sample
        for sample in samples
        if "index" in sample
    }


def _bulk_value_at_index(value: Any, index: int, sample_count: int) -> Any:
    """Read one row from a column returned by a bulk dataset query."""
    if isinstance(value, torch.Tensor):
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == sample_count:
        return value[index]
    return value


def create_collate_fn_with_dataset(
    dataset: object,
    simulator: DynamicsProtocol,
    prediction_horizon: int = 1,
):
    """Factory function to create a collate_fn with dataset access for action horizon prediction.
    
    This allows the collate function to fetch subsequent actions from the dataset when
    prediction_horizon > 1, enabling proper multi-step action predictions with episode
    boundary handling.
    
    Args:
        dataset: LeRobotDataset instance for fetching subsequent frames
        simulator: The dynamics simulator
        prediction_horizon: Number of future action steps to predict
    
    Returns:
        A collate function suitable for use with torch.utils.data.DataLoader
    """
    def collate_fn(batch: Sequence[Mapping[str, Any]]) -> tuple[StructuredObservation, torch.Tensor]:
        return collate_batch_for_policy(
            batch=batch,
            simulator=simulator,
            prediction_horizon=prediction_horizon,
            dataset=dataset,
        )
    return collate_fn
