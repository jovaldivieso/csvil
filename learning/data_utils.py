from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from systems.dynamics import DynamicsProtocol

StructuredObservation = dict[str, torch.Tensor]

# Fields stacked across observation_horizon (oldest to newest) in
# format_sample_for_policy -- shared with build_observation_history_cache so the
# two can never drift apart.
_HISTORY_STACKED_FIELDS = (
    "observation.neighbor_state",
    "observation.neighbor_mask",
    "observation.state",
    "observation.state_mask",
)


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
    observation_horizon: int = 1,
    past_samples: list[Mapping[str, Any]] | None = None,
    precomputed_actions: torch.Tensor | None = None,
    precomputed_history: Mapping[str, torch.Tensor] | None = None,
) -> tuple[StructuredObservation, torch.Tensor]:
    """Convert one native decentralized LeRobot frame into policy tensors.

    Args:
        sample: A single frame dictionary from LeRobotDataset
        simulator: The dynamics simulator for validation
        prediction_horizon: Number of future action steps to predict
        subsequent_samples: List of subsequent frame samples for multi-step predictions
        observation_horizon: Number of observation frames, including the current frame
        past_samples: Same-episode frames before the current frame, oldest first
        precomputed_actions: If given, used directly as the (prediction_horizon,
            action_dim) action target instead of building it from
            ``subsequent_samples`` -- see ``build_action_window_cache``.
        precomputed_history: If given, used directly as the already-stacked
            ``_HISTORY_STACKED_FIELDS`` values instead of building them from
            ``past_samples`` -- see ``build_observation_history_cache``.
    """
    if prediction_horizon <= 0:
        raise ValueError("'prediction_horizon' must be positive.")
    if observation_horizon <= 0:
        raise ValueError("'observation_horizon' must be positive.")

    if simulator.num_robots > 1 and "observation.neighbor_state" not in sample:
        raise RuntimeError(
            "Multi-robot training requires a decentralized dataset format. "
            "Please regenerate your dataset."
        )

    environment_state = _tensor_field(sample, "observation.environment_state")
    state = _tensor_field(sample, "observation.state")
    neighbor_state = _tensor_field(sample, "observation.neighbor_state")
    neighbor_mask = _tensor_field(sample, "observation.neighbor_mask")
    action = _tensor_field(sample, "action")

    if neighbor_mask.numel() == 0:
        if neighbor_state.numel() != 0:
            raise ValueError(
                "Neighbor state and mask feature dimensions must agree: "
                f"got {neighbor_state.numel()} and {neighbor_mask.numel()}."
            )
    elif neighbor_state.numel() == 0 or neighbor_state.numel() % neighbor_mask.numel() != 0:
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
    # Zero-fill any not-yet-collected frames rather than repeating the earliest
    # real one, so a padded (feature=0, mask=0) slot reads the same as a
    # genuine out-of-visibility-radius neighbor instead of a plausible-looking
    # duplicate of real motion history (see ObservationHistoryBuffer.append_and_stack).
    # observation.state (this robot's own proprioception) is stacked too, for
    # the same reason as the neighbor tensors: the policy needs its own
    # recent motion history to correctly interpret neighbor history, which is
    # expressed in this robot's own frame at each past instant.
    # observation.state_mask is its companion, mirroring
    # observation.neighbor_mask: always 1.0 at generation time, so a
    # zero-padded pre-episode frame is distinguishable from a genuine
    # [v=0, omega=0] reading instead of silently identical to one.
    latest_frame_fields = ("observation.environment_state",)
    observation: StructuredObservation = {}
    if precomputed_history is not None:
        for name in _HISTORY_STACKED_FIELDS:
            observation[name] = precomputed_history[name]
    else:
        history = list(past_samples or []) + [sample]
        if len(history) > observation_horizon:
            history = history[-observation_horizon:]
        pad_count = observation_horizon - len(history)
        for name in _HISTORY_STACKED_FIELDS:
            real_tensors = [_tensor_field(frame, name) for frame in history]
            padding = [torch.zeros_like(real_tensors[0]) for _ in range(pad_count)]
            observation[name] = torch.cat(padding + real_tensors, dim=0)
    observation.update(
        {name: _tensor_field(sample, name) for name in latest_frame_fields}
    )

    # Build action sequence with proper horizon handling
    if precomputed_actions is not None:
        actions = precomputed_actions
    else:
        subsequent_samples = list(subsequent_samples or [])
        actions = _build_action_sequence(sample, subsequent_samples, prediction_horizon)

    return observation, actions


def collate_batch_for_policy(
    batch: Sequence[Mapping[str, Any]],
    simulator: DynamicsProtocol,
    prediction_horizon: int = 1,
    dataset: object | None = None,
    observation_horizon: int = 1,
    action_window_cache: torch.Tensor | None = None,
    observation_history_cache: Mapping[str, torch.Tensor] | None = None,
) -> tuple[StructuredObservation, torch.Tensor]:
    """Collate native LeRobot frames into batched policy observations and actions.

    Args:
        batch: Sequence of sample dictionaries from LeRobotDataset
        simulator: The dynamics simulator for validation
        prediction_horizon: Number of future action steps to predict per frame
        observation_horizon: Number of observation frames to stack per frame
        dataset: Optional LeRobotDataset instance to fetch subsequent actions for horizons > 1
        action_window_cache: Optional (len(dataset), prediction_horizon, action_dim)
            tensor from ``build_action_window_cache``. When given, actions are read
            directly from it by absolute frame index instead of fetching and
            re-deriving them from ``prediction_horizon - 1`` future dataset rows on
            every call -- the same fetch would otherwise repeat, unchanged, on
            every epoch.
        observation_history_cache: Optional dict from ``build_observation_history_cache``.
            When given, history-stacked observation fields are read directly from it
            by absolute frame index instead of fetching and re-deriving them from
            ``observation_horizon - 1`` past dataset rows on every call.

    Returns:
        Tuple of (batched_observations_dict, batched_actions_tensor)
    """
    if not batch:
        raise ValueError("Cannot collate an empty dataset batch.")

    normalized_batch = list(batch)
    if action_window_cache is not None:
        subsequent_samples_by_item = [[] for _ in normalized_batch]
    else:
        subsequent_samples_by_item = _bulk_future_samples(
            batch=normalized_batch,
            dataset=dataset,
            prediction_horizon=prediction_horizon,
        )
    if observation_history_cache is not None:
        past_samples_by_item = [[] for _ in normalized_batch]
    else:
        past_samples_by_item = _bulk_past_samples(
            batch=normalized_batch,
            dataset=dataset,
            observation_horizon=observation_horizon,
        )
    formatted = []
    for sample, subsequent_samples, past_samples in zip(
        normalized_batch, subsequent_samples_by_item, past_samples_by_item
    ):
        frame_index = None
        if action_window_cache is not None or observation_history_cache is not None:
            frame_index = int(torch.as_tensor(sample["index"]).item())

        precomputed_actions = None
        if action_window_cache is not None:
            precomputed_actions = action_window_cache[frame_index]

        precomputed_history = None
        if observation_history_cache is not None:
            precomputed_history = {
                name: cache[frame_index] for name, cache in observation_history_cache.items()
            }

        formatted_obs, formatted_actions = format_sample_for_policy(
            sample=sample,
            simulator=simulator,
            prediction_horizon=prediction_horizon,
            subsequent_samples=subsequent_samples,
            observation_horizon=observation_horizon,
            past_samples=past_samples,
            precomputed_actions=precomputed_actions,
            precomputed_history=precomputed_history,
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

    dataset_length = len(dataset)
    valid_future_indices = [
        index for index in all_future_indices
        if 0 <= index < dataset_length
    ]
    if not valid_future_indices:
        return [[] for _ in batch]

    bulk_future_samples = _fetch_samples(dataset, valid_future_indices)

    fetched_by_index = _samples_by_index(bulk_future_samples, valid_future_indices)
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


def _bulk_past_samples(
    batch: Sequence[Mapping[str, Any]],
    dataset: object | None,
    observation_horizon: int,
) -> list[list[Mapping[str, Any]]]:
    """Fetch same-episode history frames, ordered from oldest to newest."""
    past_indices_by_item: list[list[int]] = [[] for _ in batch]
    all_past_indices: list[int] = []

    if dataset is None or observation_horizon <= 1:
        return past_indices_by_item

    for item_idx, sample in enumerate(batch):
        if "index" not in sample or "episode_index" not in sample:
            continue
        current_index = int(torch.as_tensor(sample["index"]).item())
        past_indices = [
            current_index - history_step
            for history_step in range(observation_horizon - 1, 0, -1)
            if current_index - history_step >= 0
        ]
        past_indices_by_item[item_idx] = past_indices
        all_past_indices.extend(past_indices)

    if not all_past_indices:
        return past_indices_by_item

    dataset_length = len(dataset)
    valid_past_indices = [
        index for index in all_past_indices
        if 0 <= index < dataset_length
    ]
    if not valid_past_indices:
        return [[] for _ in batch]

    bulk_past_samples = _fetch_samples(dataset, valid_past_indices)

    fetched_by_index = _samples_by_index(bulk_past_samples, valid_past_indices)
    past_samples_by_item: list[list[Mapping[str, Any]]] = []
    for sample, past_indices in zip(batch, past_indices_by_item):
        if "episode_index" not in sample:
            past_samples_by_item.append([])
            continue
        current_episode = int(torch.as_tensor(sample["episode_index"]).item())
        same_episode_samples: list[Mapping[str, Any]] = []
        for past_index in past_indices:
            past_sample = fetched_by_index.get(past_index)
            if past_sample is None:
                continue
            past_episode = past_sample.get("episode_index")
            if past_episode is None:
                continue
            if int(torch.as_tensor(past_episode).item()) == current_episode:
                same_episode_samples.append(past_sample)
        past_samples_by_item.append(same_episode_samples)
    return past_samples_by_item


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


def _fetch_samples(dataset: object, indices: Sequence[int]) -> Mapping[str, Any] | Sequence[Mapping[str, Any]]:
    """Fetch dataset rows, supporting datasets with scalar-only indexing."""
    try:
        return dataset[list(indices)]
    except (IndexError, KeyError, TypeError, AttributeError):
        return [dataset[index] for index in indices]


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
    observation_horizon: int = 1,
    action_window_cache: torch.Tensor | None = None,
    observation_history_cache: Mapping[str, torch.Tensor] | None = None,
):
    """Factory function to create a collate_fn with dataset access for action horizon prediction.

    This allows the collate function to fetch subsequent actions from the dataset when
    prediction_horizon > 1, enabling proper multi-step action predictions with episode
    boundary handling.

    Args:
        dataset: LeRobotDataset instance for fetching subsequent frames
        simulator: The dynamics simulator
        prediction_horizon: Number of future action steps to predict
        action_window_cache: Optional cache from ``build_action_window_cache``,
            forwarded to ``collate_batch_for_policy`` -- see its docstring.
        observation_history_cache: Optional cache from ``build_observation_history_cache``,
            forwarded to ``collate_batch_for_policy`` -- see its docstring.

    Returns:
        A collate function suitable for use with torch.utils.data.DataLoader
    """
    def collate_fn(batch: Sequence[Mapping[str, Any]]) -> tuple[StructuredObservation, torch.Tensor]:
        return collate_batch_for_policy(
            batch=batch,
            simulator=simulator,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            dataset=dataset,
            action_window_cache=action_window_cache,
            observation_history_cache=observation_history_cache,
        )
    return collate_fn


def build_action_window_cache(
    dataset: Sequence[Mapping[str, Any]],
    prediction_horizon: int,
) -> torch.Tensor:
    """Precompute the padded action window for every frame in ``dataset``, once.

    ``_bulk_future_samples`` re-derives each frame's (prediction_horizon, action_dim)
    target on every collate call by fetching ``prediction_horizon - 1`` future dataset
    rows -- identical work repeated on every batch of every training epoch, dominated
    by ``LeRobotDataset.__getitem__`` overhead (real LeRobotDataset instances reject
    list-style bulk indexing, so that fetch silently falls back to one Python-level
    ``dataset[i]`` call per requested row). This computes the same windows exactly
    once per dataset load by reading each frame's ``action``/``episode_index`` a
    single time and constructing all windows with one vectorized gather.

    Uses the same episode-boundary rule as ``_build_action_sequence``: since frames
    within an episode occupy a contiguous, increasing block of absolute indices (how
    ``LeRobotDataset.add_frame``/``save_episode`` write them), a window that would
    run past its own episode's last frame clamps to that last frame's action instead
    of crossing into the next episode.

    Args:
        dataset: LeRobotDataset (or compatible) instance; each row must expose
            ``"action"``, ``"index"``, and ``"episode_index"``.
        prediction_horizon: Number of action steps per window.

    Returns:
        Tensor of shape ``(len(dataset), prediction_horizon, action_dim)``, indexable
        by each frame's absolute ``"index"`` value.
    """
    if prediction_horizon <= 0:
        raise ValueError("'prediction_horizon' must be positive.")
    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot build an action-window cache for an empty dataset.")

    actions_by_position: list[torch.Tensor] = [None] * n  # type: ignore[list-item]
    episode_ids = torch.empty(n, dtype=torch.long)
    for position in range(n):
        row = dataset[position]
        frame_index = int(torch.as_tensor(row["index"]).item())
        actions_by_position[frame_index] = _tensor_field(row, "action")
        episode_ids[frame_index] = int(torch.as_tensor(row["episode_index"]).item())
    action_stack = torch.stack(actions_by_position)  # (N, action_dim)

    if prediction_horizon <= 1:
        return action_stack.unsqueeze(1)

    is_last_in_episode = torch.ones(n, dtype=torch.bool)
    is_last_in_episode[:-1] = episode_ids[:-1] != episode_ids[1:]
    last_indices = torch.nonzero(is_last_in_episode, as_tuple=True)[0]
    positions = torch.arange(n)
    episode_end = last_indices[torch.searchsorted(last_indices, positions)]

    offsets = torch.arange(prediction_horizon)
    raw_indices = positions.unsqueeze(1) + offsets.unsqueeze(0)  # (N, prediction_horizon)
    clipped_indices = torch.minimum(raw_indices, episode_end.unsqueeze(1))
    return action_stack[clipped_indices]


def build_observation_history_cache(
    dataset: Sequence[Mapping[str, Any]],
    observation_horizon: int,
) -> dict[str, torch.Tensor]:
    """Precompute the padded, flattened history window for every frame in ``dataset``,
    once, for each field in ``_HISTORY_STACKED_FIELDS``.

    ``_bulk_past_samples`` re-derives each frame's observation-history window on every
    collate call by fetching ``observation_horizon - 1`` past dataset rows -- the exact
    same anti-pattern ``build_action_window_cache`` fixes on the action side (real
    LeRobotDataset instances reject list-style bulk indexing, so that fetch silently
    falls back to one Python-level ``dataset[i]`` call per requested row, repeated
    unchanged on every epoch). This computes the same windows exactly once by reading
    each frame's history-stacked fields and ``episode_index`` a single time, then
    building all windows with one vectorized gather per field.

    Unlike ``build_action_window_cache`` (which clamps to and repeats an episode's
    last real action), a window reaching before its own episode's first frame is
    ZERO-padded here instead, matching ``format_sample_for_policy``'s rule: a
    zero/mask=0 slot must read the same as a genuine out-of-visibility-radius
    neighbor, never a plausible-looking duplicate of real motion history.

    Args:
        dataset: LeRobotDataset (or compatible) instance; each row must expose
            ``"index"``, ``"episode_index"``, and every field in
            ``_HISTORY_STACKED_FIELDS``.
        observation_horizon: Number of frames to stack (including the current one).

    Returns:
        Dict mapping each field in ``_HISTORY_STACKED_FIELDS`` to a tensor of shape
        ``(len(dataset), observation_horizon * field_dim)``, already flattened
        oldest-to-newest exactly as ``format_sample_for_policy`` concatenates it, and
        indexable by each frame's absolute ``"index"`` value.
    """
    if observation_horizon <= 0:
        raise ValueError("'observation_horizon' must be positive.")
    n = len(dataset)
    if n == 0:
        raise ValueError("Cannot build an observation-history cache for an empty dataset.")

    field_values: dict[str, list[torch.Tensor | None]] = {
        name: [None] * n for name in _HISTORY_STACKED_FIELDS  # type: ignore[misc]
    }
    episode_ids = torch.empty(n, dtype=torch.long)
    for position in range(n):
        row = dataset[position]
        frame_index = int(torch.as_tensor(row["index"]).item())
        episode_ids[frame_index] = int(torch.as_tensor(row["episode_index"]).item())
        for name in _HISTORY_STACKED_FIELDS:
            field_values[name][frame_index] = _tensor_field(row, name)

    stacked = {name: torch.stack(values) for name, values in field_values.items()}  # each (N, field_dim)

    if observation_horizon <= 1:
        return stacked

    is_first_in_episode = torch.ones(n, dtype=torch.bool)
    is_first_in_episode[1:] = episode_ids[1:] != episode_ids[:-1]
    first_indices = torch.nonzero(is_first_in_episode, as_tuple=True)[0]
    positions = torch.arange(n)
    # For each position, the largest "first-in-episode" index that is <= it --
    # i.e. where its own episode starts.
    boundary_slot = torch.searchsorted(first_indices, positions, right=True) - 1
    episode_start = first_indices[boundary_slot]

    offsets = torch.arange(observation_horizon - 1, -1, -1)  # oldest -> newest
    raw_indices = positions.unsqueeze(1) - offsets.unsqueeze(0)  # (N, observation_horizon)
    in_episode = raw_indices >= episode_start.unsqueeze(1)
    safe_indices = raw_indices.clamp(min=0)

    result: dict[str, torch.Tensor] = {}
    for name, values in stacked.items():
        field_dim = values.shape[1]
        gathered = values[safe_indices]  # (N, observation_horizon, field_dim)
        gathered = torch.where(in_episode.unsqueeze(-1), gathered, torch.zeros_like(gathered))
        result[name] = gathered.reshape(n, observation_horizon * field_dim)
    return result
