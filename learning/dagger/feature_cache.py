from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systems.dynamics import DynamicsProtocol


def is_observation_feature(feature_name: str) -> bool:
    return feature_name.startswith("observation.") or ".observation." in feature_name


def observation_feature_names(simulator: DynamicsProtocol) -> list[str]:
    return [
        feature_name
        for feature_name in simulator.get_dataset_features().keys()
        if is_observation_feature(feature_name)
    ]


def action_feature_names(simulator: DynamicsProtocol) -> list[str]:
    return [
        feature_name
        for feature_name in simulator.get_dataset_features().keys()
        if feature_name == "action" or feature_name.endswith(".action")
    ]


@dataclass(frozen=True, slots=True)
class ObservationFeaturePackCache:
    feature_names: tuple[str, ...]
    feature_indices: np.ndarray
    feature_index_slices: tuple[slice, ...]


def observation_dim_from_features(simulator: DynamicsProtocol) -> int:
    return sum(
        int(feature_info["shape"][0])
        for feature_name, feature_info in simulator.get_dataset_features().items()
        if is_observation_feature(feature_name)
    )


def build_observation_feature_pack_cache(
    simulator: DynamicsProtocol,
    feature_names: list[str],
    allow_schema_subset: bool = False,
) -> ObservationFeaturePackCache:
    dataset_features = simulator.get_dataset_features()
    schema_observation_features = tuple(
        feature_name for feature_name in dataset_features.keys() if is_observation_feature(feature_name)
    )
    provided_feature_names = tuple(feature_names)
    if not allow_schema_subset and provided_feature_names != schema_observation_features:
        mismatch_index = next(
            (
                idx
                for idx, (expected_name, provided_name) in enumerate(
                    zip(schema_observation_features, provided_feature_names)
                )
                if expected_name != provided_name
            ),
            None,
        )
        if mismatch_index is not None:
            mismatch_details = (
                f"first mismatch at index {mismatch_index}: expected "
                f"'{schema_observation_features[mismatch_index]}' but got "
                f"'{provided_feature_names[mismatch_index]}'"
            )
        else:
            mismatch_details = (
                "feature list lengths differ: "
                f"expected {len(schema_observation_features)} but got {len(provided_feature_names)}"
            )
        raise ValueError(
            "Observation feature ordering does not match simulator dataset schema; "
            "this can cause silent policy input misalignment. "
            f"{mismatch_details}"
        )

    total_dim = int(simulator.obs_dim)
    dummy_frame = simulator.format_dataset_frame(
        np.arange(total_dim, dtype=np.float32),
        np.zeros(int(simulator.nu), dtype=np.float32),
    )
    if isinstance(dummy_frame, (list, tuple)):
        if not dummy_frame:
            raise ValueError("Simulator dataset formatter returned no frames.")
        dummy_frame = dummy_frame[0]
    feature_indices: list[np.ndarray] = []
    feature_index_slices: list[slice] = []
    start = 0
    for feature_name in provided_feature_names:
        if feature_name not in dummy_frame:
            raise KeyError(f"Observation feature '{feature_name}' is missing from simulator dataset formatter.")
        feature_array = np.asarray(dummy_frame[feature_name])
        if feature_array.ndim == 0:
            raise ValueError(f"Observation feature '{feature_name}' must be indexable from the formatted frame.")
        feature_index_array = feature_array.astype(int, copy=False).reshape(-1)
        stop = start + feature_index_array.shape[0]
        feature_indices.append(feature_index_array)
        feature_index_slices.append(slice(start, stop))
        start = stop

    stacked = np.empty(0, dtype=int) if not feature_indices else np.concatenate(feature_indices).astype(int, copy=False)
    return ObservationFeaturePackCache(
        feature_names=provided_feature_names,
        feature_indices=stacked,
        feature_index_slices=tuple(feature_index_slices),
    )


def pack_observation_features_from_cache(
    observation: np.ndarray,
    feature_cache: ObservationFeaturePackCache,
) -> dict[str, np.ndarray]:
    observation_array = np.asarray(observation)
    packed_values = np.asarray(observation_array[feature_cache.feature_indices], dtype=np.float32)
    return {
        feature_name: np.asarray(packed_values[feature_slice], dtype=np.float32)
        for feature_name, feature_slice in zip(feature_cache.feature_names, feature_cache.feature_index_slices)
    }


def pack_observation_features(
    simulator: DynamicsProtocol,
    observation: np.ndarray,
    feature_names: list[str],
    feature_cache: ObservationFeaturePackCache | None = None,
) -> dict[str, np.ndarray]:
    if feature_cache is None:
        feature_cache = build_observation_feature_pack_cache(simulator, feature_names)
    return pack_observation_features_from_cache(observation, feature_cache)
