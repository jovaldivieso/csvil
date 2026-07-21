from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


class VectorValidationError(ValueError):
    """Raised when a state, observation, or action vector has the wrong shape."""


@dataclass(frozen=True)
class VectorSpec:
    """Schema for a 1D numeric vector used by simulator interfaces."""

    name: str
    size: int


def as_vector(values: Iterable[float], spec: VectorSpec) -> np.ndarray:
    """Convert values into a flat float vector with strict dimensional validation."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise VectorValidationError(
            f"{spec.name} must be 1D with shape ({spec.size},), got {arr.shape}."
        )
    if arr.shape[0] != spec.size:
        raise VectorValidationError(
            f"{spec.name} must have length {spec.size}, got {arr.shape[0]}."
        )
    return arr
