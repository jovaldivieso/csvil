from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DaggerEvalMetrics:
    success_rate: float
    mean_steps: float
    min_steps: int
    max_steps: int
    num_episodes: int
