from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DaggerEvalMetrics:
    success_rate: float
    mean_steps: float
    min_steps: int
    max_steps: int
    num_episodes: int
    # Split of the same episodes by initial-state/goal source: the leading
    # `usable_count` episodes drawn from the curriculum's explicit
    # initial_states/goal_states lists ("config") versus the remainder
    # sampled via simulator RNG fallback ("random") -- see
    # evaluate_policy_rollouts. Raw counts (not a rate) so "X/Y solved"
    # displays exactly, with no division until a caller wants a percentage.
    # Left at the defaults (0 episodes each) by callers that don't
    # distinguish a source split, e.g. aggregation's own success accounting.
    config_successes: int = 0
    config_num_episodes: int = 0
    random_successes: int = 0
    random_num_episodes: int = 0
