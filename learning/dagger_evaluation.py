from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.seed_utils import DEFAULT_MULTI_ROBOT_SEED_STRIDE
from systems.dynamics import DynamicsProtocol


@dataclass(frozen=True)
class DaggerEvalMetrics:
    success_rate: float
    mean_steps: float
    min_steps: int
    max_steps: int
    num_episodes: int


def evaluation_seed_specs(
    simulator: DynamicsProtocol,
    num_episodes: int,
    seed_start: int,
) -> list[int | list[int]]:
    if num_episodes < 0:
        raise ValueError("'num_episodes' must be non-negative.")

    if simulator.num_robots <= 1:
        return [int(seed_start) + idx for idx in range(num_episodes)]

    return [
        [
            int(seed_start) + idx + DEFAULT_MULTI_ROBOT_SEED_STRIDE * robot_idx
            for robot_idx in range(simulator.num_robots)
        ]
        for idx in range(num_episodes)
    ]


def sample_initial_state(
    simulator: DynamicsProtocol,
    seed_spec: int | list[int],
) -> np.ndarray:
    if isinstance(seed_spec, int):
        rng = np.random.default_rng(int(seed_spec))
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


def rollout_policy_with_action_fn(
    simulator: DynamicsProtocol,
    initial_state: np.ndarray,
    num_steps: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None] | None = None,
    action_noise_std: float = 0.0,
    action_noise_rng: np.random.Generator | None = None,
) -> tuple[bool, int]:
    state = simulator.reset(initial_state)
    if reset_fn is not None:
        reset_fn()

    if simulator.is_done(state):
        return True, 0

    for step in range(1, num_steps + 1):
        observation = simulator.observe(state)
        action = action_fn(observation)
        executed_action = apply_execution_noise(
            simulator=simulator,
            action=action,
            action_noise_std=action_noise_std,
            rng=action_noise_rng,
        )
        state = simulator.step(state, executed_action)

        if simulator.should_terminate_rollout(state):
            return True, step

    return False, num_steps


def action_noise_rng_from_seed_spec(seed_spec: int | list[int]) -> np.random.Generator:
    if isinstance(seed_spec, int):
        return np.random.default_rng(np.random.SeedSequence([int(seed_spec), 1]))

    return np.random.default_rng(
        np.random.SeedSequence([int(seed) for seed in seed_spec] + [1])
    )


def evaluate_policy_rollouts(
    simulator: DynamicsProtocol,
    num_episodes: int,
    num_steps: int,
    seed_start: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None] | None = None,
    action_noise_std: float = 0.0,
) -> DaggerEvalMetrics | None:
    if num_episodes == 0:
        return None
    if num_steps <= 0:
        raise ValueError("'num_steps' must be positive.")

    seed_specs = evaluation_seed_specs(
        simulator=simulator,
        num_episodes=num_episodes,
        seed_start=seed_start,
    )

    successes = 0
    steps_taken: list[int] = []
    for seed_spec in seed_specs:
        initial_state = sample_initial_state(simulator=simulator, seed_spec=seed_spec)
        action_noise_rng = action_noise_rng_from_seed_spec(seed_spec)
        reached_goal, rollout_steps = rollout_policy_with_action_fn(
            simulator=simulator,
            initial_state=initial_state,
            num_steps=num_steps,
            action_fn=action_fn,
            reset_fn=reset_fn,
            action_noise_std=action_noise_std,
            action_noise_rng=action_noise_rng,
        )
        successes += int(reached_goal)
        steps_taken.append(int(rollout_steps))

    return DaggerEvalMetrics(
        success_rate=float(successes) / float(num_episodes),
        mean_steps=float(np.mean(np.asarray(steps_taken, dtype=float))),
        min_steps=min(steps_taken),
        max_steps=max(steps_taken),
        num_episodes=num_episodes,
    )