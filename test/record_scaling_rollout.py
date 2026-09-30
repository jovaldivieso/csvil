"""Record an MP4 of one policy-only rollout, for the fleet sizes the expert cannot reach.

``test/evaluate_policy.py`` already writes a video, but it rolls the CasADi expert out
from the same start in order to show the two side by side, and the MPC solve grows
super-quadratically in the fleet size -- unusable past roughly eight robots. This CLI is
the counterpart of ``test/evaluate_scaling.py``: policy only, so a 32-robot episode can be
filmed, and it replays that script's episode seeding exactly, so the episode a results row
summarises is the episode the video shows.

Nothing here re-implements the rollout. The seeding (``evaluation_seed_specs``, the
per-episode ``torch.manual_seed``, ``action_noise_rng_for_rollout``), the joint action and
the outcome tests are the same functions ``evaluate_scaling.evaluate_fleet`` calls, and the
animation is ``test/utils.py``'s ``save_xy_rollout_video``.

Which episode to film is a question the CSV cannot answer -- it stores rates, not episodes
-- so ``--pick`` scans the seeded episodes and reports the index it chose:

    auto          the first fleet success; failing that, the best near-miss
    success       a fleet success only, and an error if there is none
    best-robots   the most robots at their goal without ever colliding

Usage:
    python test/record_scaling_rollout.py \
        --checkpoint outputs/data_mid_best/models/gnn_mlp_unicycle2_fleet_06_s0/mlp_dagger_checkpoint.pt \
        --config test/config/study/circle/unicycle2_circle_32.yaml \
        --use-config-start --action-noise-std 0.03 --episodes 200 --stride 2 \
        --output outputs/data_mid_best/videos/gnn_n06_ring32.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "test"))

from core.config import load_and_validate_system_config  # noqa: E402
from core.factory import DynamicsFactory  # noqa: E402
from evaluate_scaling import (  # noqa: E402
    collided_robots, config_start_state, load_policy, min_pair_distance, robots_at_goal,
    step_budget,
)
from learning.dagger import (  # noqa: E402
    ObservationHistoryBuffer, apply_execution_noise, build_decentralized_joint_action,
    evaluation_seed_specs, sample_initial_state,
)
from systems.dynamics import DynamicsProtocol  # noqa: E402
from systems.seed_utils import (  # noqa: E402
    action_noise_rng_for_rollout, default_action_noise_seed_for_config,
)
from utils import save_xy_rollout_video  # noqa: E402

ENCODER_LABELS = {"deepset": "DeepSet", "transformer": "Transformer", "gnn": "GNN"}


@dataclass
class Episode:
    """One rollout: the states to animate and the outcome that decides if it is the one."""

    index: int
    states: np.ndarray            # (T + 1, fleet state dim)
    goal_state: np.ndarray
    reached_goal: bool
    collided: bool
    robots_at_goal: int
    robots_collided: int
    num_robots: int
    min_pair_distance: float

    @property
    def success(self) -> bool:
        """As in evaluate_scaling: goals reached and not a single collision on the way."""
        return self.reached_goal and not self.collided

    @property
    def robot_successes(self) -> int:
        """GLAS eq. 6: at its goal at the end and never within d_collision of another."""
        return self.robots_at_goal

    def summary(self) -> str:
        verdict = "success" if self.success else ("collision" if self.collided else "timeout")
        return (
            f"episode {self.index}: {verdict}, {self.robots_at_goal}/{self.num_robots} robots "
            f"at goal, {self.robots_collided} in a collision, closest pair "
            f"{self.min_pair_distance:.2f} m, {len(self.states) - 1} steps"
        )


def roll_out(
    simulator: DynamicsProtocol,
    policy,
    device: torch.device,
    episode_index: int,
    seed_spec,
    steps: int,
    action_noise_std: float,
    action_noise_seed: int,
    fixed_initial_state: np.ndarray | None,
    observation_horizon: int,
) -> Episode:
    """One episode, seeded exactly as ``evaluate_scaling.evaluate_fleet`` seeds it.

    Any divergence here would make the video show a different episode from the one the
    results row counted, so the order of the seeding calls matters as much as their values.
    """
    torch.manual_seed(episode_index)
    if fixed_initial_state is not None:
        state = simulator.reset(fixed_initial_state.copy())
    else:
        state = simulator.reset(sample_initial_state(simulator, seed_spec))
    goal_state = simulator.goal_state.copy()
    history_buffer = (
        ObservationHistoryBuffer(observation_horizon, int(simulator.num_robots))
        if observation_horizon > 1 else None
    )
    noise_rng = action_noise_rng_for_rollout(action_noise_seed, seed_spec=seed_spec)

    states = [np.asarray(state, dtype=float).copy()]
    closest = min_pair_distance(simulator, state)
    ever_collided = collided_robots(simulator, state)
    reached_goal = False

    for _ in range(steps):
        observation = simulator.observe(state, validate=False)
        action = build_decentralized_joint_action(
            simulator, policy, observation, device,
            observation_horizon=observation_horizon, history_buffer=history_buffer,
        )
        state = simulator.step(
            state, apply_execution_noise(simulator, action, action_noise_std, noise_rng),
            validate=False,
        )
        states.append(np.asarray(state, dtype=float).copy())
        closest = min(closest, min_pair_distance(simulator, state))
        ever_collided |= collided_robots(simulator, state)
        # Episodes run on after a collision, as in evaluate_scaling: stopping here would
        # cut the clip at the first touch and score the uninvolved robots as failed.
        if simulator.should_terminate_rollout(state):
            reached_goal = True
            break

    at_goal = robots_at_goal(simulator, state)
    return Episode(
        index=episode_index,
        states=np.stack(states),
        goal_state=np.asarray(goal_state, dtype=float),
        reached_goal=reached_goal,
        collided=bool(ever_collided.any()),
        robots_at_goal=int((at_goal & ~ever_collided).sum()),
        robots_collided=int(ever_collided.sum()),
        num_robots=int(simulator.num_robots),
        min_pair_distance=closest,
    )


def choose(episodes: list[Episode], strategy: str) -> Episode:
    """The episode to film.

    ``best-robots`` ranks by robots that both arrived and stayed clear, then by how far
    the closest pair stayed apart -- of two episodes with the same count, the one that
    kept more room is the better demonstration.
    """
    successes = [episode for episode in episodes if episode.success]
    if strategy == "success":
        if not successes:
            raise SystemExit(
                "no episode succeeded at fleet level; use --pick best-robots to film the "
                "best near-miss instead"
            )
        return successes[0]
    if strategy == "auto" and successes:
        return successes[0]
    return max(episodes, key=lambda e: (e.robot_successes, e.min_pair_distance))


def default_title(checkpoint: dict, config_path: str, episode: Episode) -> str:
    encoder = str(checkpoint.get("encoder_type", "?"))
    train_fleet = int(checkpoint["neighbor_slots"]) + 1
    scenario = Path(config_path).parent.name or "scenario"
    verdict = (
        "all robots at goal, no collision" if episode.success
        else f"{episode.robots_at_goal}/{episode.num_robots} robots at goal, "
             f"{episode.robots_collided} in a collision"
    )
    return (
        f"{ENCODER_LABELS.get(encoder, encoder)} trained on {train_fleet} robots\n"
        f"{scenario}, {episode.num_robots} robots — {verdict}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="metadata .pt from train_dagger.py")
    parser.add_argument("--config", required=True, help="one multi_robot YAML config")
    parser.add_argument("--output", type=Path, required=True, help="the .mp4 to write")
    parser.add_argument("--episodes", type=int, default=50,
                        help="how many seeded episodes to scan for one worth filming")
    parser.add_argument("--episode-index", type=int, default=None,
                        help="film exactly this episode index instead of scanning")
    parser.add_argument("--seed-start", type=int, default=50000,
                        help="must match the evaluation run, or the episodes differ")
    parser.add_argument("--action-noise-std", type=float, default=0.0,
                        help="match the evaluation run (the ring was evaluated at 0.03)")
    parser.add_argument("--use-config-start", action="store_true",
                        help="start from the config's per-robot 'start' entries, as the ring does")
    parser.add_argument("--steps", type=int, default=None,
                        help="fixed step budget; derived from the config's distances when omitted")
    parser.add_argument("--step-budget-factor", type=float, default=3.0)
    parser.add_argument("--pick", choices=["auto", "success", "best-robots"], default="auto")
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth simulator step, to shorten a long clip")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--title", default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, mps; autodetected when omitted")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least 1.")

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    policy, checkpoint = load_policy(args.checkpoint, device)
    observation_horizon = int(checkpoint.get("observation_horizon", 1))
    config = load_and_validate_system_config("multi_robot", args.config)
    simulator = DynamicsFactory.create(system_name="multi_robot", config=config)
    fixed_start = config_start_state(config) if args.use_config_start else None
    if fixed_start is not None and simulator.is_collision(fixed_start):
        raise SystemExit(f"{args.config}: configured start state is already in collision.")
    steps = args.steps if args.steps is not None else step_budget(
        simulator, args.step_budget_factor, fixed_start)

    print(
        f"checkpoint: {args.checkpoint}\n"
        f"encoder: {checkpoint.get('encoder_type')} | trained on "
        f"{int(checkpoint['neighbor_slots']) + 1} robots | config: {args.config} | "
        f"{simulator.num_robots} robots | {steps} steps | device: {device}"
    )

    seed_specs = evaluation_seed_specs(simulator, args.episodes, args.seed_start)
    # A fixed start with no action noise makes every episode identical for a deterministic
    # policy -- the same collapse evaluate_scaling reports -- so scanning more is wasted.
    deterministic = (
        fixed_start is not None
        and args.action_noise_std == 0.0
        and str(checkpoint.get("policy_type", "mlp")).lower() != "flow"
    )
    if deterministic and args.episodes > 1 and args.episode_index is None:
        print("  note: fixed start, no action noise, deterministic policy — scanning 1 episode")
        seed_specs = seed_specs[:1]

    wanted = range(len(seed_specs)) if args.episode_index is None else [args.episode_index]
    if args.episode_index is not None and args.episode_index >= len(seed_specs):
        raise SystemExit(f"--episode-index {args.episode_index} needs --episodes above it.")

    episodes: list[Episode] = []
    for episode_index in wanted:
        episode = roll_out(
            simulator=simulator, policy=policy, device=device,
            episode_index=args.seed_start + episode_index,
            seed_spec=seed_specs[episode_index], steps=steps,
            action_noise_std=args.action_noise_std,
            action_noise_seed=default_action_noise_seed_for_config(config),
            fixed_initial_state=fixed_start, observation_horizon=observation_horizon,
        )
        # The index the results row would use, not the seed: they differ by seed_start.
        episode.index = episode_index
        episodes.append(episode)
        print(f"  {episode.summary()}")
        if args.pick in ("auto", "success") and episode.success:
            break

    chosen = episodes[0] if args.episode_index is not None else choose(episodes, args.pick)
    print(f"\nfilming {chosen.summary()}")

    states = chosen.states[::args.stride]
    # The last state is what the outcome was read from, so it must not be dropped by the
    # stride -- otherwise the clip ends before the arrival it claims to show.
    if not np.array_equal(states[-1], chosen.states[-1]):
        states = np.vstack([states, chosen.states[-1]])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    video_path = save_xy_rollout_video(
        simulator=simulator,
        trajectories=[states],
        path_to_output=str(args.output),
        title=args.title or default_title(checkpoint, args.config, chosen),
        show_heading=not simulator.is_euclidean,
        fps=args.fps,
        # path_labels=[f"{checkpoint.get('encoder_type')} policy"],
        goal_states=[chosen.goal_state],
    )
    if video_path is None:
        raise SystemExit("no video written: matplotlib has no ffmpeg writer available.")
    print(f"wrote {video_path} ({len(states)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
