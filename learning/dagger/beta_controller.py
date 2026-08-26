from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExpertMixBetaController:
    beta_start: float
    beta_end: float
    decay_rounds: int
    beta_decay_rate: float | None = None
    decay_after_success_rate: float | None = None
    adaptive_recovery: bool = False

    current_beta: float = 0.0
    decay_active: bool = False
    previous_eval_success_rate: float | None = None

    def __post_init__(self) -> None:
        self.current_beta = float(self.beta_start)

    def _gate_is_open(self, eval_success_rate: float) -> bool:
        if self.decay_after_success_rate is None:
            return True
        return float(eval_success_rate) > float(self.decay_after_success_rate)

    def _step_delta(self) -> float:
        if self.beta_decay_rate is not None:
            return abs(float(self.beta_decay_rate))
        if self.decay_rounds <= 1:
            return 0.0
        return abs(float(self.beta_end) - float(self.beta_start)) / float(self.decay_rounds - 1)

    def _beta_bounds(self) -> tuple[float, float]:
        if self.beta_decay_rate is not None:
            return 0.0, max(0.0, float(self.beta_start))
        return min(float(self.beta_start), float(self.beta_end)), max(float(self.beta_start), float(self.beta_end))

    def _decrease_beta(self) -> None:
        delta = self._step_delta()
        if self.beta_decay_rate is not None:
            next_beta = self.current_beta - delta
        else:
            schedule_direction = 1.0 if float(self.beta_end) > float(self.beta_start) else -1.0
            next_beta = self.current_beta + schedule_direction * delta
        lower_bound, upper_bound = self._beta_bounds()
        self.current_beta = float(min(max(next_beta, lower_bound), upper_bound))

    def _increase_beta(self) -> None:
        delta = self._step_delta()
        next_beta = self.current_beta + delta
        lower_bound, upper_bound = self._beta_bounds()
        self.current_beta = float(min(max(next_beta, lower_bound), upper_bound))

    def update_after_evaluation(self, eval_success_rate: float | None) -> None:
        if eval_success_rate is None:
            if self.decay_after_success_rate is None:
                if not self.decay_active:
                    self.decay_active = True
                self._decrease_beta()
            return
        eval_rate = float(eval_success_rate)
        if not self.decay_active:
            if self._gate_is_open(eval_rate):
                self.decay_active = True
                self._decrease_beta()
            self.previous_eval_success_rate = eval_rate
            return
        previous_rate = self.previous_eval_success_rate
        if self.adaptive_recovery and previous_rate is not None and eval_rate < previous_rate:
            self._increase_beta()
        else:
            self._decrease_beta()
        self.previous_eval_success_rate = eval_rate

    def prime_from_evaluation(self, eval_success_rate: float | None) -> None:
        """Prime gate state from an evaluation without advancing the beta schedule."""
        if eval_success_rate is None:
            return
        eval_rate = float(eval_success_rate)
        if self._gate_is_open(eval_rate):
            self.decay_active = True
        self.previous_eval_success_rate = eval_rate


def scheduled_expert_mix_beta(
    round_offset: int,
    beta_start: float,
    beta_end: float,
    decay_rounds: int,
    beta_decay_rate: float | None = None,
) -> float:
    if beta_decay_rate is not None:
        clamped_offset = max(round_offset, 0)
        return float(max(0.0, beta_start - float(beta_decay_rate) * float(clamped_offset)))
    if decay_rounds <= 1:
        return float(beta_start)
    clamped_offset = min(max(round_offset, 0), decay_rounds - 1)
    progress = float(clamped_offset) / float(decay_rounds - 1)
    return float(beta_start + (beta_end - beta_start) * progress)
