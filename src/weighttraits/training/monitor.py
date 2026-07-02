"""Loss monitoring rules for long training jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LossMonitorConfig:
    metric: str = "eval_loss"
    mode: str = "min"
    min_delta: float = 0.0
    patience: int | None = None
    plateau_window: int | None = None
    plateau_min_delta: float = 0.0
    loss_increase_relative: float | None = None
    loss_increase_patience: int = 1


@dataclass(frozen=True)
class TrainingEvent:
    step: int
    train_loss: float | None = None
    eval_loss: float | None = None
    learning_rate: float | None = None

    def metric_value(self, metric: str) -> float | None:
        if metric == "train_loss":
            return self.train_loss
        if metric == "eval_loss":
            return self.eval_loss
        raise ValueError(f"unsupported loss monitor metric: {metric}")


@dataclass(frozen=True)
class MonitorDecision:
    step: int
    value: float | None
    best_value: float | None
    should_stop: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


class LossMonitor:
    """Stateful early-warning and stopping logic for scalar training losses."""

    def __init__(self, config: LossMonitorConfig | None = None) -> None:
        self.config = config or LossMonitorConfig()
        if self.config.mode not in {"min", "max"}:
            raise ValueError(f"unsupported monitor mode: {self.config.mode}")
        if self.config.patience is not None and self.config.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.config.plateau_window is not None and self.config.plateau_window < 2:
            raise ValueError("plateau_window must be at least 2")
        self.best_value: float | None = None
        self.best_step: int | None = None
        self.bad_count = 0
        self.loss_increase_count = 0
        self.values: list[tuple[int, float]] = []

    def update(self, event: TrainingEvent) -> MonitorDecision:
        value = event.metric_value(self.config.metric)
        if value is None:
            return self._decision(event.step, None, [], [])

        value = float(value)
        self.values.append((event.step, value))
        warnings: list[str] = []
        reasons: list[str] = []

        improved = self._is_improvement(value)
        if improved:
            self.best_value = value
            self.best_step = event.step
            self.bad_count = 0
            self.loss_increase_count = 0
        else:
            self.bad_count += 1
            if self._is_loss_increase(value):
                self.loss_increase_count += 1
            else:
                self.loss_increase_count = 0

        if self._loss_increase_warning_ready():
            warnings.append(
                f"{self.config.metric} has increased relative to best for "
                f"{self.loss_increase_count} monitored step(s)"
            )

        if self.config.patience is not None and self.bad_count >= self.config.patience:
            reasons.append(
                f"no {self.config.metric} improvement for {self.bad_count} monitored step(s)"
            )

        if self._plateau_reached():
            reasons.append(
                f"{self.config.metric} absolute change <= {self.config.plateau_min_delta} "
                f"over last {self.config.plateau_window} monitored values"
            )

        return self._decision(event.step, value, reasons, warnings)

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.config.mode == "min":
            return value < self.best_value - self.config.min_delta
        return value > self.best_value + self.config.min_delta

    def _is_loss_increase(self, value: float) -> bool:
        if self.best_value is None or self.config.loss_increase_relative is None:
            return False
        threshold = abs(self.best_value) * self.config.loss_increase_relative
        if self.config.mode == "min":
            return value > self.best_value + threshold
        return value < self.best_value - threshold

    def _loss_increase_warning_ready(self) -> bool:
        if self.config.loss_increase_relative is None:
            return False
        return self.loss_increase_count >= max(1, self.config.loss_increase_patience)

    def _plateau_reached(self) -> bool:
        window = self.config.plateau_window
        if window is None or len(self.values) < window:
            return False
        recent = [value for _, value in self.values[-window:]]
        diffs = [abs(curr - prev) for prev, curr in zip(recent, recent[1:])]
        return all(diff <= self.config.plateau_min_delta for diff in diffs)

    def _decision(
        self,
        step: int,
        value: float | None,
        reasons: list[str],
        warnings: list[str],
    ) -> MonitorDecision:
        return MonitorDecision(
            step=step,
            value=value,
            best_value=self.best_value,
            should_stop=bool(reasons),
            reasons=reasons,
            warnings=warnings,
            state={
                "best_step": self.best_step,
                "bad_count": self.bad_count,
                "loss_increase_count": self.loss_increase_count,
                "n_observations": len(self.values),
            },
        )
