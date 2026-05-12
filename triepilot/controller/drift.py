from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EwmaDriftDetector:
    alpha: float = 0.05
    threshold: float = 0.35
    baseline: float | None = None
    current: float | None = None

    def update(self, value: float) -> bool:
        value = float(value)
        if self.current is None:
            self.current = value
            self.baseline = value
            return False

        self.current = self.alpha * value + (1.0 - self.alpha) * self.current
        assert self.baseline is not None
        return abs(self.current - self.baseline) >= self.threshold

    def reset(self) -> None:
        self.baseline = self.current

