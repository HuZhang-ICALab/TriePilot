from __future__ import annotations

import math
import random
from dataclasses import dataclass

from triepilot.controller.features import ControllerFeatures
from triepilot.controller.tiers import TierConfig


@dataclass(slots=True)
class ArmState:
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        return 0.0 if self.pulls == 0 else self.reward_sum / self.pulls


class UcbBanditPolicy:
    def __init__(
        self,
        tiers: dict[str, TierConfig],
        exploration: float = 0.6,
        rng: random.Random | None = None,
    ):
        self.tiers = tiers
        self.exploration = exploration
        self.rng = rng or random.Random(0)
        self.arms = {
            name: ArmState()
            for name, tier in tiers.items()
            if name != "off" and not tier.is_off
        }
        self.total_pulls = 0

    def select(self, features: ControllerFeatures) -> TierConfig:
        f = features.normalized()
        if f.kv_usage >= 0.88:
            return self.tiers["off"]
        if not self.arms:
            return self.tiers["off"]

        for name, state in self.arms.items():
            if state.pulls == 0:
                return self.tiers[name]

        log_total = math.log(max(2, self.total_pulls))
        scores = {
            name: state.mean_reward
            + self.exploration * math.sqrt(log_total / state.pulls)
            for name, state in self.arms.items()
        }
        best_name = max(scores, key=scores.get)
        return self.tiers[best_name]

    def observe(self, tier_name: str, reward: float) -> None:
        if tier_name not in self.arms:
            return
        state = self.arms[tier_name]
        state.pulls += 1
        state.reward_sum += float(reward)
        self.total_pulls += 1

