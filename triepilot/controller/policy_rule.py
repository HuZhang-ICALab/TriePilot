from __future__ import annotations

from triepilot.controller.features import ControllerFeatures
from triepilot.controller.tiers import TierConfig


class TrieRulePolicy:
    """Conservative fast-path policy used before hardware-specific profiling."""

    def __init__(self, tiers: dict[str, TierConfig]):
        self.tiers = tiers

    def select(self, features: ControllerFeatures) -> TierConfig:
        f = features.normalized()

        if f.kv_usage >= 0.88:
            return self.tiers["off"]

        if f.slo_violation_ema >= 0.50 and f.recent_accept_ema < 1.0:
            return self.tiers["off"]

        if f.recent_accept_ema < 0.25 and f.mean_match_depth < 2:
            return self.tiers["off"]

        if f.recent_accept_ema >= 1.5 and f.mean_match_depth >= 5:
            return self.tiers.get("medium", self.tiers.get("small", self.tiers["off"]))

        if f.recent_accept_ema >= 0.8 or f.mean_match_depth >= 3:
            return self.tiers.get("small", self.tiers.get("tiny", self.tiers["off"]))

        return self.tiers.get("tiny", self.tiers["off"])

