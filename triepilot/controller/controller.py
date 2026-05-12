from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from triepilot.controller.features import ControllerFeatures
from triepilot.controller.policy_rule import TrieRulePolicy
from triepilot.controller.tiers import TierConfig


class Recorder(Protocol):
    def write(self, event: dict[str, Any]) -> None:
        ...


@dataclass(slots=True)
class ControllerState:
    recent_accept_ema: float = 0.0
    mean_match_depth: float = 0.0
    slo_violation_ema: float = 0.0
    step_id: int = 0


class TriePilotController:
    def __init__(
        self,
        tiers: dict[str, TierConfig],
        policy: TrieRulePolicy | None = None,
        recorder: Recorder | None = None,
        ema_alpha: float = 0.20,
    ):
        self.tiers = tiers
        self.policy = policy or TrieRulePolicy(tiers)
        self.recorder = recorder
        self.ema_alpha = ema_alpha
        self.state = ControllerState()

    def select(self, features: ControllerFeatures) -> tuple[TierConfig, int]:
        merged = self._merge_state(features)
        start_ns = time.perf_counter_ns()
        tier = self.policy.select(merged)
        overhead_ns = time.perf_counter_ns() - start_ns
        return tier, overhead_ns

    def record_step(
        self,
        *,
        run_id: str,
        model: str,
        device: str,
        features: ControllerFeatures,
        accepted_drafts_sum: int = 0,
        step_latency_us: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> TierConfig:
        tier, overhead_ns = self.select(features)
        event = {
            "run_id": run_id,
            "step_id": self.state.step_id,
            "model": model,
            "device": device,
            "features": self._merge_state(features).to_dict(),
            "accepted_drafts_sum": int(accepted_drafts_sum),
            "accepted_drafts_mean": float(accepted_drafts_sum)
            / max(1, features.batch_size),
            "draft_latency_us": 0,
            "verify_latency_us": 0,
            "step_latency_us": int(step_latency_us),
            "controller_overhead_ns": overhead_ns,
        }
        event.update(tier.to_event_fields())
        if extra:
            event.update(extra)
        if self.recorder is not None:
            self.recorder.write(event)
        self.state.step_id += 1
        return tier

    def observe(
        self,
        *,
        accepted_drafts_mean: float,
        mean_match_depth: float,
        slo_violation: bool = False,
    ) -> None:
        a = self.ema_alpha
        self.state.recent_accept_ema = (
            a * float(accepted_drafts_mean)
            + (1.0 - a) * self.state.recent_accept_ema
        )
        self.state.mean_match_depth = (
            a * float(mean_match_depth) + (1.0 - a) * self.state.mean_match_depth
        )
        self.state.slo_violation_ema = (
            a * float(slo_violation) + (1.0 - a) * self.state.slo_violation_ema
        )

    def _merge_state(self, features: ControllerFeatures) -> ControllerFeatures:
        f = features.normalized()
        return ControllerFeatures(
            batch_size=f.batch_size,
            queue_len=f.queue_len,
            kv_usage=f.kv_usage,
            seq_len_mean=f.seq_len_mean,
            recent_accept_ema=f.recent_accept_ema or self.state.recent_accept_ema,
            mean_match_depth=f.mean_match_depth or self.state.mean_match_depth,
            request_rate=f.request_rate,
            slo_violation_ema=f.slo_violation_ema or self.state.slo_violation_ema,
        ).normalized()

