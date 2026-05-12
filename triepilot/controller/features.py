from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ControllerFeatures:
    batch_size: int = 1
    queue_len: int = 0
    kv_usage: float = 0.0
    seq_len_mean: float = 0.0
    recent_accept_ema: float = 0.0
    mean_match_depth: float = 0.0
    request_rate: float = 0.0
    slo_violation_ema: float = 0.0

    def normalized(self) -> "ControllerFeatures":
        return ControllerFeatures(
            batch_size=max(0, int(self.batch_size)),
            queue_len=max(0, int(self.queue_len)),
            kv_usage=min(1.0, max(0.0, float(self.kv_usage))),
            seq_len_mean=max(0.0, float(self.seq_len_mean)),
            recent_accept_ema=max(0.0, float(self.recent_accept_ema)),
            mean_match_depth=max(0.0, float(self.mean_match_depth)),
            request_rate=max(0.0, float(self.request_rate)),
            slo_violation_ema=min(1.0, max(0.0, float(self.slo_violation_ema))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ControllerFeatures":
        source = data.get("features", data)
        return cls(
            batch_size=int(source.get("batch_size", 1)),
            queue_len=int(source.get("queue_len", 0)),
            kv_usage=float(source.get("kv_usage", 0.0)),
            seq_len_mean=float(source.get("seq_len_mean", 0.0)),
            recent_accept_ema=float(source.get("recent_accept_ema", 0.0)),
            mean_match_depth=float(source.get("mean_match_depth", 0.0)),
            request_rate=float(source.get("request_rate", 0.0)),
            slo_violation_ema=float(source.get("slo_violation_ema", 0.0)),
        ).normalized()

