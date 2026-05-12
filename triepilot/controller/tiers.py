from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TierConfig:
    name: str
    enabled: bool
    draft_tokens: int
    max_match_window_size: int
    max_bfs_breadth: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_off(self) -> bool:
        return self.name == "off" or not self.enabled or self.draft_tokens <= 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(data.pop("metadata"))
        return data

    def to_event_fields(self) -> dict[str, Any]:
        return {
            "tier": self.name,
            "draft_tokens": self.draft_tokens,
            "ngram_match_window": self.max_match_window_size,
            "ngram_bfs_breadth": self.max_bfs_breadth,
        }

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> "TierConfig":
        known = {
            "enabled",
            "draft_tokens",
            "max_match_window_size",
            "max_bfs_breadth",
        }
        return cls(
            name=name,
            enabled=bool(data.get("enabled", name != "off")),
            draft_tokens=int(data.get("draft_tokens", 0)),
            max_match_window_size=int(data.get("max_match_window_size", 0)),
            max_bfs_breadth=int(data.get("max_bfs_breadth", 0)),
            metadata={k: v for k, v in data.items() if k not in known},
        )


def load_tiers_from_mapping(config: dict[str, Any]) -> dict[str, TierConfig]:
    tiers_data = config.get("tiers", config)
    tiers = {
        name: TierConfig.from_mapping(name, data)
        for name, data in tiers_data.items()
    }
    if "off" not in tiers:
        tiers["off"] = TierConfig("off", False, 0, 0, 0)
    return tiers

