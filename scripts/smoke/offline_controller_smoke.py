from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triepilot.controller import (  # noqa: E402
    ControllerFeatures,
    TriePilotController,
    load_tiers_from_mapping,
)
from triepilot.logging import TriePilotRecorder, validate_trace_event  # noqa: E402
from triepilot.replay.build_profile_table import build_profile_table  # noqa: E402
from triepilot.replay.distill_rule import distill_text  # noqa: E402


TIERS = {
    "tiers": {
        "off": {
            "enabled": False,
            "draft_tokens": 0,
            "max_match_window_size": 0,
            "max_bfs_breadth": 0,
        },
        "tiny": {
            "enabled": True,
            "draft_tokens": 2,
            "max_match_window_size": 4,
            "max_bfs_breadth": 1,
        },
        "small": {
            "enabled": True,
            "draft_tokens": 4,
            "max_match_window_size": 8,
            "max_bfs_breadth": 2,
        },
    }
}


def main() -> None:
    out = Path("experiments/traces/local_3050_offline_smoke.jsonl")
    if out.exists():
        out.unlink()

    controller = TriePilotController(
        tiers=load_tiers_from_mapping(TIERS),
        recorder=TriePilotRecorder(out),
    )
    scenarios = [
        ControllerFeatures(kv_usage=0.10, seq_len_mean=64),
        ControllerFeatures(kv_usage=0.20, seq_len_mean=96, mean_match_depth=4),
        ControllerFeatures(kv_usage=0.30, seq_len_mean=128, recent_accept_ema=1.2),
        ControllerFeatures(kv_usage=0.91, seq_len_mean=160, recent_accept_ema=2.0),
    ]

    for i in range(24):
        features = scenarios[i % len(scenarios)]
        tier = controller.record_step(
            run_id="local_3050_offline_smoke",
            model="offline-simulator",
            device="RTX3050",
            features=features,
            accepted_drafts_sum=i % 5,
            step_latency_us=500 + i * 7,
        )
        controller.observe(
            accepted_drafts_mean=0.0 if tier.is_off else (i % 5),
            mean_match_depth=features.mean_match_depth,
        )

    errors = []
    for line in out.read_text(encoding="utf-8").splitlines():
        import json

        errors.extend(validate_trace_event(json.loads(line)))
    if errors:
        raise SystemExit("\n".join(errors))

    profile = build_profile_table(out)
    print(f"trace={out}")
    print(f"profile_tiers={sorted(profile)}")
    print(distill_text(out))


if __name__ == "__main__":
    main()
