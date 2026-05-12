import json
import tempfile
import unittest
from pathlib import Path

from triepilot.controller import ControllerFeatures, TriePilotController, load_tiers_from_mapping
from triepilot.logging import TriePilotRecorder, validate_trace_event


class RecorderTest(unittest.TestCase):
    def test_recorded_event_matches_schema(self):
        tiers = load_tiers_from_mapping(
            {
                "tiers": {
                    "off": {"enabled": False, "draft_tokens": 0},
                    "tiny": {
                        "enabled": True,
                        "draft_tokens": 2,
                        "max_match_window_size": 4,
                        "max_bfs_breadth": 1,
                    },
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            with TriePilotRecorder(path) as recorder:
                controller = TriePilotController(tiers=tiers, recorder=recorder)
                controller.record_step(
                    run_id="test",
                    model="offline",
                    device="cpu",
                    features=ControllerFeatures(mean_match_depth=3),
                    accepted_drafts_sum=1,
                )

            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(validate_trace_event(row), [])


if __name__ == "__main__":
    unittest.main()

