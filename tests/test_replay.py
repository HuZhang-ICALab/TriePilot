import json
import tempfile
import unittest
from pathlib import Path

from triepilot.replay.build_profile_table import build_profile_table
from triepilot.replay.offline_oracle import label_event


class ReplayTest(unittest.TestCase):
    def test_oracle_uses_candidate_rewards(self):
        self.assertEqual(label_event({"candidate_rewards": {"tiny": 1, "small": 2}}), "small")

    def test_profile_table_groups_by_tier(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            rows = [
                {"tier": "tiny", "accepted_drafts_mean": 1, "step_latency_us": 10},
                {"tier": "tiny", "accepted_drafts_mean": 3, "step_latency_us": 20},
                {"tier": "off", "accepted_drafts_mean": 0, "step_latency_us": 30},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            table = build_profile_table(path)
            self.assertEqual(table["tiny"]["count"], 2)
            self.assertEqual(table["tiny"]["accepted_drafts_mean"], 2)


if __name__ == "__main__":
    unittest.main()

