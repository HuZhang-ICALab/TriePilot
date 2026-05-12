import json
import tempfile
import unittest
from pathlib import Path

from triepilot.workloads.builder import (
    build_binary_mix,
    build_homogeneous,
    build_shift_schedule,
    write_jsonl,
)


class WorkloadBuilderTest(unittest.TestCase):
    def test_homogeneous_workload_is_seeded_and_labeled(self):
        rows = build_homogeneous("sharegpt", count=3, seed=7)

        self.assertEqual([row["request_id"] for row in rows], ["sharegpt-7-000000", "sharegpt-7-000001", "sharegpt-7-000002"])
        self.assertEqual({row["dataset"] for row in rows}, {"sharegpt"})
        self.assertEqual({row["workload_type"] for row in rows}, {"homogeneous"})

    def test_binary_mix_respects_ratio_and_interleaves(self):
        rows = build_binary_mix(
            left_dataset="instructcoder",
            right_dataset="gsm8k",
            left_ratio=0.8,
            count=10,
            seed=11,
        )

        self.assertEqual(sum(row["dataset"] == "instructcoder" for row in rows), 8)
        self.assertEqual(sum(row["dataset"] == "gsm8k" for row in rows), 2)
        self.assertEqual(rows[0]["dataset"], "instructcoder")
        self.assertEqual(rows[-1]["workload_type"], "binary_mixed")

    def test_shift_schedule_keeps_phase_boundaries(self):
        rows = build_shift_schedule(
            phases=["instructcoder", "gsm8k", "json_tool", "instructcoder"],
            per_phase=2,
            seed=5,
        )

        self.assertEqual([row["phase"] for row in rows], [1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual([row["dataset"] for row in rows], ["instructcoder", "instructcoder", "gsm8k", "gsm8k", "json_tool", "json_tool", "instructcoder", "instructcoder"])

    def test_write_jsonl_creates_parent_and_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workloads" / "sample.jsonl"
            write_jsonl(path, build_homogeneous("random", count=2, seed=1))

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["dataset"], "random")


if __name__ == "__main__":
    unittest.main()
