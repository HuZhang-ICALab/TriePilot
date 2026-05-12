import unittest
import tempfile
from pathlib import Path

from triepilot.experiments.analysis import aggregate_metrics, read_jsonl


class AnalysisTest(unittest.TestCase):
    def test_aggregate_metrics_computes_latency_and_node_ratios(self):
        rows = [
            {
                "TPOT": 10,
                "verified_nodes": 4,
                "accepted_tokens": 2,
                "wasted_nodes": 2,
                "step_latency_us": 1000,
                "controller_time_us": 10,
                "strategy_bank_hit": True,
            },
            {
                "TPOT": 20,
                "verified_nodes": 6,
                "accepted_tokens": 3,
                "wasted_nodes": 3,
                "step_latency_us": 2000,
                "controller_time_us": 30,
                "strategy_bank_hit": False,
            },
        ]

        metrics = aggregate_metrics(rows)

        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["mean_TPOT"], 15)
        self.assertEqual(metrics["accepted_per_verified_node"], 0.5)
        self.assertEqual(metrics["wasted_node_ratio"], 0.5)
        self.assertEqual(metrics["strategy_bank_hit_rate"], 0.5)
        self.assertEqual(metrics["controller_overhead_p50_us"], 10)

    def test_read_jsonl_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('\ufeff{"TPOT": 1}\n', encoding="utf-8")

            self.assertEqual(read_jsonl(path), [{"TPOT": 1}])


if __name__ == "__main__":
    unittest.main()
