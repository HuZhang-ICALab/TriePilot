import json
import tempfile
import unittest
from pathlib import Path

from triepilot.workloads.materialize import materialize_workload_to_sharegpt


class WorkloadMaterializeTest(unittest.TestCase):
    def test_materialize_mixed_workload_rows_to_sharegpt_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized_dir = root / "normalized"
            normalized_dir.mkdir()
            (normalized_dir / "code.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "fix bug", "reference": "done"}),
                        json.dumps({"prompt": "add test", "reference": "ok"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (normalized_dir / "math.jsonl").write_text(
                json.dumps({"messages": [{"role": "user", "content": "2+2?"}]})
                + "\n",
                encoding="utf-8",
            )
            workload = [
                {"request_id": "r0", "dataset": "code", "sample_index": 1},
                {"request_id": "r1", "dataset": "math", "sample_index": 0},
            ]
            output = root / "mixed.json"

            rows = materialize_workload_to_sharegpt(
                workload,
                normalized_dir=normalized_dir,
                output_path=output,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["conversations"][0]["value"], "add test")
            self.assertEqual(rows[0]["triepilot_dataset"], "code")
            self.assertEqual(rows[1]["conversations"][0]["value"], "2+2?")
            self.assertTrue(output.exists())

    def test_materialize_can_replace_context_overflow_with_same_dataset_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized_dir = root / "normalized"
            normalized_dir.mkdir()
            (normalized_dir / "json_tool.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "too long prompt", "reference": "skip"}),
                        json.dumps({"prompt": "short prompt", "reference": "use"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            workload = [
                {"request_id": "r0", "dataset": "json_tool", "sample_index": 0}
            ]
            output = root / "mixed.json"

            rows = materialize_workload_to_sharegpt(
                workload,
                normalized_dir=normalized_dir,
                output_path=output,
                prompt_token_counter=lambda text: len(text.split()),
                max_prompt_tokens=2,
                fill_filtered=True,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["conversations"][0]["value"], "short prompt")
            self.assertEqual(rows[0]["triepilot_original_sample_index"], 0)
            self.assertEqual(rows[0]["triepilot_sample_index"], 1)


if __name__ == "__main__":
    unittest.main()
