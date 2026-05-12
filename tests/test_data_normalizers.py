import json
import tempfile
import unittest
from pathlib import Path

from triepilot.data.normalizers import (
    normalize_code_edit_record,
    normalize_cnn_dailymail_record,
    normalize_gsm8k_record,
    normalize_json_tool_record,
    normalize_qa_with_docs_record,
    normalize_sharegpt_record,
    write_normalized_dataset,
)


class DataNormalizerTest(unittest.TestCase):
    def test_sharegpt_record_uses_first_human_prompt_and_assistant_reference(self):
        row = normalize_sharegpt_record(
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi"},
                ]
            },
            dataset="sharegpt",
            index=3,
            seed=9,
        )

        self.assertEqual(row["sample_id"], "sharegpt-9-000003")
        self.assertEqual(row["prompt"], "Hello")
        self.assertEqual(row["reference"], "Hi")
        self.assertEqual(row["messages"][0]["role"], "user")

    def test_gsm8k_record_formats_math_prompt(self):
        row = normalize_gsm8k_record(
            {"question": "What is 2+2?", "answer": "4"},
            dataset="gsm8k",
            index=0,
            seed=1,
        )

        self.assertIn("Solve the math problem", row["prompt"])
        self.assertIn("What is 2+2?", row["prompt"])
        self.assertEqual(row["reference"], "4")

    def test_cnn_dailymail_record_formats_summary_prompt(self):
        row = normalize_cnn_dailymail_record(
            {"article": "Long article", "highlights": "Short summary"},
            dataset="cnn_dailymail",
            index=0,
            seed=1,
        )

        self.assertIn("Summarize", row["prompt"])
        self.assertEqual(row["reference"], "Short summary")

    def test_json_tool_record_preserves_schema_metadata(self):
        row = normalize_json_tool_record(
            {"prompt": "Return JSON", "target_schema": {"type": "object"}},
            dataset="json_tool",
            index=0,
            seed=1,
        )

        self.assertEqual(row["metadata"]["target_schema"], {"type": "object"})

    def test_json_tool_record_builds_prompt_from_json_schema(self):
        row = normalize_json_tool_record(
            {"json_schema": {"type": "object", "properties": {"name": {"type": "string"}}}},
            dataset="json_tool",
            index=0,
            seed=1,
        )

        self.assertIn("valid JSON", row["prompt"])
        self.assertEqual(row["metadata"]["target_schema"]["type"], "object")

    def test_code_edit_record_builds_instruction_prompt(self):
        row = normalize_code_edit_record(
            {"instruction": "Fix bug", "input": "print(1)", "output": "print(2)"},
            dataset="instructcoder",
            index=0,
            seed=1,
        )

        self.assertIn("Fix bug", row["prompt"])
        self.assertIn("print(1)", row["prompt"])
        self.assertEqual(row["reference"], "print(2)")

    def test_qa_with_docs_record_includes_passages(self):
        row = normalize_qa_with_docs_record(
            {"question": "Q?", "answers": ["A"], "docs_sorted": ["doc1", "doc2"]},
            dataset="shared_prefix",
            index=0,
            seed=1,
        )

        self.assertIn("Passages:", row["prompt"])
        self.assertIn("doc1", row["prompt"])
        self.assertEqual(row["reference"], "A")

    def test_write_normalized_dataset_writes_jsonl_and_sample_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "data.jsonl"
            sample_ids = Path(tmp) / "ids.json"
            write_normalized_dataset(
                output,
                sample_ids,
                [
                    {
                        "dataset": "random",
                        "sample_id": "random-1-000000",
                        "prompt": "p",
                        "reference": "",
                    }
                ],
                dataset="random",
                seed=1,
                source="unit",
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            ids = json.loads(sample_ids.read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["sample_id"], "random-1-000000")
            self.assertEqual(ids["sample_ids"], ["random-1-000000"])


if __name__ == "__main__":
    unittest.main()
