import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class NormalizeDatasetCliTest(unittest.TestCase):
    def test_cli_normalizes_json_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "sharegpt.json"
            output = tmp_path / "sharegpt.jsonl"
            ids = tmp_path / "ids.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "conversations": [
                                {"from": "human", "value": "Prompt"},
                                {"from": "gpt", "value": "Answer"},
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/normalize_dataset.py",
                    "--dataset",
                    "sharegpt",
                    "--normalizer",
                    "sharegpt",
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--sample-ids",
                    str(ids),
                    "--seed",
                    "7",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            )

            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("wrote 1 rows", result.stdout)
            self.assertEqual(row["sample_id"], "sharegpt-7-000000")
            self.assertEqual(json.loads(ids.read_text(encoding="utf-8"))["count"], 1)


if __name__ == "__main__":
    unittest.main()
