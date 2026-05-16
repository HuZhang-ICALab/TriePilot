import argparse
import unittest

from scripts.run_bench import build_command


class RunBenchCommandTest(unittest.TestCase):
    def _args(self, dataset_name: str) -> argparse.Namespace:
        return argparse.Namespace(
            backend="sglang",
            conda_bin="/root/anaconda3/bin/conda",
            conda_env="sglang",
            host="127.0.0.1",
            port="30000",
            model="/models/qwen",
            dataset_name=dataset_name,
            dataset_path=None,
            num_prompts=4,
            max_concurrency=1,
            request_rate=1.0,
            random_input_len=32,
            random_output_len=16,
            random_range_ratio=0.25,
            sharegpt_output_len=None,
            sharegpt_context_len=None,
            seed=20260512,
            output_file="/tmp/out.jsonl",
            dry_run=False,
        )

    def test_random_id_datasets_forward_length_arguments(self):
        cmd = build_command(self._args("random-ids"))

        self.assertIn("--random-input-len", cmd)
        self.assertIn("32", cmd)
        self.assertIn("--random-output-len", cmd)
        self.assertIn("16", cmd)
        self.assertIn("--random-range-ratio", cmd)
        self.assertIn("0.25", cmd)

    def test_sharegpt_datasets_forward_length_and_seed_arguments(self):
        args = self._args("sharegpt")
        args.dataset_path = "/tmp/sharegpt.json"
        args.sharegpt_output_len = 16
        args.sharegpt_context_len = 4096

        cmd = build_command(args)

        self.assertIn("--dataset-path", cmd)
        self.assertIn("/tmp/sharegpt.json", cmd)
        self.assertIn("--sharegpt-output-len", cmd)
        self.assertIn("16", cmd)
        self.assertIn("--sharegpt-context-len", cmd)
        self.assertIn("4096", cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("20260512", cmd)


if __name__ == "__main__":
    unittest.main()
