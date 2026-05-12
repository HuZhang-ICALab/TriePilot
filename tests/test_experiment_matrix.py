import unittest

from triepilot.experiments.matrix import build_baseline_matrix


class ExperimentMatrixTest(unittest.TestCase):
    def test_matrix_expands_models_workloads_and_methods(self):
        rows = build_baseline_matrix(
            models=["qwen3_8b"],
            workloads=["homogeneous/sharegpt"],
            methods=["ar_no_spec", "sglang_ngram_default"],
            seed=42,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["run_id"], "qwen3_8b__homogeneous_sharegpt__ar_no_spec__seed42")
        self.assertEqual({row["framework"] for row in rows}, {"sglang"})

    def test_oracle_is_marked_offline(self):
        rows = build_baseline_matrix(
            models=["qwen3_8b"],
            workloads=["mixed/instructcoder_gsm8k_50_50"],
            methods=["oracle_allocation"],
            seed=3,
        )

        self.assertEqual(rows[0]["execution_mode"], "offline_replay")
        self.assertFalse(rows[0]["deployable"])


if __name__ == "__main__":
    unittest.main()
