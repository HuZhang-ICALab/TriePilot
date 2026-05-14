from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class A100LaunchScriptTest(unittest.TestCase):
    def test_a100_launchers_delegate_to_run_server_wrapper(self):
        for script_name in ("a100_no_spec.sh", "a100_ngram_static.sh"):
            with self.subTest(script=script_name):
                text = (REPO_ROOT / "scripts" / "launch" / script_name).read_text(
                    encoding="utf-8"
                )

                self.assertIn("scripts/run_server.sh", text)
                self.assertNotIn("python -m sglang.launch_server", text)

    def test_run_server_passes_optional_triepilot_trace_flags(self):
        text = (REPO_ROOT / "scripts" / "run_server.sh").read_text(encoding="utf-8")

        self.assertIn("WORKSPACE_SGLANG_SRC", text)
        self.assertIn("third_party/sglang_flex/python", text)
        self.assertIn("WORKSPACE_ROOT", text)
        self.assertIn("PYTHONPATH=\"${WORKSPACE_ROOT}:${WORKSPACE_SGLANG_SRC}\"", text)
        self.assertIn("TRIEPILOT_TRACE_PATH", text)
        self.assertIn("--triepilot-trace-path", text)
        self.assertIn("TRIEPILOT_RUN_ID", text)
        self.assertIn("--triepilot-run-id", text)
        self.assertIn("TRIEPILOT_ALLOCATION_POLICY", text)
        self.assertIn("--triepilot-allocation-policy", text)
        self.assertIn("TRIEPILOT_BATCH_BUDGET", text)
        self.assertIn("--triepilot-batch-budget", text)
        self.assertIn("TRIEPILOT_SHAPE_BUCKETS", text)
        self.assertIn("--triepilot-shape-buckets", text)
        self.assertIn("TRIEPILOT_SHAPE_BUCKET_MODE", text)
        self.assertIn("--triepilot-shape-bucket-mode", text)

    def test_session5_allocator_runner_covers_request_level_baselines(self):
        text = (
            REPO_ROOT / "scripts" / "remote" / "a100_session5_allocator_baselines.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("MIXED_PAIRS", text)
        self.assertIn("LEFT_RATIOS", text)
        self.assertIn("B_BATCH_VALUES", text)
        self.assertIn("batch_global_budget([0-9]+)", text)
        self.assertIn("equal_budget_allocation", text)
        self.assertIn("random_budget_allocation", text)
        self.assertIn("match_depth_greedy", text)
        self.assertIn("accept_ema_greedy", text)
        self.assertIn("triepilot_allocation", text)
        self.assertIn("TRIEPILOT_BATCH_BUDGET", text)
        self.assertIn("allocator_summary.csv", text)
        self.assertIn("session5_sweep_summary.csv", text)
        self.assertIn("mean_target_forward_time_us", text)
        self.assertIn("cuda_graph_token_shape_ok_ratio", text)
        self.assertIn("TRIEPILOT_SHAPE_BUCKETS", text)
        self.assertIn("TRIEPILOT_SHAPE_BUCKET_MODE", text)
        self.assertIn("CUDA_GRAPH_MAX_BS", text)
        self.assertIn("MAX_RUNNING_REQUESTS", text)
        self.assertIn("cuda_graph_max_bs:", text)
        self.assertIn("max_running_requests:", text)
        self.assertIn('"cuda_graph_max_bs":', text)
        self.assertIn('"max_running_requests":', text)
        self.assertIn("MATERIALIZE_FILL_FILTERED", text)
        self.assertIn("--tokenizer-model", text)
        self.assertIn("--context-len", text)

    def test_a100_static_tiers_runner_covers_step1_budget_set(self):
        text = (
            REPO_ROOT / "scripts" / "remote" / "a100_step1_static_tiers.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("BUDGETS=\"0 2 4 8 16 24 32\"", text)
        self.assertIn("raw_step_events.jsonl", text)
        self.assertIn("tier_summary.csv", text)
        self.assertIn("notes.md", text)
        self.assertIn("output_throughput", text)
        self.assertIn("mean_tpot_ms", text)


if __name__ == "__main__":
    unittest.main()
