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
        self.assertIn("TRIEPILOT_TRACE_PATH", text)
        self.assertIn("--triepilot-trace-path", text)
        self.assertIn("TRIEPILOT_RUN_ID", text)
        self.assertIn("--triepilot-run-id", text)

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
