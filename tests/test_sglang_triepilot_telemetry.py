import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SGLANG_ROOT_CANDIDATES = [
    Path(os.environ["SGLANG_SOURCE_ROOT"]).expanduser() / "python"
    if os.environ.get("SGLANG_SOURCE_ROOT")
    else None,
    REPO_ROOT / "third_party" / "sglang_flex" / "python",
    Path("/root/sglang_flex_test/sglang_flex/python"),
]
SGLANG_ROOT = next(
    (
        candidate
        for candidate in SGLANG_ROOT_CANDIDATES
        if candidate is not None and (candidate / "sglang" / "version.py").exists()
    ),
    REPO_ROOT / "third_party" / "sglang_flex" / "python",
)
SERVER_ARGS_PATH = SGLANG_ROOT / "sglang" / "srt" / "server_args.py"
NGRAM_WORKER_PATH = (
    SGLANG_ROOT / "sglang" / "srt" / "speculative" / "ngram_worker.py"
)
SCHEDULE_BATCH_PATH = (
    SGLANG_ROOT / "sglang" / "srt" / "managers" / "schedule_batch.py"
)
RECORDER_PATH = (
    SGLANG_ROOT
    / "sglang"
    / "srt"
    / "speculative"
    / "triepilot"
    / "recorder.py"
)
BUDGET_PATH = (
    SGLANG_ROOT
    / "sglang"
    / "srt"
    / "speculative"
    / "triepilot"
    / "budget.py"
)
FEATURES_PATH = (
    SGLANG_ROOT
    / "sglang"
    / "srt"
    / "speculative"
    / "triepilot"
    / "features.py"
)


def load_recorder_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_triepilot_recorder", RECORDER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_budget_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_triepilot_budget", BUDGET_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_features_module():
    spec = importlib.util.spec_from_file_location(
        "sglang_triepilot_features", FEATURES_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SglangTriePilotTelemetryTest(unittest.TestCase):
    def setUp(self):
        if not SERVER_ARGS_PATH.exists() or not NGRAM_WORKER_PATH.exists():
            self.skipTest("FlexKV-aligned SGLang source is not present locally")

    def test_ngram_recorder_writes_step_telemetry_jsonl(self):
        module = load_recorder_module()

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "ngram_steps.jsonl"
            recorder = module.TriePilotNgramRecorder(
                path=trace_path,
                run_id="unit-run",
            )
            recorder.record_step(
                step_id=3,
                batch_size=2,
                request_ids=["r0", "r1"],
                seq_lens=[11, 17],
                draft_token_num=8,
                accept_lens=[1, 4],
                num_accepted_tokens=5,
                can_run_cuda_graph=True,
                timings_ns={
                    "ngram_query": 1_000,
                    "target_forward": 2_000,
                    "verify": 3_000,
                    "step": 6_000,
                },
            )
            recorder.close()

            rows = trace_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(rows), 1)
        event = json.loads(rows[0])
        self.assertEqual(event["run_id"], "unit-run")
        self.assertEqual(event["event"], "ngram_step")
        self.assertEqual(event["step_id"], 3)
        self.assertEqual(event["batch_size"], 2)
        self.assertEqual(event["allocated_budget"], 16)
        self.assertEqual(event["actual_draft_nodes"], 16)
        self.assertEqual(event["verified_nodes"], 16)
        self.assertEqual(event["accepted_tokens"], 5)
        self.assertEqual(event["wasted_nodes"], 11)
        self.assertEqual(event["accept_lens"], [1, 4])
        self.assertEqual(event["request_ids"], ["r0", "r1"])
        self.assertEqual(event["seq_lens"], [11, 17])
        self.assertEqual(event["ngram_query_time_us"], 1.0)
        self.assertEqual(event["verify_time_us"], 3.0)
        self.assertTrue(event["can_run_cuda_graph"])

    def test_sglang_sources_expose_triepilot_trace_flags_and_worker_hook(self):
        server_args = SERVER_ARGS_PATH.read_text(encoding="utf-8")
        ngram_worker = NGRAM_WORKER_PATH.read_text(encoding="utf-8")
        schedule_batch = SCHEDULE_BATCH_PATH.read_text(encoding="utf-8")

        self.assertIn("triepilot_trace_path", server_args)
        self.assertIn("--triepilot-trace-path", server_args)
        self.assertIn("triepilot_run_id", server_args)
        self.assertIn("--triepilot-run-id", server_args)
        self.assertIn("TriePilotNgramRecorder", ngram_worker)
        self.assertIn("record_step", ngram_worker)
        self.assertIn("self.spec_verify_ct = 0", schedule_batch)
        self.assertIn("self.spec_accepted_tokens = 0", schedule_batch)

    def test_budget_resolver_reads_custom_params_and_keeps_off_requests_active(self):
        module = load_budget_module()

        class Req:
            def __init__(self, custom_params):
                self.sampling_params = type(
                    "SamplingParams", (), {"custom_params": custom_params}
                )()

        reqs = [
            Req({"triepilot_draft_budget": 16}),
            Req({"triepilot_draft_budget": 0}),
            Req({"triepilot_draft_budget": 99}),
            Req(None),
        ]

        budgets, active_lengths = module.resolve_triepilot_draft_budgets(
            reqs, default_budget=16
        )

        self.assertEqual(budgets, [16, 0, 16, 16])
        self.assertEqual(active_lengths, [16, 1, 16, 16])

    def test_recorder_tracks_budget_vectors_and_verify_token_count(self):
        module = load_recorder_module()

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "ngram_steps.jsonl"
            recorder = module.TriePilotNgramRecorder(
                path=trace_path,
                run_id="unit-run",
            )
            recorder.record_step(
                step_id=4,
                batch_size=3,
                request_ids=["r0", "r1", "r2"],
                seq_lens=[11, 17, 23],
                draft_token_num=16,
                requested_draft_budgets=[16, 0, 8],
                active_draft_lengths=[16, 1, 8],
                accept_lens=[1, 0, 3],
                num_accepted_tokens=4,
                can_run_cuda_graph=False,
                timings_ns={
                    "ngram_query": 1_000,
                    "target_forward": 2_000,
                    "verify": 3_000,
                    "step": 6_000,
                },
            )
            recorder.close()

            event = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertEqual(event["allocated_budgets"], [16, 0, 8])
        self.assertEqual(event["active_draft_lengths"], [16, 1, 8])
        self.assertEqual(event["allocated_budget"], 24)
        self.assertEqual(event["actual_draft_nodes"], 24)
        self.assertEqual(event["verify_input_tokens"], 25)
        self.assertEqual(event["verified_nodes"], 24)
        self.assertEqual(event["wasted_nodes"], 20)

    def test_ngram_tree_features_capture_depth_and_branching(self):
        module = load_features_module()

        # Tree: root -> node1 -> node2, and root -> node3.
        mask = [
            1, 0, 0, 0,
            1, 1, 0, 0,
            1, 1, 1, 0,
            1, 0, 0, 1,
        ]

        features = module.compute_ngram_tree_features(
            req_drafts=[10, 11, 12, 13],
            mask=mask,
            draft_token_num=4,
            active_draft_lengths=[4],
        )

        self.assertEqual(len(features), 1)
        self.assertEqual(features[0]["filled_nodes"], 4)
        self.assertEqual(features[0]["match_depth"], 2)
        self.assertEqual(features[0]["candidate_count"], 2)
        self.assertAlmostEqual(features[0]["top_branch_ratio"], 2 / 3)
        self.assertAlmostEqual(features[0]["branch_entropy"], 0.9182958340544896)

    def test_recorder_writes_structural_feature_vectors_and_batch_summaries(self):
        module = load_recorder_module()

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "ngram_steps.jsonl"
            recorder = module.TriePilotNgramRecorder(
                path=trace_path,
                run_id="unit-run",
            )
            recorder.record_step(
                step_id=5,
                batch_size=2,
                request_ids=["r0", "r1"],
                seq_lens=[11, 17],
                draft_token_num=8,
                accept_lens=[1, 4],
                num_accepted_tokens=5,
                can_run_cuda_graph=True,
                timings_ns={
                    "ngram_query": 1_000,
                    "target_forward": 2_000,
                    "verify": 3_000,
                    "step": 6_000,
                },
                structural_features=[
                    {
                        "match_depth": 3,
                        "candidate_count": 2,
                        "branch_entropy": 0.5,
                        "top_branch_ratio": 0.75,
                        "filled_nodes": 8,
                    },
                    {
                        "match_depth": 1,
                        "candidate_count": 1,
                        "branch_entropy": 0.0,
                        "top_branch_ratio": 1.0,
                        "filled_nodes": 8,
                    },
                ],
            )
            recorder.close()

            event = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertEqual(event["match_depths"], [3, 1])
        self.assertEqual(event["candidate_counts"], [2, 1])
        self.assertEqual(event["filled_nodes"], [8, 8])
        self.assertEqual(event["match_depth"], 2.0)
        self.assertEqual(event["candidate_count"], 1.5)
        self.assertEqual(event["branch_entropy"], 0.25)
        self.assertEqual(event["top_branch_ratio"], 0.875)


if __name__ == "__main__":
    unittest.main()
