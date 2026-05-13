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
TRIEPILOT_INTEGRATION_ROOT = REPO_ROOT / "triepilot" / "sglang_integration"
RECORDER_PATH = (
    TRIEPILOT_INTEGRATION_ROOT / "recorder.py"
)
BUDGET_PATH = (
    TRIEPILOT_INTEGRATION_ROOT / "budget.py"
)
FEATURES_PATH = (
    TRIEPILOT_INTEGRATION_ROOT / "features.py"
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
        self.assertEqual(event["draft_token_num"], 8)
        self.assertEqual(event["cuda_graph_expected_tokens"], 16)
        self.assertEqual(event["cuda_graph_actual_tokens"], 16)
        self.assertTrue(event["cuda_graph_token_shape_ok"])
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
        self.assertIn("triepilot_allocation_policy", server_args)
        self.assertIn("--triepilot-allocation-policy", server_args)
        self.assertIn("triepilot_batch_budget", server_args)
        self.assertIn("--triepilot-batch-budget", server_args)
        self.assertIn("triepilot_allocation", server_args)
        self.assertIn("triepilot.sglang_integration.budget", ngram_worker)
        self.assertIn("triepilot.sglang_integration.features", ngram_worker)
        self.assertIn("triepilot.sglang_integration.recorder", ngram_worker)
        self.assertNotIn("sglang.srt.speculative.triepilot", ngram_worker)
        self.assertIn("TriePilotNgramRecorder", ngram_worker)
        self.assertIn("record_step", ngram_worker)
        self.assertIn("observe_triepilot_accept_lengths", ngram_worker)
        self.assertIn("TriePilotStrategyBank", ngram_worker)
        self.assertIn("self.spec_verify_ct = 0", schedule_batch)
        self.assertIn("self.spec_accepted_tokens = 0", schedule_batch)

    def test_triepilot_integration_logic_lives_in_project_package(self):
        self.assertTrue(BUDGET_PATH.exists())
        self.assertTrue(FEATURES_PATH.exists())
        self.assertTrue(RECORDER_PATH.exists())
        sglang_triepilot_dir = (
            SGLANG_ROOT / "sglang" / "srt" / "speculative" / "triepilot"
        )
        self.assertFalse((sglang_triepilot_dir / "budget.py").exists())
        self.assertFalse((sglang_triepilot_dir / "features.py").exists())
        self.assertFalse((sglang_triepilot_dir / "recorder.py").exists())

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

    def test_equal_allocator_respects_batch_budget_and_caps(self):
        module = load_budget_module()

        reqs = [object(), object(), object(), object()]

        budgets, active_lengths = module.resolve_triepilot_draft_budgets(
            reqs,
            default_budget=16,
            allocation_policy="equal_budget_allocation",
            batch_budget=10,
        )

        self.assertEqual(budgets, [3, 3, 2, 2])
        self.assertEqual(sum(budgets), 10)
        self.assertEqual(active_lengths, [3, 3, 2, 2])

    def test_match_depth_greedy_allocator_ranks_by_structural_feature(self):
        module = load_budget_module()

        reqs = [object(), object(), object()]
        structural_features = [
            {"match_depth": 1},
            {"match_depth": 5},
            {"match_depth": 3},
        ]

        budgets, active_lengths = module.resolve_triepilot_draft_budgets(
            reqs,
            default_budget=8,
            allocation_policy="match_depth_greedy",
            batch_budget=20,
            structural_features=structural_features,
        )

        self.assertEqual(budgets, [4, 8, 8])
        self.assertEqual(active_lengths, [4, 8, 8])

    def test_accept_ema_greedy_allocator_uses_request_history(self):
        module = load_budget_module()

        class Req:
            def __init__(self, accept_ema):
                self.triepilot_accept_len_ema = accept_ema

        reqs = [Req(0.5), Req(2.0), Req(1.0)]

        budgets, active_lengths = module.resolve_triepilot_draft_budgets(
            reqs,
            default_budget=8,
            allocation_policy="accept_ema_greedy",
            batch_budget=10,
        )

        self.assertEqual(budgets, [0, 8, 2])
        self.assertEqual(active_lengths, [1, 8, 2])

    def test_accept_ema_observer_updates_request_state(self):
        module = load_budget_module()

        class Req:
            def __init__(self, accept_ema):
                self.triepilot_accept_len_ema = accept_ema

        reqs = [Req(1.0), Req(0.0)]

        module.observe_triepilot_accept_lengths(reqs, [3, 2], alpha=0.25)

        self.assertAlmostEqual(reqs[0].triepilot_accept_len_ema, 1.5)
        self.assertAlmostEqual(reqs[1].triepilot_accept_len_ema, 0.5)

    def test_triepilot_allocator_uses_regime_strategy_bank_and_returns_metadata(self):
        module = load_budget_module()

        class Req:
            def __init__(self, accept_ema):
                self.triepilot_accept_len_ema = accept_ema

        reqs = [Req(3.0), Req(0.0), Req(1.0)]
        structural_features = [
            {
                "match_depth": 6,
                "candidate_count": 1,
                "branch_entropy": 0.0,
                "top_branch_ratio": 1.0,
                "filled_nodes": 16,
            },
            {
                "match_depth": 0,
                "candidate_count": 0,
                "branch_entropy": 0.0,
                "top_branch_ratio": 0.0,
                "filled_nodes": 16,
            },
            {
                "match_depth": 2,
                "candidate_count": 4,
                "branch_entropy": 1.5,
                "top_branch_ratio": 0.4,
                "filled_nodes": 16,
            },
        ]
        bank = module.TriePilotStrategyBank(max_budget=16)

        budgets, active_lengths, metadata = module.resolve_triepilot_draft_budgets(
            reqs,
            default_budget=16,
            allocation_policy="triepilot_allocation",
            batch_budget=32,
            structural_features=structural_features,
            strategy_bank=bank,
            return_metadata=True,
        )

        self.assertEqual(budgets, [16, 0, 8])
        self.assertEqual(active_lengths, [16, 1, 8])
        self.assertEqual(metadata["regime_ids"][0], "R_high_match_low_entropy_high_accept_high_load")
        self.assertEqual(metadata["regime_ids"][1], "R_low_match_low_entropy_low_accept_high_load")
        self.assertEqual(metadata["preferred_budgets"], [16, 0, 8])
        self.assertFalse(any(metadata["strategy_bank_hits"]))
        self.assertGreater(
            metadata["expected_gain_per_node"][0],
            metadata["expected_gain_per_node"][2],
        )

    def test_strategy_bank_observer_updates_gain_and_negative_counts(self):
        module = load_budget_module()

        class Req:
            triepilot_negative_gain_count = 0

        reqs = [Req(), Req()]
        bank = module.TriePilotStrategyBank(max_budget=8)
        metadata = {
            "regime_ids": ["R_high_match_low_entropy_high_accept_low_load", "R_low_match_low_entropy_low_accept_low_load"],
            "strategy_bank_hits": [False, False],
        }

        module.observe_triepilot_strategy_feedback(
            reqs,
            strategy_bank=bank,
            allocation_metadata=metadata,
            requested_draft_budgets=[8, 4],
            accept_lens=[4, 0],
            alpha=0.5,
        )

        self.assertGreater(
            bank.records["R_high_match_low_entropy_high_accept_low_load"].expected_gain_per_node,
            bank.records["R_low_match_low_entropy_low_accept_low_load"].expected_gain_per_node,
        )
        self.assertEqual(reqs[0].triepilot_negative_gain_count, 0)
        self.assertEqual(reqs[1].triepilot_negative_gain_count, 1)

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
                allocation_metadata={
                    "regime_ids": ["R_medium_match_low_entropy_high_accept_low_load", "R_low_match_low_entropy_low_accept_low_load"],
                    "strategy_bank_hits": [True, False],
                    "expected_gain_per_node": [0.7, 0.1],
                    "preferred_budgets": [8, 0],
                    "strategy_confidences": [0.8, 0.2],
                    "exploration_flags": [False, True],
                },
                controller_time_ns=7_000,
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
        self.assertEqual(
            event["regime_ids"],
            ["R_medium_match_low_entropy_high_accept_low_load", "R_low_match_low_entropy_low_accept_low_load"],
        )
        self.assertEqual(event["strategy_bank_hits"], [True, False])
        self.assertEqual(event["strategy_bank_hit_rate"], 0.5)
        self.assertEqual(event["expected_gain_per_node"], [0.7, 0.1])
        self.assertEqual(event["preferred_budgets"], [8, 0])
        self.assertEqual(event["exploration_flags"], [False, True])
        self.assertEqual(event["controller_time_us"], 7.0)


if __name__ == "__main__":
    unittest.main()
