import unittest

from triepilot.experiments.registry import (
    BASELINE_METHODS,
    DATASET_PROFILES,
    MODEL_PROFILES,
    required_baseline_names,
)


class ExperimentRegistryTest(unittest.TestCase):
    def test_required_baselines_match_agents_section_7(self):
        expected = {
            "ar_no_spec",
            "sglang_ngram_default",
            "static_ngram_tiers",
            "best_static_per_workload",
            "batch_global_ema",
            "batch_global_cost_aware",
            "bandit_global_tier",
            "equal_budget_allocation",
            "random_budget_allocation",
            "match_depth_greedy",
            "accept_ema_greedy",
            "triepilot_allocation",
            "oracle_allocation",
        }

        self.assertEqual(required_baseline_names(), expected)
        self.assertTrue(all(name in BASELINE_METHODS for name in expected))

    def test_model_profiles_stay_in_8b_class_for_first_pass(self):
        self.assertEqual(
            {profile["size_class"] for profile in MODEL_PROFILES.values()},
            {"7b_8b"},
        )

    def test_dataset_profiles_cover_main_workload_families(self):
        self.assertEqual(
            {profile["family"] for profile in DATASET_PROFILES.values()},
            {
                "code",
                "json_tool",
                "chat",
                "math",
                "summarization",
                "negative_control",
                "shared_prefix",
            },
        )


if __name__ == "__main__":
    unittest.main()
