import unittest

from triepilot.controller import (
    ControllerFeatures,
    TrieRulePolicy,
    load_tiers_from_mapping,
)


class RulePolicyTest(unittest.TestCase):
    def setUp(self):
        self.tiers = load_tiers_from_mapping(
            {
                "tiers": {
                    "off": {"enabled": False, "draft_tokens": 0},
                    "tiny": {
                        "enabled": True,
                        "draft_tokens": 2,
                        "max_match_window_size": 4,
                        "max_bfs_breadth": 1,
                    },
                    "small": {
                        "enabled": True,
                        "draft_tokens": 4,
                        "max_match_window_size": 8,
                        "max_bfs_breadth": 2,
                    },
                }
            }
        )
        self.policy = TrieRulePolicy(self.tiers)

    def test_high_kv_turns_off(self):
        tier = self.policy.select(ControllerFeatures(kv_usage=0.90))
        self.assertEqual(tier.name, "off")

    def test_good_acceptance_uses_small(self):
        tier = self.policy.select(
            ControllerFeatures(recent_accept_ema=1.0, mean_match_depth=3)
        )
        self.assertEqual(tier.name, "small")

    def test_low_signal_turns_off(self):
        tier = self.policy.select(
            ControllerFeatures(recent_accept_ema=0.1, mean_match_depth=1)
        )
        self.assertEqual(tier.name, "off")


if __name__ == "__main__":
    unittest.main()

