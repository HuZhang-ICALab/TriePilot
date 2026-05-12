import unittest

from triepilot.data.synthetic import build_random_prompts, build_shared_prefix_prompts


class SyntheticDataTest(unittest.TestCase):
    def test_random_prompts_are_deterministic_and_bounded(self):
        rows = build_random_prompts(count=2, seed=9, input_tokens=8, output_tokens=4)

        self.assertEqual(rows, build_random_prompts(count=2, seed=9, input_tokens=8, output_tokens=4))
        self.assertEqual(rows[0]["dataset"], "random")
        self.assertEqual(rows[0]["max_new_tokens"], 4)
        self.assertEqual(len(rows[0]["prompt"].split()), 8)

    def test_shared_prefix_prompts_reuse_prefix(self):
        rows = build_shared_prefix_prompts(count=3, seed=4, prefix_tokens=5, suffix_tokens=3, output_tokens=6)

        prefix = rows[0]["prompt"].split()[:5]
        self.assertTrue(all(row["prompt"].split()[:5] == prefix for row in rows))
        self.assertEqual({row["dataset"] for row in rows}, {"shared_prefix"})
        self.assertEqual({row["max_new_tokens"] for row in rows}, {6})


if __name__ == "__main__":
    unittest.main()
