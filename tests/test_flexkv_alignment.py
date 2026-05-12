import importlib.util
import sys
import unittest


class FlexKvAlignmentTest(unittest.TestCase):
    def test_python_minor_matches_flexkv(self):
        if sys.version_info[:2] != (3, 10):
            self.skipTest("Python 3.10 alignment is enforced in the A100 sglang env")
        self.assertEqual(sys.version_info[:2], (3, 10))

    def test_installed_sglang_version_matches_a100_baseline(self):
        if importlib.util.find_spec("sglang") is None:
            self.skipTest("Installed sglang is only required in the A100 sglang env")
        import sglang

        self.assertEqual(sglang.__version__, "0.5.6.post2")


if __name__ == "__main__":
    unittest.main()
