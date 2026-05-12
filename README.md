# TriePilot

TriePilot is a development scaffold for adaptive NGRAM/Trie speculative
decoding experiments.

The local RTX 3050 machine is used for engineering smoke tests,
instrumentation, greedy correctness checks, and offline replay. Main
performance claims, profiles, and paper figures should be regenerated on the
A100 platform.

## Current Local Notes

- This workspace starts on Windows under `E:\TriePilot`.
- SGLang's server path is expected to be more reliable on Linux/WSL or the
  A100 host than on native Windows.
- Match FlexKV's serving environment: Python 3.10.x, target 3.10.19.
- Match FlexKV's SGLang source version: 0.5.6.post2.
- The system Python 3.14 is not a good target for the SGLang/PyTorch stack.
- Python and conda install scripts in `scripts/env` default to domestic mirrors.

## First Milestones

1. Create the Python 3.10.19 environment.
2. Run the offline controller smoke test.
3. Use the FlexKV-aligned SGLang source under `third_party/sglang_flex`.
4. Run local 0.5B no-spec and NGRAM smoke tests once Linux/WSL/A100 is ready.
5. Add SGLang instrumentation hooks after the standalone controller tests pass.
