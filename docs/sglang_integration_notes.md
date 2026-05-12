# SGLang Integration Notes

Reference source:

- `third_party/sglang_flex`
- SGLang version: `0.5.6.post2`
- FlexKV serving env: `sglang-flex`, Python `3.10.19`

## Observations

1. `ServerArgs._handle_speculative_decoding` treats `NGRAM` as a launch-time
   mode. It disables overlap scheduling and mixed chunked prefill, and fills
   missing `speculative_num_draft_tokens` from the max match window.

2. Runtime `set_internal_state` only allows a narrow set of fields:
   `pp_max_micro_batch_size`, `speculative_accept_threshold_single`, and
   `speculative_accept_threshold_acc`. NGRAM budget fields cannot currently be
   changed through the public internal-state endpoint.

3. `NGRAMWorker.__init__` fixes these values at construction:
   `draft_token_num`, `branch_length`, `max_match_window_size`, and the
   `NgramCache` parameters. It also preallocates tensors sized by
   `max_batch_size * draft_token_num`.

4. `NGRAMWorker.forward_batch_generation` returns `num_accepted_tokens` and
   `accept_lens`, which are the first metrics to use for TriePilot logging.

5. `NgramVerifyInput.verify` updates per-request counters:
   `req.spec_verify_ct` and `req.spec_accepted_tokens`. These can be used for
   request-level acceptance statistics.

## Implications

- Static baselines are straightforward: launch separate servers with fixed
  NGRAM arguments.
- Dynamic off/tiny/small inside one server is not available as a config-only
  change. It needs an SGLang patch.
- The conservative first patch should not resize tensors per step. Preallocate
  for the largest local tier and mask or slice smaller active budgets.
- The first logging patch can be read-only: record current tier, batch size,
  acceptance length, sequence lengths, and controller overhead without changing
  verification behavior.

## First Patch Plan

1. Add TriePilot recorder/config args to `ServerArgs`.
2. Add a small `triepilot` package under
   `python/sglang/srt/speculative/triepilot`.
3. Instantiate a recorder in `NGRAMWorker`.
4. Emit one JSONL event after verification in
   `NGRAMWorker.forward_batch_generation`.
5. Validate that static NGRAM logs `accept_lens` and total accepted tokens.
6. Only after logging works, add tier selection.

