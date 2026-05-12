# Dataset Source Status

This note tracks the first-pass TriePilot dataset landing on the A100 host.
All generated files live under `/root/TriePilot/data`; existing model and
FlexKV dataset directories are read-only inputs.

## Normalized Main Sources

| Dataset | Rows | Normalized file | Source |
| --- | ---: | --- | --- |
| InstructCoder | 1000 | `/root/TriePilot/data/normalized/instructcoder.jsonl` | `likaixin/InstructCoder` validation JSON |
| JSON/tool-call | 1000 | `/root/TriePilot/data/normalized/json_tool.jsonl` | JSONSchemaBench aggregate test parquet |
| ShareGPT-style chat | 1000 | `/root/TriePilot/data/normalized/sharegpt.jsonl` | Existing FlexKV RAG-derived ShareGPT-format file |
| GSM8K | 1000 | `/root/TriePilot/data/normalized/gsm8k.jsonl` | OpenAI grade-school-math test JSONL |
| CNN/DailyMail | 1000 | `/root/TriePilot/data/normalized/cnn_dailymail.jsonl` | CNN/DailyMail 3.0.0 test parquet |
| random | 1000 | `/root/TriePilot/data/normalized/random.jsonl` | Synthetic TriePilot generator |
| shared_prefix | 1000 | `/root/TriePilot/data/normalized/shared_prefix.jsonl` | Synthetic TriePilot generator |

## Extra Local Candidates

| Dataset | Rows | Normalized file | Source |
| --- | ---: | --- | --- |
| shared_prefix_flexkv | 100 | `/root/TriePilot/data/normalized/shared_prefix_flexkv.jsonl` | Existing exact-prefix FlexKV file |
| rag_qa_hotpot | 500 | `/root/TriePilot/data/normalized/rag_qa_hotpot.jsonl` | Existing HotpotQA-style FlexKV file |
| rag_qa_pubmed | 500 | `/root/TriePilot/data/normalized/rag_qa_pubmed.jsonl` | Existing PubMed-style FlexKV file |

## Notes

- The A100 server cannot reach Hugging Face directly, so external files were
  downloaded locally and copied to `/root/TriePilot/data/raw`.
- Do not describe the FlexKV ShareGPT-format source as the original ShareGPT
  distribution. It is a chat-shaped RAG workload candidate.
- `configs/dataset/source_manifest.yaml` records raw paths, normalized paths,
  sample id paths, row counts, and URLs.
