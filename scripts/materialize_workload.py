from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triepilot.workloads.materialize import materialize_workload_to_sharegpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize TriePilot workload rows into a ShareGPT-style JSON file."
    )
    parser.add_argument("--workload", required=True)
    parser.add_argument("--normalized-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--tokenizer-model")
    parser.add_argument("--context-len", type=int)
    parser.add_argument("--fixed-output-len", type=int, default=0)
    parser.add_argument("--fill-filtered", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workload_path = Path(args.workload)
    rows = [
        json.loads(line)
        for line in workload_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompt_token_counter = None
    max_prompt_tokens = None
    if args.tokenizer_model:
        if args.context_len is None:
            raise SystemExit("--context-len is required with --tokenizer-model")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_model,
            trust_remote_code=True,
        )
        prompt_token_counter = lambda text: len(tokenizer.encode(text))
        max_prompt_tokens = args.context_len - max(args.fixed_output_len, 0)
        if max_prompt_tokens < 2:
            raise SystemExit("--context-len must exceed --fixed-output-len by at least 2")

    materialized = materialize_workload_to_sharegpt(
        rows,
        normalized_dir=args.normalized_dir,
        output_path=args.output,
        max_rows=args.max_rows,
        prompt_token_counter=prompt_token_counter,
        max_prompt_tokens=max_prompt_tokens,
        fill_filtered=args.fill_filtered,
    )
    print(f"materialized {len(materialized)} requests to {args.output}")


if __name__ == "__main__":
    main()
