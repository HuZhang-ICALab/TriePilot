from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triepilot.data.synthetic import (
    build_random_prompts,
    build_shared_prefix_prompts,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TriePilot synthetic datasets.")
    parser.add_argument("--output-dir", default="/root/TriePilot/data/normalized")
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--random-input-tokens", type=int, default=512)
    parser.add_argument("--random-output-tokens", type=int, default=256)
    parser.add_argument("--shared-prefix-tokens", type=int, default=384)
    parser.add_argument("--shared-suffix-tokens", type=int, default=128)
    parser.add_argument("--shared-output-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random_rows = build_random_prompts(
        count=args.count,
        seed=args.seed,
        input_tokens=args.random_input_tokens,
        output_tokens=args.random_output_tokens,
    )
    shared_rows = build_shared_prefix_prompts(
        count=args.count,
        seed=args.seed,
        prefix_tokens=args.shared_prefix_tokens,
        suffix_tokens=args.shared_suffix_tokens,
        output_tokens=args.shared_output_tokens,
    )
    write_jsonl(f"{args.output_dir}/random.jsonl", random_rows)
    write_jsonl(f"{args.output_dir}/shared_prefix.jsonl", shared_rows)


if __name__ == "__main__":
    main()
