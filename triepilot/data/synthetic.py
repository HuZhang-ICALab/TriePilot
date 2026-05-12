from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


_VOCAB = tuple(f"tok{i:04d}" for i in range(4096))


def _tokens(rng: random.Random, count: int) -> list[str]:
    return [rng.choice(_VOCAB) for _ in range(count)]


def _row(dataset: str, sample_id: str, prompt: str, max_new_tokens: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
    }


def build_random_prompts(
    count: int,
    seed: int,
    input_tokens: int,
    output_tokens: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    return [
        _row(
            "random",
            f"random-{seed}-{index:06d}",
            " ".join(_tokens(rng, input_tokens)),
            output_tokens,
        )
        for index in range(count)
    ]


def build_shared_prefix_prompts(
    count: int,
    seed: int,
    prefix_tokens: int,
    suffix_tokens: int,
    output_tokens: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    prefix = _tokens(rng, prefix_tokens)
    rows: list[dict[str, object]] = []
    for index in range(count):
        suffix = _tokens(rng, suffix_tokens)
        rows.append(
            _row(
                "shared_prefix",
                f"shared-prefix-{seed}-{index:06d}",
                " ".join(prefix + suffix),
                output_tokens,
            )
        )
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic TriePilot datasets.")
    parser.add_argument("--kind", choices=["random", "shared_prefix"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--prefix-tokens", type=int, default=384)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.kind == "random":
        rows = build_random_prompts(
            count=args.count,
            seed=args.seed,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
        )
    else:
        rows = build_shared_prefix_prompts(
            count=args.count,
            seed=args.seed,
            prefix_tokens=args.prefix_tokens,
            suffix_tokens=args.suffix_tokens,
            output_tokens=args.output_tokens,
        )
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
