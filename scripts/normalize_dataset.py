from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triepilot.data.normalizers import (  # noqa: E402
    NORMALIZERS,
    load_records,
    normalize_records,
    write_normalized_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a dataset into TriePilot JSONL.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--normalizer", choices=sorted(NORMALIZERS), required=True)
    parser.add_argument("--input", help="Local JSON or JSONL source path.")
    parser.add_argument("--input-format", choices=["auto", "json", "parquet"], default="auto")
    parser.add_argument("--hf-name", help="Hugging Face dataset name.")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-ids", required=True)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--missing-manifest")
    return parser.parse_args()


def _load_hf_records(args: argparse.Namespace) -> list[dict[str, object]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(f"datasets package unavailable: {exc}") from exc

    split = args.hf_split
    if args.limit is not None:
        split = f"{split}[:{args.limit}]"
    dataset = load_dataset(args.hf_name, args.hf_config, split=split)
    return [dict(row) for row in dataset]


def _write_missing_manifest(args: argparse.Namespace, error: Exception) -> None:
    if not args.missing_manifest:
        return
    path = Path(args.missing_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "normalizer": args.normalizer,
        "input": args.input,
        "hf_name": args.hf_name,
        "hf_config": args.hf_config,
        "hf_split": args.hf_split,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    try:
        if args.input:
            records = load_records(args.input, args.input_format)
            source = args.input
        elif args.hf_name:
            records = _load_hf_records(args)
            source = f"hf:{args.hf_name}/{args.hf_config or 'default'}:{args.hf_split}"
        else:
            raise SystemExit("Either --input or --hf-name is required")

        rows = normalize_records(
            records,
            dataset=args.dataset,
            seed=args.seed,
            normalizer=NORMALIZERS[args.normalizer],
            limit=args.limit,
        )
        write_normalized_dataset(
            args.output,
            args.sample_ids,
            rows,
            dataset=args.dataset,
            seed=args.seed,
            source=source,
        )
        print(f"wrote {len(rows)} rows to {args.output}")
    except Exception as exc:
        _write_missing_manifest(args, exc)
        raise


if __name__ == "__main__":
    main()
