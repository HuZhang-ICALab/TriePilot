from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach run metadata to raw bench output.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--framework", default="sglang")
    return parser.parse_args()


def iter_rows(path: Path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return
    if text.startswith("["):
        for row in json.loads(text):
            yield row
        return
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": args.run_id,
        "method": args.method,
        "model": args.model,
        "dataset": args.dataset,
        "framework": args.framework,
    }
    with output.open("w", encoding="utf-8") as handle:
        for step_id, row in enumerate(iter_rows(Path(args.input)) or []):
            merged = dict(row)
            merged.update(metadata)
            merged.setdefault("step_id", step_id)
            handle.write(json.dumps(merged, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
