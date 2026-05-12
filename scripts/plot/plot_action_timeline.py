from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--output", default="experiments/figures/action_timeline.png")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib unavailable: {exc}") from exc

    rows = [
        json.loads(line)
        for line in Path(args.trace).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tiers = ["off", "tiny", "small", "medium", "large"]
    tier_to_y = {name: i for i, name in enumerate(tiers)}
    x = [row["step_id"] for row in rows]
    y = [tier_to_y.get(row.get("tier", "off"), 0) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.step(x, y, where="post")
    ax.set_yticks(range(len(tiers)), tiers)
    ax.set_xlabel("decode step")
    ax.set_ylabel("tier")
    ax.grid(True, axis="y", alpha=0.25)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()

