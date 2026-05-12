from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from triepilot.controller.features import ControllerFeatures
from triepilot.replay.offline_oracle import label_event

FEATURES = [
    "batch_size",
    "kv_usage",
    "seq_len_mean",
    "recent_accept_ema",
    "mean_match_depth",
    "request_rate",
    "slo_violation_ema",
]


def load_xy(path: str | Path) -> tuple[list[list[float]], list[str]]:
    x_rows: list[list[float]] = []
    labels: list[str] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            features = ControllerFeatures.from_mapping(row).to_dict()
            x_rows.append([float(features[name]) for name in FEATURES])
            labels.append(label_event(row))
    return x_rows, labels


def distill_text(path: str | Path, max_depth: int = 3) -> str:
    x_rows, labels = load_xy(path)
    if not x_rows:
        return "empty trace"

    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
    except Exception:
        counts = Counter(labels)
        majority = counts.most_common(1)[0][0]
        return f"sklearn unavailable; majority_label={majority}; counts={dict(counts)}"

    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=2)
    clf.fit(x_rows, labels)
    return export_text(clf, feature_names=FEATURES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--output")
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()

    text = distill_text(args.trace, max_depth=args.max_depth)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

