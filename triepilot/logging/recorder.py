from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TriePilotRecorder:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: dict[str, Any]) -> None:
        if self.fh is None:
            return
        row = dict(event)
        row["time_ns"] = time.time_ns()
        self.fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None

    def __enter__(self) -> "TriePilotRecorder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

