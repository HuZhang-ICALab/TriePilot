from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    version_py = root / "third_party" / "sglang_flex" / "python" / "sglang" / "version.py"
    version_ns: dict[str, str] = {}
    exec(version_py.read_text(encoding="utf-8"), version_ns)
    print(f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"sglang={version_ns.get('__version__')}")
    print(f"sglang_version_file={version_py}")


if __name__ == "__main__":
    main()

