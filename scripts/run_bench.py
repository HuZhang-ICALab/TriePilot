from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SGLang/vLLM bench_serving job.")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--conda-bin", default="/root/anaconda3/bin/conda")
    parser.add_argument("--conda-env", default="sglang")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="30000")
    parser.add_argument("--model", default="/root/sglang_flex_test/models/Qwen/Qwen3-8B")
    parser.add_argument("--dataset-name", default="random")
    parser.add_argument("--dataset-path")
    parser.add_argument("--num-prompts", type=int, default=320)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--request-rate", type=float, default=32)
    parser.add_argument("--random-input-len", type=int, default=512)
    parser.add_argument("--random-output-len", type=int, default=256)
    parser.add_argument("--random-range-ratio", type=float, default=0.5)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.conda_bin,
        "run",
        "-n",
        args.conda_env,
        "python",
        "-m",
        "sglang.bench_serving",
        "--backend",
        args.backend,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--dataset-name",
        args.dataset_name,
        "--num-prompts",
        str(args.num_prompts),
        "--max-concurrency",
        str(args.max_concurrency),
        "--request-rate",
        str(args.request_rate),
        "--output-file",
        args.output_file,
        "--output-details",
    ]
    if args.dataset_path:
        cmd.extend(["--dataset-path", args.dataset_path])
    if args.dataset_name == "random":
        cmd.extend(
            [
                "--random-input-len",
                str(args.random_input_len),
                "--random-output-len",
                str(args.random_output_len),
                "--random-range-ratio",
                str(args.random_range_ratio),
            ]
        )
    return cmd


def main() -> None:
    args = parse_args()
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args)
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
