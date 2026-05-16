from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer
from vllm.v1.spec_decode.ngram_proposer import (
    _find_longest_matched_ngram_and_propose_tokens,
)

WORKSPACE = Path(os.environ.get("WORKSPACE", "/root/TriePilot"))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from triepilot.sglang_integration import budget as triepilot_budget

RUN_ID = os.environ.get(
    "RUN_ID", "20260516_session10_vllm_ngram_portability_replay_seed20260520_22"
)
RUN_DIR = Path(os.environ.get("RUN_DIR", str(WORKSPACE / "runs" / RUN_ID)))
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/root/sglang_flex_test/models/Qwen/Qwen3-8B"
)
DATA_DIR = WORKSPACE / "data" / "normalized"
SCENARIOS = [
    ("instructcoder", "gsm8k"),
    ("cnn_dailymail", "random"),
    ("json_tool", "sharegpt"),
]
SEEDS = [20260520, 20260521, 20260522]
B_BATCH_VALUES = [100, 160]
METHODS = [
    "batch_global_budget2",
    "batch_global_budget8",
    "equal_budget_allocation",
    "match_depth_greedy",
    "accept_ema_greedy",
    "triepilot_allocation",
]
NUM_PROMPTS = 64
BATCH_SIZE = 8
LEFT_RATIO = 0.5
MIN_N = 2
MAX_N = 12
MAX_DRAFT_TOKENS = 16
MAX_MODEL_LEN = 4096
CUT_FRACTIONS = [0.50, 0.65, 0.80, 0.90]
ACCEPT_EMA_ALPHA = 0.20

RAW_EVENTS = RUN_DIR / "vllm_ngram_replay_events.jsonl"
SUMMARY_CSV = RUN_DIR / "session10_vllm_ngram_portability_summary.csv"
STATE_CSV = RUN_DIR / "session10_vllm_ngram_state_summary.csv"
INTEGRITY_JSON = RUN_DIR / "integrity_session10_vllm_ngram_portability.json"
ANALYSIS_MD = RUN_DIR / "analysis_session10_vllm_ngram_portability.md"
CONFIG_YAML = RUN_DIR / "config.yaml"
ENV_JSON = RUN_DIR / "env.json"
GIT_COMMIT = RUN_DIR / "git_commit.txt"
GPU_MEMORY = RUN_DIR / "gpu_memory.txt"


@dataclass
class ReplayReq:
    request_id: str
    triepilot_accept_len_ema: float = 0.0
    triepilot_negative_gain_count: int = 0
    triepilot_request_local_probe_count: int = 0


def shell_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def load_rows(dataset: str) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{dataset}.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty dataset: {path}")
    return rows


def mix_rows(left: str, right: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed * 131 + sum(ord(ch) for ch in f"{left}:{right}"))
    left_rows = load_rows(left)
    right_rows = load_rows(right)
    left_count = int(round(NUM_PROMPTS * LEFT_RATIO))
    right_count = NUM_PROMPTS - left_count
    chosen: list[dict[str, Any]] = []
    for dataset, rows, count in [(left, left_rows, left_count), (right, right_rows, right_count)]:
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        for row_index in indices[:count]:
            row = dict(rows[row_index])
            row["source_dataset"] = dataset
            chosen.append(row)
    rng.shuffle(chosen)
    return chosen


def prompt_from_row(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(
            str(msg.get("content", "")) for msg in messages if isinstance(msg, dict)
        )
    return ""


def brute_match_depth(tokens: list[int], min_n: int, max_n: int) -> int:
    total = len(tokens)
    upper = min(max_n, total - 1)
    for ngram_len in range(upper, min_n - 1, -1):
        suffix = tokens[-ngram_len:]
        for start in range(0, total - ngram_len):
            if tokens[start : start + ngram_len] == suffix:
                return ngram_len
    return 0


def lcp_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a_token, b_token in zip(left, right):
        if a_token != b_token:
            break
        count += 1
    return count


def vllm_propose(context: list[int]) -> list[int]:
    arr = np.asarray(context, dtype=np.int32)
    out = _find_longest_matched_ngram_and_propose_tokens(
        arr, MIN_N, MAX_N, MAX_MODEL_LEN, MAX_DRAFT_TOKENS
    )
    return [int(token) for token in out.tolist()]


def make_state_records(tokenizer: Any) -> dict[tuple[int, str, str], list[dict[str, Any]]]:
    records: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for seed in SEEDS:
        for left, right in SCENARIOS:
            states: list[dict[str, Any]] = []
            for req_idx, row in enumerate(mix_rows(left, right, seed)):
                prompt = prompt_from_row(row)
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(token_ids) > MAX_MODEL_LEN:
                    token_ids = token_ids[:MAX_MODEL_LEN]
                if len(token_ids) < MIN_N + 4:
                    continue
                request_id = str(row.get("sample_id") or f"{row['source_dataset']}_{req_idx}")
                for round_idx, frac in enumerate(CUT_FRACTIONS):
                    cut = max(MIN_N, min(len(token_ids) - 1, int(len(token_ids) * frac)))
                    context = token_ids[:cut]
                    truth = token_ids[cut : cut + MAX_DRAFT_TOKENS]
                    proposal = vllm_propose(context)
                    match_depth = brute_match_depth(context, MIN_N, MAX_N)
                    proposal_len = len(proposal)
                    candidate_count = 1 if proposal_len > 0 else 0
                    states.append(
                        {
                            "seed": seed,
                            "scenario": f"{left}+{right}",
                            "left_dataset": left,
                            "right_dataset": right,
                            "request_index": req_idx,
                            "request_id": request_id,
                            "source_dataset": row["source_dataset"],
                            "sample_id": request_id,
                            "round_idx": round_idx,
                            "cut_fraction": frac,
                            "seq_len": cut,
                            "prompt_token_len": len(token_ids),
                            "truth_len": len(truth),
                            "proposal_tokens": proposal,
                            "proposal_len": proposal_len,
                            "proxy_accept_full": lcp_len(proposal, truth),
                            "features": {
                                "match_depth": match_depth,
                                "candidate_count": candidate_count,
                                "branch_entropy": 0.0,
                                "top_branch_ratio": 1.0 if candidate_count else 0.0,
                                "filled_nodes": proposal_len,
                                "seq_len_bucket": int(math.log2(max(cut, 1))),
                            },
                        }
                    )
            states.sort(key=lambda item: (item["round_idx"], item["request_index"]))
            records[(seed, left, right)] = states
    return records


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[idx])


def allocate_batch(
    method: str,
    reqs: list[ReplayReq],
    features: list[dict[str, Any]],
    *,
    b_batch: int,
    step_id: int,
    seed: int,
    strategy_bank: triepilot_budget.TriePilotStrategyBank | None,
) -> tuple[list[int], dict[str, Any], float]:
    start = time.perf_counter_ns()
    if method.startswith("batch_global_budget"):
        value = int(method.replace("batch_global_budget", ""))
        budgets = [value for _ in reqs]
        metadata = {
            "regime_ids": ["batch_global"] * len(reqs),
            "strategy_bank_hits": [False] * len(reqs),
            "expected_gain_per_node": [0.0] * len(reqs),
            "ablation_modes": [],
        }
    else:
        budgets, _, metadata = triepilot_budget.resolve_triepilot_draft_budgets(
            reqs,
            default_budget=MAX_DRAFT_TOKENS,
            allocation_policy=method,
            batch_budget=b_batch,
            structural_features=features,
            step_id=step_id,
            random_seed=seed,
            strategy_bank=strategy_bank,
            return_metadata=True,
        )
    overhead_us = (time.perf_counter_ns() - start) / 1000.0
    return [int(value) for value in budgets], metadata, overhead_us


def replay_method(
    states: list[dict[str, Any]],
    *,
    seed: int,
    left: str,
    right: str,
    method: str,
    b_batch: int,
    raw_fh: Any,
) -> dict[str, Any]:
    req_map = {
        state["request_index"]: ReplayReq(str(state["request_id"])) for state in states
    }
    strategy_bank = (
        triepilot_budget.TriePilotStrategyBank(max_budget=MAX_DRAFT_TOKENS)
        if method == "triepilot_allocation"
        else None
    )
    totals = {
        "requested_budget": 0,
        "actual_draft_tokens": 0,
        "accepted_proxy_tokens": 0,
        "wasted_proxy_tokens": 0,
        "proposal_shortfall_tokens": 0,
        "budget_violation_count": 0,
        "proposal_positive_states": 0,
        "strategy_bank_hits": 0,
        "strategy_bank_decisions": 0,
    }
    overheads: list[float] = []
    match_depths: list[float] = []
    proposal_lens: list[float] = []
    batch_count = 0
    state_count = 0

    by_round: dict[int, list[dict[str, Any]]] = {}
    for state in states:
        by_round.setdefault(int(state["round_idx"]), []).append(state)

    for round_idx in sorted(by_round):
        round_states = sorted(by_round[round_idx], key=lambda item: item["request_index"])
        for offset in range(0, len(round_states), BATCH_SIZE):
            batch_states = round_states[offset : offset + BATCH_SIZE]
            reqs = [req_map[state["request_index"]] for state in batch_states]
            features = [state["features"] for state in batch_states]
            step_id = round_idx * 1000 + offset // BATCH_SIZE
            budgets, metadata, overhead_us = allocate_batch(
                method,
                reqs,
                features,
                b_batch=b_batch,
                step_id=step_id,
                seed=seed,
                strategy_bank=strategy_bank,
            )
            overheads.append(overhead_us)
            requested_sum = sum(budgets)
            if not method.startswith("batch_global_budget") and requested_sum > b_batch:
                totals["budget_violation_count"] += 1

            accepted_lens: list[int] = []
            actual_lens: list[int] = []
            for idx, state in enumerate(batch_states):
                proposal_len = int(state["proposal_len"])
                actual = min(int(budgets[idx]), proposal_len)
                accepted = min(int(state["proxy_accept_full"]), actual)
                actual_lens.append(actual)
                accepted_lens.append(accepted)
                totals["requested_budget"] += int(budgets[idx])
                totals["actual_draft_tokens"] += actual
                totals["accepted_proxy_tokens"] += accepted
                totals["wasted_proxy_tokens"] += max(actual - accepted, 0)
                totals["proposal_shortfall_tokens"] += max(int(budgets[idx]) - actual, 0)
                totals["proposal_positive_states"] += 1 if proposal_len > 0 else 0
                match_depths.append(float(state["features"]["match_depth"]))
                proposal_lens.append(float(proposal_len))

            triepilot_budget.observe_triepilot_accept_lengths(
                reqs, accepted_lens, alpha=ACCEPT_EMA_ALPHA
            )
            if strategy_bank is not None:
                triepilot_budget.observe_triepilot_strategy_feedback(
                    reqs,
                    strategy_bank=strategy_bank,
                    allocation_metadata=metadata,
                    requested_draft_budgets=budgets,
                    accept_lens=accepted_lens,
                    alpha=ACCEPT_EMA_ALPHA,
                    step_id=step_id,
                )

            hits = metadata.get("strategy_bank_hits", []) if isinstance(metadata, dict) else []
            totals["strategy_bank_hits"] += sum(1 for hit in hits if hit)
            totals["strategy_bank_decisions"] += len(hits)
            raw_fh.write(
                json.dumps(
                    {
                        "run_id": RUN_ID,
                        "seed": seed,
                        "scenario": f"{left}+{right}",
                        "method": method,
                        "b_batch": b_batch,
                        "round_idx": round_idx,
                        "batch_index": offset // BATCH_SIZE,
                        "batch_size": len(batch_states),
                        "requested_budgets": budgets,
                        "requested_budget_sum": requested_sum,
                        "actual_draft_tokens": sum(actual_lens),
                        "accepted_proxy_tokens": sum(accepted_lens),
                        "proposal_lens": [
                            int(state["proposal_len"]) for state in batch_states
                        ],
                        "match_depths": [
                            int(state["features"]["match_depth"]) for state in batch_states
                        ],
                        "regime_ids": (
                            metadata.get("regime_ids", [])
                            if isinstance(metadata, dict)
                            else []
                        ),
                        "strategy_bank_hits": hits,
                        "controller_overhead_us": overhead_us,
                        "budget_violation": (
                            not method.startswith("batch_global_budget")
                        )
                        and requested_sum > b_batch,
                        "vllm_proposer": (
                            "vllm.v1.spec_decode.ngram_proposer."
                            "_find_longest_matched_ngram_and_propose_tokens"
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            batch_count += 1
            state_count += len(batch_states)

    actual = totals["actual_draft_tokens"]
    accepted = totals["accepted_proxy_tokens"]
    requested = totals["requested_budget"]
    return {
        "seed": seed,
        "scenario": f"{left}+{right}",
        "left_dataset": left,
        "right_dataset": right,
        "method": method,
        "b_batch": b_batch,
        "num_states": state_count,
        "num_batches": batch_count,
        "requested_budget": requested,
        "actual_draft_tokens": actual,
        "accepted_proxy_tokens": accepted,
        "wasted_proxy_tokens": totals["wasted_proxy_tokens"],
        "proposal_shortfall_tokens": totals["proposal_shortfall_tokens"],
        "proxy_apv": accepted / actual if actual else 0.0,
        "proxy_accept_per_requested": accepted / requested if requested else 0.0,
        "proposal_coverage": (
            totals["proposal_positive_states"] / state_count if state_count else 0.0
        ),
        "mean_match_depth": statistics.mean(match_depths) if match_depths else 0.0,
        "mean_vllm_proposal_len": statistics.mean(proposal_lens) if proposal_lens else 0.0,
        "controller_overhead_mean_us": statistics.mean(overheads) if overheads else 0.0,
        "controller_overhead_p99_us": percentile(overheads, 0.99),
        "strategy_bank_hit_rate": (
            totals["strategy_bank_hits"] / totals["strategy_bank_decisions"]
            if totals["strategy_bank_decisions"]
            else 0.0
        ),
        "budget_violation_count": totals["budget_violation_count"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    before_gpu = shell_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    GIT_COMMIT.write_text(shell_output(["git", "rev-parse", "HEAD"]) + "\n", encoding="utf-8")
    env = {
        "python": sys.version,
        "vllm_version": shell_output(
            [
                "/root/anaconda3/envs/vllm/bin/python",
                "-c",
                "import vllm; print(getattr(vllm, '__version__', 'unknown'))",
            ]
        ),
        "transformers_tokenizer_model": MODEL_PATH,
        "nvidia_smi_before": before_gpu,
    }
    ENV_JSON.write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    CONFIG_YAML.write_text(
        "\n".join(
            [
                f"run_id: {RUN_ID}",
                f"model_path: {MODEL_PATH}",
                f"scenarios: {[f'{a}+{b}' for a, b in SCENARIOS]}",
                f"seeds: {SEEDS}",
                f"b_batch_values: {B_BATCH_VALUES}",
                f"methods: {METHODS}",
                f"num_prompts: {NUM_PROMPTS}",
                f"batch_size: {BATCH_SIZE}",
                f"left_ratio: {LEFT_RATIO}",
                f"min_n: {MIN_N}",
                f"max_n: {MAX_N}",
                f"max_draft_tokens: {MAX_DRAFT_TOKENS}",
                f"cut_fractions: {CUT_FRACTIONS}",
                "note: vLLM NGram trace-driven replay only; no target-model online throughput claim.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    GPU_MEMORY.write_text("before\n" + before_gpu + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    _ = vllm_propose([1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3])
    state_records = make_state_records(tokenizer)

    state_rows: list[dict[str, Any]] = []
    for (seed, left, right), states in state_records.items():
        state_rows.append(
            {
                "seed": seed,
                "scenario": f"{left}+{right}",
                "states": len(states),
                "requests": len({state["request_index"] for state in states}),
                "proposal_coverage": (
                    sum(1 for state in states if state["proposal_len"] > 0) / len(states)
                    if states
                    else 0.0
                ),
                "mean_match_depth": (
                    statistics.mean(float(state["features"]["match_depth"]) for state in states)
                    if states
                    else 0.0
                ),
                "mean_vllm_proposal_len": (
                    statistics.mean(float(state["proposal_len"]) for state in states)
                    if states
                    else 0.0
                ),
                "mean_proxy_accept_full": (
                    statistics.mean(float(state["proxy_accept_full"]) for state in states)
                    if states
                    else 0.0
                ),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    with RAW_EVENTS.open("w", encoding="utf-8") as raw_fh:
        for seed in SEEDS:
            for left, right in SCENARIOS:
                states = state_records[(seed, left, right)]
                for b_batch in B_BATCH_VALUES:
                    for method in METHODS:
                        summary_rows.append(
                            replay_method(
                                states,
                                seed=seed,
                                left=left,
                                right=right,
                                method=method,
                                b_batch=b_batch,
                                raw_fh=raw_fh,
                            )
                        )

    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(STATE_CSV, state_rows)

    expected_rows = len(SEEDS) * len(SCENARIOS) * len(B_BATCH_VALUES) * len(METHODS)
    missing = []
    seen = {(r["seed"], r["scenario"], r["b_batch"], r["method"]) for r in summary_rows}
    for seed in SEEDS:
        for left, right in SCENARIOS:
            for b_batch in B_BATCH_VALUES:
                for method in METHODS:
                    key = (seed, f"{left}+{right}", b_batch, method)
                    if key not in seen:
                        missing.append(key)
    total_budget_violations = sum(int(row["budget_violation_count"]) for row in summary_rows)
    after_gpu = shell_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    residual = shell_output(
        ["bash", "-lc", "pgrep -af 'vllm|sglang.launch_server|bench_serving' || true"]
    )
    with GPU_MEMORY.open("a", encoding="utf-8") as fh:
        fh.write("after\n" + after_gpu + "\n")

    trie_rows = [row for row in summary_rows if row["method"] == "triepilot_allocation"]
    bg8_rows = [row for row in summary_rows if row["method"] == "batch_global_budget8"]
    bg8_by_key = {(r["seed"], r["scenario"], r["b_batch"]): r for r in bg8_rows}
    trie_deltas = []
    for row in trie_rows:
        base = bg8_by_key.get((row["seed"], row["scenario"], row["b_batch"]))
        if base:
            trie_deltas.append(
                {
                    "actual_delta": row["actual_draft_tokens"] - base["actual_draft_tokens"],
                    "apv_delta": row["proxy_apv"] - base["proxy_apv"],
                }
            )
    mean_actual_delta = (
        statistics.mean(delta["actual_delta"] for delta in trie_deltas)
        if trie_deltas
        else 0.0
    )
    mean_apv_delta = (
        statistics.mean(delta["apv_delta"] for delta in trie_deltas)
        if trie_deltas
        else 0.0
    )
    raw_event_count = sum(1 for _ in RAW_EVENTS.open("r", encoding="utf-8"))
    integrity = {
        "run_id": RUN_ID,
        "expected_summary_rows": expected_rows,
        "summary_rows": len(summary_rows),
        "state_rows": len(state_rows),
        "missing": [list(item) for item in missing],
        "budget_violation_count": total_budget_violations,
        "raw_event_count": raw_event_count,
        "vllm_ngram_import": True,
        "tokenizer_class": type(tokenizer).__name__,
        "nvidia_smi_before": before_gpu,
        "nvidia_smi_after": after_gpu,
        "residual_processes": residual.splitlines() if residual else [],
        "triepilot_vs_batch_global8_mean_actual_draft_delta": mean_actual_delta,
        "triepilot_vs_batch_global8_mean_proxy_apv_delta": mean_apv_delta,
        "notes": (
            "Trace-driven replay uses prompt-token continuations as proxy acceptance; "
            "no online target-model throughput claim."
        ),
    }
    INTEGRITY_JSON.write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    method_lines = []
    for method, rows in sorted(by_method.items()):
        method_lines.append(
            f"- {method}: actual_draft_tokens_mean="
            f"{statistics.mean(float(r['actual_draft_tokens']) for r in rows):.1f}, "
            f"proxy_apv_mean={statistics.mean(float(r['proxy_apv']) for r in rows):.4f}, "
            f"controller_p99_us_mean="
            f"{statistics.mean(float(r['controller_overhead_p99_us']) for r in rows):.2f}, "
            f"strategy_bank_hit_rate_mean="
            f"{statistics.mean(float(r['strategy_bank_hit_rate']) for r in rows):.4f}"
        )
    ANALYSIS_MD.write_text(
        "# Session 10 vLLM NGram Portability Replay\n\n"
        "This is a trace-driven portability replay on the A100 host. It imports "
        "vLLM 0.16.0 NGram proposer, tokenizes representative TriePilot workloads, "
        "derives probability-free symbolic features, and feeds them to the same "
        "TriePilot allocator. Prompt-token continuations are used only as proxy "
        "acceptance, so this file must not be cited as online throughput evidence.\n\n"
        f"- summary_rows={len(summary_rows)}/{expected_rows}\n"
        f"- state_rows={len(state_rows)}\n"
        f"- budget_violation_count={total_budget_violations}\n"
        f"- raw_event_count={raw_event_count}\n"
        f"- tokenizer={type(tokenizer).__name__}\n"
        f"- GPU before={before_gpu}; after={after_gpu}\n"
        f"- TriePilot vs batch_global_budget8 mean actual draft delta="
        f"{mean_actual_delta:.2f}\n"
        f"- TriePilot vs batch_global_budget8 mean proxy APV delta="
        f"{mean_apv_delta:.4f}\n\n"
        "Method aggregates:\n"
        + "\n".join(method_lines)
        + "\n\nBoundary: vLLM NGram currently exposes a linear prompt-lookup "
        "proposer, not SGLang tree verification. The replay validates the "
        "allocator/proposer abstraction boundary and budget accounting, not "
        "end-to-end serving speed.\n",
        encoding="utf-8",
    )

    print(json.dumps(integrity, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
