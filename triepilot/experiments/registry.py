from __future__ import annotations

from typing import Any


MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "qwen2_5_7b_instruct": {
        "provider": "Qwen",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "size_class": "7b_8b",
        "role": "main",
        "remote_path_candidates": [
            "/root/sglang_flex_test/models/Qwen/Qwen2.5-7B-Instruct",
        ],
    },
    "qwen3_8b": {
        "provider": "Qwen",
        "model_id": "Qwen/Qwen3-8B",
        "size_class": "7b_8b",
        "role": "main",
        "remote_path_candidates": [
            "/root/sglang_flex_test/models/Qwen/Qwen3-8B",
        ],
    },
    "llama3_1_8b_instruct": {
        "provider": "Meta",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "size_class": "7b_8b",
        "role": "main",
        "remote_path_candidates": [
            "/root/sglang_flex_test/models/AI-ModelScope/Llama-3.1-8B-Instruct",
            "/root/sglang_flex_test/models/AI-ModelScope/Llama-3___1-8B-Instruct",
        ],
    },
    "qwen2_5_coder_7b_instruct": {
        "provider": "Qwen",
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "size_class": "7b_8b",
        "role": "code",
        "remote_path_candidates": [
            "/root/sglang_flex_test/models/Qwen/Qwen2.5-Coder-7B-Instruct",
        ],
    },
}


DATASET_PROFILES: dict[str, dict[str, Any]] = {
    "instructcoder": {
        "family": "code",
        "source": "external_or_user_supplied",
        "normalized_path": "data/normalized/instructcoder.jsonl",
    },
    "json_tool": {
        "family": "json_tool",
        "source": "BFCL_or_JSONSchemaBench",
        "normalized_path": "data/normalized/json_tool.jsonl",
    },
    "sharegpt": {
        "family": "chat",
        "source": "ShareGPT_or_existing_sharegpt_json",
        "normalized_path": "data/normalized/sharegpt.jsonl",
    },
    "gsm8k": {
        "family": "math",
        "source": "GSM8K_or_MATH500",
        "normalized_path": "data/normalized/gsm8k.jsonl",
    },
    "cnn_dailymail": {
        "family": "summarization",
        "source": "CNN/DailyMail",
        "normalized_path": "data/normalized/cnn_dailymail.jsonl",
    },
    "random": {
        "family": "negative_control",
        "source": "synthetic",
        "normalized_path": "data/normalized/random.jsonl",
    },
    "shared_prefix": {
        "family": "shared_prefix",
        "source": "synthetic_or_sglang_generated",
        "normalized_path": "data/normalized/shared_prefix.jsonl",
    },
}


BASELINE_METHODS: dict[str, dict[str, Any]] = {
    "ar_no_spec": {
        "category": "serving",
        "deployable": True,
        "description": "Autoregressive decoding without speculation.",
    },
    "sglang_ngram_default": {
        "category": "static_ngram",
        "deployable": True,
        "description": "SGLang 0.5.6 default NGRAM speculative decoding.",
    },
    "static_ngram_tiers": {
        "category": "static_ngram",
        "deployable": True,
        "description": "Fixed NGRAM budget tiers: 0, 2, 4, 8, 16, 24, 32.",
    },
    "best_static_per_workload": {
        "category": "offline_selection",
        "deployable": False,
        "description": "Post-hoc best static tier for each workload.",
    },
    "batch_global_ema": {
        "category": "batch_global",
        "deployable": True,
        "description": "Batch-level tier selected from accepted-token EMA.",
    },
    "batch_global_cost_aware": {
        "category": "batch_global",
        "deployable": True,
        "description": "Batch-level tier selected by accepted tokens per verified node.",
    },
    "bandit_global_tier": {
        "category": "batch_global",
        "deployable": True,
        "description": "Bandit-style global tier selector.",
    },
    "equal_budget_allocation": {
        "category": "request_allocation",
        "deployable": True,
        "description": "Split B_batch evenly across active requests.",
    },
    "random_budget_allocation": {
        "category": "request_allocation",
        "deployable": True,
        "description": "Random per-request budget assignment under B_batch.",
    },
    "match_depth_greedy": {
        "category": "request_allocation",
        "deployable": True,
        "description": "Greedy allocation ranked by symbolic match depth.",
    },
    "accept_ema_greedy": {
        "category": "request_allocation",
        "deployable": True,
        "description": "Greedy allocation ranked by recent accepted length EMA.",
    },
    "triepilot_allocation": {
        "category": "request_allocation",
        "deployable": True,
        "description": "Probability-free regime and Strategy Bank allocation.",
    },
    "oracle_allocation": {
        "category": "offline_upper_bound",
        "deployable": False,
        "description": "Replay oracle allocation under the same B_batch constraint.",
    },
}


def required_baseline_names() -> set[str]:
    return set(BASELINE_METHODS)
