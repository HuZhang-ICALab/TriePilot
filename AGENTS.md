# Agents.md：TriePilot 论文与实验推进指南

先给一个边界确认：**宽泛的 mixed-batch draft-token / token-tree allocation 已经有人做，不能主张“没人做过”；但公开论文中，我没有看到完全同构于“无 draft logits / confidence / speculator probability 的符号式 NGRAM/Trie proposer 下，做 mixed-batch draft-node allocation + strategy reuse”的成果。** TETRIS 已经做 batch speculative decoding 中为每个 request 选择更有希望被接受的 draft tokens；AdaServe 已经做 multi-SLO 下的 token-tree selection，但它明确依赖 draft-model logits / speculator 概率；SGLang 和 TensorRT-LLM 都支持不依赖额外草稿模型的 NGRAM 类 speculative decoding。([ACL Anthology][1])

---

## 0. 项目核心定位

本项目目标是撰写并实现一篇学术论文，暂定题目：

**TriePilot: Probability-Free Draft-Node Budget Allocation for Mixed-Batch Symbolic Speculative Decoding**
中文：**TriePilot：面向混合批符号式投机解码的无概率信号草稿节点预算分配**

不要把本项目写成“一个 SGLang feature”。SGLang 0.5.6 只是主要实现载体。方法本身应抽象为一个框架无关的控制层，适用于 NGRAM、Trie、Prompt Lookup、Suffix/SAM 等不依赖草稿模型 logits 的 symbolic proposer（符号式候选生成器）。

根据原始方案，项目早期想法包含 Fast/Slow、Bandit、Rule、Goodput 触发等内容；现在需要收敛为更稳的论文主线：**在没有 draft logits、confidence scores、speculator probabilities 的情况下，如何把有限 draft-node verification budget 分配给 mixed batch 中的不同请求。**

---

## 1. 不要写错的 claim

禁止主张：

* “我们首次做 mixed-batch speculative budget allocation。”
* “我们首次做动态树预算。”
* “我们首次做 NGRAM/Trie speculative decoding。”
* “我们首次做 Bandit-based speculative decoding。”
* “我们的方法是 SGLang 专用优化。”
* “零开销 zero-cost。”

推荐主张：

* **TriePilot 研究的是 probability-free variant（无概率信号变体）**：已有方法可以使用 draft logits、confidence、speculator probabilities；TriePilot 不使用这些信号。
* **TriePilot 面向 symbolic proposer（符号式候选生成器）**：例如 NGRAM/Trie/Prompt Lookup/Suffix/SAM。
* **TriePilot 解决 mixed-batch draft-node allocation（混合批草稿节点预算分配）**：不是整个 batch 选一个统一 tier，而是在 batch 内给不同请求分配不同草稿节点预算。
* **TriePilot 维护 Strategy Bank（策略库）**：把 regime → allocation strategy 沉淀下来，类似 skills 的“见过相似场景后复用策略”。

论文中应使用：

> To the best of our knowledge, TriePilot is the first system to study mixed-batch draft-node allocation for probability-free symbolic speculative decoding.

中文：

> 据我们所知，TriePilot 是首个系统研究无概率信号符号式投机解码中 mixed-batch draft-node 分配问题的系统。

---

## 2. 方法核心实现思路

### 2.1 基本抽象

每个推理 batch 有一个总草稿节点预算：

```text
B_batch = 本轮 verifier 最多允许验证的 draft nodes 数量
```

每个请求 i 有自己的状态：

```text
x_i = {
  match_depth,
  candidate_count,
  branch_entropy,
  top_branch_ratio,
  filled_nodes,
  accept_len_ema,
  negative_gain_count,
  generated_len,
  batch_size,
  queue_len,
  kv_usage,
  seq_len_bucket
}
```

控制器输出：

```text
b_i = 给请求 i 分配多少个 draft nodes
```

约束：

```text
Σ b_i ≤ B_batch
```

目标：

```text
maximize Σ estimated_gain(x_i, b_i)
```

中文解释：系统资源有限，本轮只能验证有限数量的候选草稿节点。TriePilot 要判断哪些请求更值得投机，把预算优先分给高收益请求。

---

### 2.2 树和缓存的正确理解

不要维护一棵跨所有请求的全局可写草稿树。

第一版设计：

```text
每个请求维护自己的 NGRAM/Trie cache
每个 verification round 临时构造 per-request draft tree
一个 batch 中多棵 per-request draft trees 组成 draft forest
target model 对 draft forest 做 batched verification
```

中文：

* 历史缓存是 per-request 的。
* 本轮草稿树是 per-request 临时构造的。
* batch 级别看到的是 draft forest（草稿森林）。
* Strategy Bank 不存树，只存策略。

如果未来扩展 global corpus trie，只能作为只读外部语料结构，并需要处理隐私、多租户隔离和公平性问题。第一篇不做全局可写树。

---

### 2.3 核心模块

#### A. SymbolicProposer

输入：

```text
request 当前 suffix
node budget b_i
tree shape preference
```

输出：

```text
bounded draft tree
structural features
```

适配对象：

```text
SGLang NGRAM
vLLM N-Gram
TensorRT-LLM NGram
Prompt Lookup
Suffix/SAM proposer
```

#### B. Regime Encoder

作用：把当前请求状态编码成 regime id。

例子：

```text
high_match + low_entropy + high_accept + low_load → R_deep_repeat
low_match + low_accept + high_load → R_low_value
medium_match + high_branch + medium_load → R_wide_branch
```

Regime Encoder 只负责识别“当前像什么场景”，不直接选择预算。

#### C. Strategy Bank

作用：保存每个 regime 的历史策略。

每条 skill 结构：

```text
regime_id
preferred_budget
safe_budget_set
tree_shape_preference
expected_gain_per_node
fallback_budget
confidence
exploration_allowed
last_update_time
```

Strategy Bank 负责回答：“这个 regime 以前学到过什么预算策略？”

#### D. Budget Allocator

输入：

```text
B_batch
每个请求的 regime
Strategy Bank 返回的 expected_gain_per_node
```

输出：

```text
每个请求的 b_i
```

第一版使用 greedy allocator：

```text
按 expected_gain_per_node 从高到低分配 budget
直到 Σ b_i ≤ B_batch
```

第二版可以做 marginal upgrade：

```text
off → tiny → small → medium → large
每次选择单位 node 收益最高的升级
```

#### E. Telemetry Collector

每轮记录：

```text
request_id
batch_id
step_id
method
dataset
regime_id
allocated_budget
actual_draft_nodes
verified_nodes
accepted_tokens
wasted_nodes
step_latency
controller_time
ngram_query_time
verify_time
TPOT
ITL
TTFT
kv_usage
queue_len
batch_size
match_depth
branch_entropy
accept_len_ema
strategy_bank_hit
```

#### F. Slow Explorer

仅在以下情况触发：

```text
unknown regime
actual_goodput << expected_goodput
negative_speedup 连续出现
Strategy Bank confidence 下降
```

Slow Explorer 可以使用：

```text
safe exploration
Bandit-style exploration
budget sweep
```

它不是主算法，而是用于发现新 regime 的策略并写回 Strategy Bank。

---

## 3. 创新点与护城河

论文贡献建议写成三点。

### 贡献 1：Probability-Free Utility Estimation

中文：无概率信号收益估计。

问题：NGRAM/Trie 这类 symbolic proposer 没有 draft logits、confidence scores 或 speculator probabilities。TriePilot 只能使用符号匹配特征和历史反馈估计每个请求拿到 draft nodes 是否值得。

核心特征：

```text
match_depth
candidate_count
branch_entropy
top_branch_ratio
filled_nodes
accept_len_ema
negative_gain_count
serving pressure
```

### 贡献 2：Mixed-Batch Draft-Node Allocation

中文：混合批草稿节点预算分配。

不是整个 batch 选一个统一 tier，而是在一个 batch 内给不同请求分配不同预算。

关键对比：

```text
batch-global tier selection：整个 batch 用同一预算
equal allocation：每个请求平均分预算
TriePilot：根据请求收益估计分配预算
```

### 贡献 3：Regime-to-Strategy Reuse

中文：场景到策略的复用。

Strategy Bank 将“场景 → 策略”沉淀下来。相似 regime 再次出现时，直接复用已有预算策略；未知或漂移 regime 才触发探索。这是最接近原始“skills”想法的部分。

---

## 4. 实验环境准备

### 4.1 硬件

主实验：

```text
1 × A100 40G
CUDA 版本记录
NVIDIA driver 版本记录
Python 版本记录
PyTorch 版本记录
SGLang 版本固定为 0.5.6 或指定 commit
```

可选迁移实验：

```text
vLLM N-Gram 或 TensorRT-LLM NGram
```

### 4.2 模型

主模型优先级：

```text
Qwen2.5-7B-Instruct 或 Qwen3-8B
Llama-3.1-8B-Instruct
Qwen2.5-Coder-7B-Instruct，用于代码场景
```

不要第一版上 14B/32B/70B，避免显存和吞吐不稳定。

### 4.3 代码目录建议

```text
configs/
  model/
  dataset/
  method/
  workload/

scripts/
  run_server.sh
  run_bench.py
  run_mixed_workload.py
  collect_trace.py
  analyze_results.py

triepilot/
  regime_encoder.py
  strategy_bank.py
  allocator.py
  telemetry.py
  explorer.py

runs/
  YYYYMMDD_experiment_name/
    config.yaml
    env.json
    git_commit.txt
    raw_events.jsonl
    request_metrics.jsonl
    step_metrics.jsonl
    aggregate.csv
    plots/
```

每次实验必须保存：

```text
git commit
完整配置
数据集 sample ids
随机种子
模型名
框架名
method 名
原始逐步日志
聚合结果
图表
```

---

## 5. 数据集准备

主数据集：

```text
InstructCoder：代码编辑，高 NGRAM 收益场景
JSONSchemaBench 或 BFCL：结构化输出、tool-call、JSON
ShareGPT：真实聊天分布
GSM8K 或 MATH-500：数学推理，低/中 NGRAM 收益
CNN/DailyMail：摘要和长文档生成
SGLang random / random-ids：低可预测负控制
SGLang generated-shared-prefix：可控共享前缀压力测试
```

可选质量 sanity：

```text
HumanEval
MBPP
```

这些只用于验证 greedy 输出一致性或代码质量，不作为主吞吐表。

---

## 6. 工作负载设计

### 6.1 同质 workload

每个数据集单独跑：

```text
InstructCoder only
JSON/tool-call only
ShareGPT only
GSM8K/MATH only
CNN/DailyMail only
random only
```

目的：确认每类请求的 NGRAM 投机收益分布。

### 6.2 二元 mixed workload

必须做：

```text
InstructCoder + GSM8K
InstructCoder + ShareGPT
JSON/tool-call + ShareGPT
CNN/DailyMail + random
```

比例：

```text
50/50
20/80
80/20
```

目的：证明 batch-global 策略会浪费预算，request-level allocation 有价值。

### 6.3 Workload shift

阶段式负载：

```text
Phase 1: InstructCoder
Phase 2: GSM8K
Phase 3: JSON/tool-call
Phase 4: InstructCoder again
```

要观察：

```text
Strategy Bank hit rate
goodput recovery time
negative speedup ratio
budget distribution
```

---

## 7. 基线准备

### 7.1 主公平基线

必须在同一框架、同一模型、同一数据集、同一硬件下跑。

```text
AR / no speculation
SGLang static NGRAM default
Static NGRAM tiers
Best static per workload
Batch-global EMA
Batch-global cost-aware selector
Bandit-style global tier selector
Equal budget allocation
Random budget allocation
Match-depth greedy allocation
Accept-EMA greedy allocation
TriePilot allocation
Oracle allocation
```

### 7.2 近邻论文基线

必须讨论，能复现则复现。

```text
TETRIS：batch speculative decoding draft-token selection
AdaServe：model-based / SLO-aware token-tree selection
SuffixDecoding：model-free suffix-tree adaptive speculation
SAM-Decoding：suffix automaton based model-free speculation
Token Recycling：train-free / draft-free candidate reuse
```

如果无法公平复现，写成 external reference 或 discussion，不放主公平表。

### 7.3 Oracle 定义

Oracle allocation：

```text
通过离线重复运行或 replay，事后知道每个请求在不同 budget 下的收益。
在总 B_batch 约束下选择最优分配。
不可部署，只作为上界。
```

Best static：

```text
每个 workload 事后选择一个固定最优 tier。
可作为强静态调参 baseline。
```

---

## 7.4 当前进度标注（2026-05-12）

已完成：

```text
[x] A100 1 卡服务器清理完成：仅保留 /root/anaconda3/envs/sglang、/root/anaconda3/envs/vllm，以及 /root/sglang_flex_test/datasets、/root/sglang_flex_test/models。
[x] 新增 /root/TriePilot 工作区，不改动现有 sglang/vllm/triton/model 环境。
[x] 采集 A100 环境元数据：env.json、pip freeze、models/datasets manifest。
[x] 确认远端 sglang env：Python 3.10、SGLang 0.5.6.post2；确认 vLLM env 可导入。
[x] 建立代码管理：推送到 https://github.com/HuZhang-ICALab/TriePilot.git。
[x] 落地数据源并规范化到 /root/TriePilot/data/normalized：
    InstructCoder、JSONSchemaBench、ShareGPT-format、GSM8K、CNN/DailyMail、random、shared_prefix。
[x] 保存 sample ids 到 /root/TriePilot/data/sample_ids，并记录 source manifest。
[x] 生成 homogeneous / mixed / shift workload JSONL。
[x] 生成 baseline matrix：/root/TriePilot/matrix/a100_main_matrix_seed20260512.jsonl，共 494 条 run plan。
[x] 配置第 7.1 节所有主公平基线的 method registry / yaml。
[x] 本地与远端 unittest 通过；远端 normalized 数据完成 prompt 非空和 sample_id 唯一性检查。
[x] A100 上跑通 AR/no-spec 端到端 smoke：Qwen3-8B，SGLang 0.5.6.post2，random-ids，8 prompts，结果保存在
    /root/TriePilot/runs/20260512_1650_ar_no_spec_random_smoke_seed20260512/raw_events_len32x16.jsonl。
[x] A100 上跑通 static NGRAM default 端到端 smoke：Qwen3-8B，NGRAM，draft_tokens=8，match_window=12，bfs_breadth=4，
    random-ids，8 prompts，结果保存在
    /root/TriePilot/runs/20260512_1715_static_ngram_default_random_smoke_seed20260512/raw_events_len32x16.jsonl。
[x] 记录并规避 smoke 阻塞点：旧 FlashInfer JIT cache 中残留 /root/anaconda3/envs/sglang-flex include 路径，会导致
    flashinfer/attention/prefill.cuh 与 flashinfer/page.cuh 找不到；本轮使用 run-local FLASHINFER_WORKSPACE_BASE 生成新 cache。
[x] 修复 scripts/run_bench.py：dataset_name 以 random 开头时均透传 random-input-len / random-output-len / random-range-ratio，
    以支持 A100 离线环境下的 random-ids smoke。
[x] A100 上完成 Step 0 feasibility probe：SGLang 0.5.6.post2 NGRAM 仍是 server-level draft_tokens；
    /set_internal_state 尝试把 speculative_num_draft_tokens 从 16 改为 8 返回 200，但 server_info 仍为 16；
    单请求 sampling_params 注入 speculative_num_draft_tokens / triepilot_draft_budget 返回 500。
[x] A100 上完成 Step 0 全局预算对照 smoke：Qwen3-8B，random-ids，16 prompts，max_concurrency=8，
    all-16 与 all-8 结果分别保存在
    /root/TriePilot/runs/20260512_1728_step0_ngram_budget_probe_seed20260512/raw_events_all16_random_ids_len32x16_c8.jsonl
    和 /root/TriePilot/runs/20260512_1738_step0_ngram_all8_seed20260512/raw_events_all8_random_ids_len32x16_c8.jsonl；
    汇总结论保存在 /root/TriePilot/runs/20260512_1740_step0_microbenchmark_summary_seed20260512/notes.md。
[x] 修复工程环境问题：本地建立 .miniconda/envs/triepilot Python 3.10.19 开发环境并安装 requirements-dev；
    本地 pytest 通过 31 passed / 1 skipped。
[x] 修复远端工程环境问题：/root/TriePilot 已补成 git checkout，origin 指向
    https://github.com/HuZhang-ICALab/TriePilot.git；新增项目级 /root/TriePilot/.venv，不改动已有 sglang/vllm conda env；
    远端 .venv pytest 通过 31 passed / 1 skipped。
[x] 修复 A100 启动脚本环境问题：scripts/run_server.sh 显式使用 /root/anaconda3/envs/sglang/bin/python、
    补齐 PATH/PYTHONPATH/HF_ENDPOINT/FLASHINFER_WORKSPACE_BASE；scripts/launch/a100_no_spec.sh 与
    scripts/launch/a100_ngram_static.sh 统一委托 run_server.sh，避免裸 python 启动时找不到 ninja 或误用旧 FlashInfer cache。
[x] A100 上完成 SGLang 最小 read-only NGRAM step telemetry patch 与真实 smoke 验证：新增
    third_party/sglang_flex/python/sglang/srt/speculative/triepilot/recorder.py；
    ServerArgs 增加 --triepilot-trace-path / --triepilot-run-id；
    NGRAMWorker.forward_batch_generation 在 target verify 后记录 batch_size、allocated_budget、actual_draft_nodes、
    verified_nodes、accepted_tokens、wasted_nodes、accept_lens、seq_lens、ngram_query_time_us、verify_time_us、
    step_latency_us、can_run_cuda_graph 等 JSONL 字段。
[x] 修复工作区 SGLang 源码与 A100 conda 安装版的 NGRAM 兼容差异：Req 初始化补回 spec_verify_ct /
    spec_accepted_tokens，避免 NgramVerifyInput.verify 在真实请求上访问缺失字段。
[x] A100 telemetry patch 测试验证完成：新增 tests/test_sglang_triepilot_telemetry.py，扩展
    tests/test_launch_scripts.py，并新增 scripts/remote/a100_ngram_telemetry_smoke.sh；A100 上
    SGLANG_SOURCE_ROOT=/root/TriePilot/third_party/sglang_flex pytest 通过 34 passed / 1 skipped / 2 subtests passed；
    /root/anaconda3/envs/sglang/bin/python py_compile 覆盖 schedule_batch.py、ngram_worker.py、triepilot/recorder.py、
    server_args.py。
[x] A100 上重跑 static NGRAM telemetry smoke：Qwen3-8B，NGRAM，draft_tokens=8，match_window=12，
    bfs_breadth=4，random-ids，8 prompts，max_concurrency=4，结果保存在
    /root/TriePilot/runs/20260512_telemetry_ngram_smoke_seed20260512/raw_events_len32x16.jsonl；
    step telemetry 保存在
    /root/TriePilot/runs/20260512_telemetry_ngram_smoke_seed20260512/raw_step_events.jsonl，共 54 条事件；
    首条事件包含 actual_draft_nodes=8、verified_nodes=8、accepted_tokens、wasted_nodes、verify_time_us、
    step_latency_us、can_run_cuda_graph 等字段。
[x] A100 上完成 Step 0 最小 per-request active budget mask/slice patch 与 Case A/B/C 验证：新增
    third_party/sglang_flex/python/sglang/srt/speculative/triepilot/budget.py，NGRAM path 支持从
    sampling_params.custom_params.triepilot_draft_budget 读取 request-level budget，并在 mixed batch 中构造
    compact verify input；A100 上 SGLANG_SOURCE_ROOT=/root/TriePilot/third_party/sglang_flex pytest 覆盖
    test_sglang_triepilot_telemetry.py / test_launch_scripts.py / test_run_bench.py，结果 7 passed /
    2 subtests passed；/root/anaconda3/envs/sglang/bin/python py_compile 覆盖本次 SGLang patch 文件。
[x] A100 Step 0 Case A/B/C 真实验证完成：Qwen3-8B，NGRAM server-level draft_tokens=16，8 prompts，
    结果保存在 /root/TriePilot/runs/20260512_225000_step0_per_request_budget_cases_seed20260512；
    Case A all-16 首轮 actual_draft_nodes=128、verify_input_tokens=128、can_run_cuda_graph=True、
    mean verify_time_us=17432.32；Case B half16/half0 首轮 actual_draft_nodes=64、
    verify_input_tokens=68、can_run_cuda_graph=False、mean verify_time_us=1621.80；Case C heterogeneous
    首轮 actual_draft_nodes=54、verify_input_tokens=56、can_run_cuda_graph=False、mean verify_time_us=1586.52。
    结论：per-request budget 已经能减少实际 verified draft nodes；但 Case B/C 进入 non-CUDA-graph runtime path，
    step_latency 仍受 padding/shape/kernel path 影响，后续主实验必须同时记录 verify_time 与 end-to-end TPOT/p99。
[x] A100 上完成 Step 1 NGRAM 结构特征插桩第一版：新增
    third_party/sglang_flex/python/sglang/srt/speculative/triepilot/features.py，
    从 NGRAM tree mask 记录 match_depths、candidate_counts、branch_entropies、top_branch_ratios、filled_nodes，
    并在 batch 级记录 match_depth、candidate_count、branch_entropy、top_branch_ratio、filled_nodes_mean。
    本地目标 pytest 9 passed；A100 上
    SGLANG_SOURCE_ROOT=/root/TriePilot/third_party/sglang_flex pytest 覆盖
    test_sglang_triepilot_telemetry.py / test_launch_scripts.py，结果 9 passed；
    /root/anaconda3/envs/sglang/bin/python py_compile 覆盖 ngram_worker.py、triepilot/recorder.py、triepilot/features.py。
[x] A100 上完成 Step 1 static NGRAM budget tiers 首轮真实验证：Qwen3-8B，random-ids，16 prompts，
    max_concurrency=8，request_rate=8，random_input_len=32，random_output_len=16，
    match_window=12，bfs_breadth=4，branch_length=18，budgets=0/2/4/8/16/24/32。
    结果保存在 /root/TriePilot/runs/20260512_step1_static_ngram_tiers_random_ids_seed20260512；
    每个 NGRAM budget 均保存 raw_step_events.jsonl 与 raw_events_len32x16.jsonl，汇总保存在
    tier_summary.csv 和 notes.md。random-ids 负控制上 accepted_per_verified_node 随 budget 增大下降：
    budget 2 为 0.1062，4 为 0.0671，8 为 0.0497，16 为 0.0214，24 为 0.0143，32 为 0.0103；
    mean TPOT 分别约为 AR 18.55ms、budget 8 20.51ms、budget 16 22.35ms、budget 24/32 约 43ms，
    说明低可预测场景下大 budget 会显著增加 wasted verified nodes 和尾部延迟压力。
[x] A100 上完成 Step 1 static NGRAM tier library 第一版扩展验证：Qwen3-8B，normalized JSONL
    通过 runner 临时转换为 SGLang bench_serving 可读的 ShareGPT-style JSON，num_prompts=16，
    max_concurrency=8，request_rate=8，sharegpt_output_len=16，sharegpt_context_len=4096，
    budgets=0/2/4/8/16/24/32。覆盖 InstructCoder、JSON/tool-call、ShareGPT、GSM8K、
    CNN/DailyMail、shared_prefix，并完成两组 tree shape：默认 mw12/b4/branch18 与对照
    mw24/b8/branch34。正式结果共 12 个 sweep、84 行 combined summary、72 个非零 budget
    raw_step_events.jsonl；driver 汇总保存在
    /root/TriePilot/runs/20260512_step1_static_tier_library_driver_seed20260512/combined_tier_summary.csv，
    notes 保存在 /root/TriePilot/runs/20260512_step1_static_tier_library_driver_seed20260512/notes.md。
    远端 runner smoke 通过，pytest 覆盖 tests/test_run_bench.py / tests/test_launch_scripts.py，
    结果 5 passed / 2 subtests passed；正式 driver 未出现 traceback、missing trace、OOM 或 error。
    第一版短输出/小样本结果显示：所有 dataset/shape 的非零预算中 accepted_per_verified_node 最优均为
    budget 2；AR/budget 0 在 mean TPOT 和 p99 TPOT 上通常最优，只有 shared_prefix 的 mw24/b8 对照中
    p99 TPOT 最优为 budget 2。budget 32 的 wasted_node_ratio 在真实 workload 上约 0.986-0.993，
    shared_prefix 上约 0.968-0.977，说明大 budget 仍显著浪费 verifier nodes；mw24/b8 相比 mw12/b4
    在 GSM8K、ShareGPT、shared_prefix 等 workload 上略提高 budget 2 的 APV，但短输出配置下尚不足以
    抵消 TPOT/p99 压力。该结果作为 Pareto tier library 和后续 allocator baseline 的输入，不作为最终主表。
[x] 进入 Session 5 并完成最小 allocator baseline 实现与 A100 smoke 验证：新增 request-level
    allocation policy 支持 custom / equal_budget_allocation / random_budget_allocation /
    match_depth_greedy / accept_ema_greedy；ServerArgs 与 scripts/run_server.sh 增加
    --triepilot-allocation-policy / --triepilot-batch-budget / --triepilot-random-seed /
    --triepilot-accept-ema-alpha；NGRAMWorker 在每轮先构造 full-budget NGRAM tree 读取结构特征，
    再按 policy 切 active draft lengths，并记录 allocation_policy、batch_budget、accept_len_ema。
[x] 新增 workload materialize 工具与 Session 5 runner：triepilot/workloads/materialize.py、
    scripts/materialize_workload.py、scripts/remote/a100_session5_allocator_baselines.sh。
    本地 targeted pytest 通过 15 passed；A100 上
    SGLANG_SOURCE_ROOT=/root/TriePilot/third_party/sglang_flex pytest 覆盖
    test_sglang_triepilot_telemetry.py / test_launch_scripts.py / test_workload_materialize.py，
    结果 15 passed；/root/anaconda3/envs/sglang/bin/python py_compile 覆盖 server_args.py、
    ngram_worker.py、triepilot/budget.py、triepilot/recorder.py，.venv py_compile 覆盖 materialize 工具。
[x] A100 上完成 Session 5 allocator baseline smoke：Qwen3-8B，InstructCoder + GSM8K 50/50，
    16 prompts，max_concurrency=8，request_rate=8，sharegpt_output_len=16，server max draft_tokens=16，
    B_batch=64；结果保存在
    /root/TriePilot/runs/20260513_session5_allocator_baselines_smoke_seed20260512，
    汇总为 allocator_summary.csv，notes.md。运行方法包括 batch_global_budget2、equal_budget_allocation、
    random_budget_allocation、match_depth_greedy、accept_ema_greedy；driver 内置检查确认 request-level
    policies 每轮 Σ allocated_budgets ≤ 64，样例 heterogeneous budgets 包括 equal 的
    [10,9,9,9,9,9,9]、random 的 [13,4,10,2,4,12,4,15]、greedy 的 [16,16,16,16,0,0,0,0]。
    smoke 结果：batch_global_budget2 mean TPOT=17.53ms、p99 TPOT=19.49ms、APV=0.0598；
    equal 为 21.06ms / 28.12ms / 0.0259；random 为 21.30ms / 28.77ms / 0.0269；
    match-depth greedy 为 22.11ms / 28.75ms / 0.0215；accept-EMA greedy 为
    22.97ms / 30.18ms / 0.0190。结论：baseline allocator plumbing 已经跑通并能产生 batch 内异构预算；
    但在短输出小样本下，B_batch=64 的 request-level variable path 仍受 non-CUDA-graph / packing path 影响，
    TPOT 与 wasted-node ratio 劣于全局小 budget=2。该结果是 Session 5 基线基础设施 smoke，不作为最终主表。
[x] A100 上完成 Session 5 正式 TriePilot allocation 最小实现与路径边界修正：TriePilot 的
    budget / features / recorder 逻辑已放回项目自身目录
    /root/TriePilot/triepilot/sglang_integration/{budget.py,features.py,recorder.py}；
    third_party/sglang_flex 下仅保留必要 SGLang runtime hook（server_args.py 与 ngram_worker.py），
    ngram_worker.py 从 triepilot.sglang_integration 导入，不再从 sglang.srt.speculative.triepilot 导入。
[x] 修复 A100 runtime import 路径：scripts/run_server.sh 将 PYTHONPATH 设置为
    /root/TriePilot:/root/TriePilot/third_party/sglang_flex/python（若外部已有 PYTHONPATH 则前置这两项），
    确保 SGLang server 真实启动时能 import TriePilot 项目包；远端 dry-run 已确认输出正确 PYTHONPATH。
[x] A100 上完成搬迁后远端验证：SGLANG_SOURCE_ROOT=/root/TriePilot/third_party/sglang_flex
    .venv/bin/python -m pytest tests/test_sglang_triepilot_telemetry.py tests/test_launch_scripts.py
    tests/test_run_bench.py tests/test_workload_materialize.py -q 通过，结果 20 passed / 2 subtests passed；
    /root/anaconda3/envs/sglang/bin/python py_compile 覆盖 server_args.py、ngram_worker.py、
    triepilot/sglang_integration/budget.py、features.py、recorder.py；带 conda PATH 的 import 检查输出
    imports_ok TriePilotStrategyBank TriePilotStrategyBank。
[x] A100 上完成搬迁后真实 runtime smoke：Qwen3-8B，InstructCoder + GSM8K 50/50，
    NUM_PROMPTS=4、max_concurrency=2、request_rate=2、sharegpt_output_len=16、B_batch=64、
    method=triepilot_allocation，结果保存在
    /root/TriePilot/runs/20260513_session5_post_move_runtime_smoke_seed20260512。
    验证脚本确认 session5_sweep_summary.csv 1 行、raw_step_events.jsonl 共 50 条事件、regime_ids 与
    strategy_bank_hit_rate 字段存在、每轮 Σ allocated_budgets ≤ 64、server.log 包含正确 PYTHONPATH、
    无残留 sglang.launch_server 进程。该 smoke 的 accepted_per_verified_node=0.2，
    mean TPOT=20.36ms，mean verify_time_us=1550.95，strategy_bank_hit_rate=0.87。
[x] A100 上完成 Session 5 小型 B_batch 诊断 sweep：Qwen3-8B，InstructCoder+GSM8K 与
    CNN/DailyMail+random 两组 50/50 mixed workload，NUM_PROMPTS=32、max_concurrency=8、
    request_rate=8、sharegpt_output_len=32，B_batch=16/32/64，对比 batch_global_budget2、
    equal_budget_allocation、match_depth_greedy、accept_ema_greedy、triepilot_allocation。
    结果保存在
    /root/TriePilot/runs/20260513_0945_session5_bbatch_diagnostic_seed20260512；
    session5_sweep_summary.csv 共 30 行，raw step trace 共 4342 条事件；远端完整性检查确认
    missing_files=0、budget_violations=0、driver log 无 Traceback / missing trace / B_batch exceeded /
    server health error，实验结束后 GPU 回到 1 MiB 且无残留 sglang.launch_server / bench_serving 进程。
    诊断结果：InstructCoder+GSM8K 上 batch_global_budget2 在 B=16/32/64 的 mean TPOT 与 p99 TPOT
    均最优（mean 17.69-17.83ms，p99 22.61-22.83ms）；TriePilot verified_nodes 仅 94，
    APV=0.1170-0.1277、mean verify_time_us≈1.5ms，但 mean TPOT=22.58-23.24ms，说明
    verify compute 节省尚未转化为端到端收益。CNN/DailyMail+random 上 batch_global_budget2 的
    mean TPOT 仍最优（35.16-35.35ms）；TriePilot verified_nodes 仅 133/231/241，
    APV=0.1255-0.1353、mean verify_time_us≈1.5ms，其中 B=16/32 的 p99 TPOT 最优
    （50.83/49.81ms，对比 batch_global_budget2 的 56.33/56.34ms），但 mean TPOT 仍慢
    （37.10/37.45/40.25ms）。结论：当前 page_size=1 variable verification path 可以显著减少
    verified draft nodes 并改善部分长上下文尾部延迟，但 mean TPOT 仍普遍受 non-CUDA-graph /
    packing / runtime shape 影响；暂不启动全量 mixed workload 主表。
[x] A100 上完成 Session 5 强静态 batch-global baseline 补强：在相同两组 50/50 诊断 workload 上
    加入 batch_global_budget4/8/16，并保留 batch_global_budget2 对照；NUM_PROMPTS=32、
    max_concurrency=8、request_rate=8、sharegpt_output_len=32。结果保存在
    /root/TriePilot/runs/20260513_1030_session5_batch_global_baseline_seed20260512；
    session5_sweep_summary.csv 共 8 行。InstructCoder+GSM8K 上 mean TPOT 最优为
    batch_global_budget8（17.21ms），p99 TPOT 最优为 batch_global_budget2（22.67ms）；
    CNN/DailyMail+random 上 mean TPOT 最优为 batch_global_budget8（34.51ms），p99 TPOT
    最优为 batch_global_budget2（56.44ms）。结论：强静态 baseline 不能只用 budget2 表示；
    mean-optimal 与 tail-optimal static budget 会分离，后续主表至少需要报告 tuned batch-global。
[x] A100 上完成 variable verification path 根因诊断与 telemetry 加固：runner 支持
    batch_global_budgetN 正则方法名，summary 新增 mean_target_forward_time_us、
    mean_ngram_query_time_us、cuda_graph_token_shape_ok_ratio；recorder 新增 draft_token_num、
    cuda_graph_expected_tokens、cuda_graph_actual_tokens、cuda_graph_token_shape_ok。A100 targeted
    pytest 通过 17 passed / 2 subtests passed，sglang conda env 下 py_compile 覆盖
    triepilot/sglang_integration/recorder.py；runtime smoke 保存在
    /root/TriePilot/runs/20260513_1045_instrumentation_runtime_smoke_seed20260512。该 smoke 显示
    batch_global_budget2 的 cuda_graph_token_shape_ok_ratio=1.0、mean_target_forward_time_us≈1.0ms；
    triepilot_allocation 的 cuda_graph_token_shape_ok_ratio=0.0、mean_target_forward_time_us≈19.4ms，
    而 mean verify_time_us 仍约 1.5ms。结论：variable allocation 当前主要瓶颈不是 verifier
    post-processing，而是 compact verify input 触发 CUDA graph token-shape mismatch，使 target forward
    进入慢路径。
[x] A100 上完成 Session 5 shape bucket runtime path 第一版：新增 shape bucket 解析与量化逻辑，
    默认 bucket set 为 0/1、2、4、8、16；ServerArgs / scripts/run_server.sh / Session 5 runner
    支持 --triepilot-shape-buckets 与 --triepilot-shape-bucket-mode。NGRAMWorker 在
    batch_max 模式下按 batch 内最大 active bucket 构造固定 verify token shape，同时记录
    requested_draft_budgets、bucketed_draft_budgets、active_draft_lengths、bucket_ids、
    bucket_padding_nodes、shape_padding_tokens、verify_draft_token_num、cuda_graph token-shape telemetry。
    CUDA graph runner 支持 NGRAM 按 (num_tokens_per_bs, bs) 捕获/回放多个 token shape。
[x] A100 上完成 shape bucket runtime 根因修复与 smoke 验证：初始
    triepilot_allocation + batch_max 在 verify 后 _fill_requests 报 CUDA illegal memory access；
    系统排查确认 graph / output buffer 已按 (tokens_per_bs, bs) 分 key，但 FlashInfer target-verify
    wrapper metadata 仍只按 bs 覆盖，导致 replay 更新错 shape wrapper。已将 FlashInfer target-verify
    prefill_cuda_graph_metadata 改为按 (draft_token_num, bs) key。A100 targeted pytest 覆盖
    tests/test_sglang_triepilot_telemetry.py 与 tests/test_launch_scripts.py，结果 20 passed /
    2 subtests passed；sglang conda env 下 py_compile 覆盖 server_args.py、ngram_info.py、
    ngram_worker.py、cuda_graph_runner.py、flashinfer_backend.py、triepilot/sglang_integration/budget.py、
    recorder.py。真实 smoke 1 保存在
    /root/TriePilot/runs/20260513_shape_bucket_metadata_fix_smoke_seed20260512：InstructCoder+GSM8K
    50/50，NUM_PROMPTS=4、max_concurrency=2、B_batch=16、equal_budget_allocation 与
    triepilot_allocation 均完成，无 Traceback / CUDA error；TriePilot cuda_graph_ratio=1.0、
    cuda_graph_token_shape_ok_ratio=1.0、mean_target_forward_time_us=1121.97us、verified_nodes=72、
    accepted_per_verified_node=0.2222、mean_verify_input_tokens=2.77、mean TPOT=13.94ms、
    p99 TPOT=17.46ms。真实 smoke 2 保存在
    /root/TriePilot/runs/20260513_shape_bucket_triepilot_c8_smoke_seed20260512：NUM_PROMPTS=16、
    max_concurrency=8、B_batch=16、triepilot_allocation 完成，cuda_graph_ratio=1.0、
    cuda_graph_token_shape_ok_ratio=1.0、mean_target_forward_time_us=1204.84us、verified_nodes=94、
    accepted_per_verified_node=0.1277、mean_verify_input_tokens=10.875、mean TPOT=18.88ms、
    p99 TPOT=21.95ms。实验结束后 GPU 回到 1 MiB，且无残留 sglang.launch_server /
    bench_serving 进程。
[x] A100 上完成 Session 5 shape-bucket 同构诊断复跑：按
    20260513_0945_session5_bbatch_diagnostic_seed20260512 同构配置覆盖 InstructCoder+GSM8K 与
    CNN/DailyMail+random 两组 50/50，NUM_PROMPTS=32、max_concurrency=8、request_rate=8、
    sharegpt_output_len=32、B_batch=16/32/64。结果保存在
    /root/TriePilot/runs/20260513_1254_session5_shape_bucket_diagnostic_seed20260512；
    compact/off + batch_global_budget2/4/8/16 完成 30 行，summary 为
    compact_off/session5_sweep_summary.csv；shape_bucket_batch_max 在默认
    CUDA_GRAPH_MAX_BS=32/MAX_RUNNING_REQUESTS=32 下短上下文完成 3 行，但在
    CNN/DailyMail+random B=16 触发 CUDA OOM，根因是 shape-bucket graph capture 同时捕获
    bs=1..32 与 tokens_per_bs=1/2/4/8/16，graph capture 后仅剩约 0.15GB 显存，长上下文
    4096-token prefill 进入 target forward 时 OOM。随后补跑 shape_bucket_batch_max_cgbs8：
    CUDA_GRAPH_MAX_BS=8、MAX_RUNNING_REQUESTS=8、bucket set=0/1,2,4,8,16，完成 6 行，
    summary 为 shape_bucket_batch_max_cgbs8/session5_sweep_summary.csv；combined summary 为
    combined_session5_shape_bucket_diagnostic_cgbs8.csv，comparison summary 为
    comparison_session5_shape_bucket_cgbs8.csv，notes 为 notes_cgbs8.md。完整性检查确认
    compact/off 30 行、cgbs8 6 行、combined 39 行、comparison 6 行，request-level budget
    violation=0，cgbs8 log 无 Traceback / OOM，实验结束后 GPU 回到 1 MiB 且无残留
    sglang.launch_server / bench_serving 进程。
[x] shape-bucket 诊断结论：cgbs8 将 TriePilot 从 compact variable 慢路径拉回 CUDA graph path，
    cuda_graph_ratio=1.0、cuda_graph_token_shape_ok_ratio=1.0，mean_target_forward_time_us
    约 1.04-1.10ms（compact/off TriePilot 约 18.6-19.5ms），同时保持 verified_nodes 远低于
    tuned batch-global。InstructCoder+GSM8K 上，B=16 的 shape-bucket mean/p99 TPOT 为
    17.29/18.67ms，对比 best-static mean budget8=17.32ms、best-static p99 budget2=22.85ms；
    B=32/64 的 mean TPOT 比 best-static mean 慢约 0.74-0.77ms，但 p99 优约 3.5-3.6ms。
    CNN/DailyMail+random 上，shape-bucket 在 B=16/32/64 的 mean TPOT 为
    32.23/32.28/32.17ms，均优于 best-static mean budget8 的 34.40/34.39/34.45ms；
    p99 TPOT 为 48.46/48.18/48.39ms，均优于 best-static p99 budget2 的
    56.38/56.35/56.47ms。说明 shape bucket 在长上下文混合负载上已同时改善 mean 与 tail，
    在短上下文混合负载上主要改善 tail，具备进入更大 mixed workload 的工程前提。
[x] A100 上完成 Session 5 cgbs8 batch-global fairness smoke：先用 TDD 在 A100 上验证 runner
    元数据补丁，新增 tests/test_launch_scripts.py 断言 Session 5 runner 将 CUDA_GRAPH_MAX_BS 与
    MAX_RUNNING_REQUESTS 写入 config.yaml 与 session5_sweep_summary.csv；RED 阶段按预期失败，
    补丁后 A100 上 pytest tests/test_launch_scripts.py 通过 4 passed / 2 subtests passed，
    bash -n scripts/remote/a100_session5_allocator_baselines.sh 通过。公平性 smoke 在相同
    CUDA_GRAPH_MAX_BS=8、MAX_RUNNING_REQUESTS=8 约束下，对 InstructCoder+GSM8K 与
    CNN/DailyMail+random 两组 50/50、B_batch=16/32/64 复跑 batch_global_budget2/4/8/16，
    结果保存在
    /root/TriePilot/runs/20260513_1445_session5_cgbs8_batch_global_fairness_seed20260512；
    session5_sweep_summary.csv 共 24 行，comparison summary 为
    comparison_shape_bucket_vs_batch_global_cgbs8_fairness.csv。完整性检查确认 missing_files=0、
    budget/CUDA graph shape violations=0，config 记录 cuda_graph_max_bs: 8 与
    max_running_requests: 8，实验结束后 GPU 回到 1 MiB 且无残留 sglang.launch_server /
    bench_serving 进程。
[x] cgbs8 fairness 结论：在相同 cgbs8 运行约束下，batch-global 的 mean-optimal 仍为
    batch_global_budget8，tail-optimal 仍为 batch_global_budget2。相对公平 cgbs8 best static，
    shape-bucket TriePilot 在 InstructCoder+GSM8K 上 B=16 的 mean TPOT 慢 0.14ms、p99 快
    3.83ms；B=32/64 的 mean TPOT 慢 0.83/0.73ms、p99 快 3.62/3.43ms。CNN/DailyMail+random
    上 shape-bucket TriePilot 在 B=16/32/64 的 mean TPOT 快 2.54/2.39/2.57ms，p99 快
    2.20/2.47/2.33ms。结论：shape-bucket 的长上下文收益不是单纯来自 max_running /
    cuda_graph cap；短上下文场景仍主要是 tail 改善，mean 需要靠更强策略或更合适 workload。
[x] A100 上完成 Session 5 四组 mixed pairs 50/50 小主表：Qwen3-8B，
    CUDA_GRAPH_MAX_BS=8、MAX_RUNNING_REQUESTS=8、NUM_PROMPTS=32、max_concurrency=8、
    request_rate=8、sharegpt_output_len=32、B_batch=16/32/64。覆盖
    InstructCoder+GSM8K、InstructCoder+ShareGPT、JSON/tool-call+ShareGPT、
    CNN/DailyMail+random。拆成三条 runtime path：1）batch_global_budget2/4/8/16 +
    shape_bucket_mode=off；2）compact TriePilot runtime ablation：triepilot_allocation +
    shape_bucket_mode=off；3）shape-bucket TriePilot：triepilot_allocation +
    shape_bucket_mode=batch_max + bucket set=0/1,2,4,8,16。结果保存在
    /root/TriePilot/runs/20260513_1555_session5_mixed50_small_table_seed20260512；
    combined summary 为 combined_session5_mixed50_small_table.csv，comparison summary 为
    comparison_session5_mixed50_small_table.csv，notes 为 notes.md。完整性检查确认
    batch_global 48 行、compact 12 行、shape-bucket 12 行；raw_step_events 与 bench outputs
    各 72 个；raw trace 预算违规=0；shape-bucket graph/shape ratio <0.99 的行数=0；
    log 中 Traceback / CUDA out of memory / RuntimeError / server health error 均为 0；
    实验结束后 GPU 回到 1 MiB，且无残留 sglang.launch_server / bench_serving 进程。
[x] 四组 50/50 小主表结论：shape-bucket TriePilot 的 p99 TPOT 在 12/12 个
    pair×B_batch 对比中优于 best tuned batch-global，mean TPOT 在 5/12 个对比中优于
    best tuned batch-global；相对 compact TriePilot，shape-bucket 的 mean 与 p99 均为
    12/12 胜出。CNN/DailyMail+random 上 B=16/32/64 的 shape-bucket mean TPOT 分别快
    2.50/2.47/2.57ms，p99 快 1.54/2.52/2.54ms；InstructCoder+GSM8K 上 mean 慢
    0.23/0.74/0.63ms，但 p99 快 4.27/3.52/3.33ms；InstructCoder+ShareGPT 上
    B=32/64 mean 快 0.82/0.69ms，三档 p99 均快 0.80-2.30ms；JSON/tool-call+ShareGPT 上
    mean 约持平或略慢 0.03-1.60ms，但 p99 快 1.72-2.03ms。shape-bucket 12 行
    cuda_graph_ratio=1.0、cuda_graph_token_shape_ok_ratio=1.0，mean_target_forward_time_us
    约 1.06-1.12ms；compact TriePilot 12 行 cuda_graph_ratio=0.0，target forward 约
    18.5-20.1ms。该结果说明 50/50 小主表已具备进入比例扩展的工程稳定性；论文主张上应强调
    tail-latency 与 wasted verifier node 降低，不应把当前策略写成所有场景 mean TPOT 都优于 tuned static。
[x] A100 上完成 Session 5 四组 mixed pairs 的 20/80 与 80/20 比例敏感性扩展：Qwen3-8B，
    CUDA_GRAPH_MAX_BS=8、MAX_RUNNING_REQUESTS=8、NUM_PROMPTS=32、max_concurrency=8、
    request_rate=8、sharegpt_output_len=32、B_batch=16/32/64。覆盖 InstructCoder+GSM8K、
    InstructCoder+ShareGPT、JSON/tool-call+ShareGPT、CNN/DailyMail+random，并继续拆成三条
    runtime path：batch_global_budget2/4/8/16 + shape_bucket_mode=off、compact TriePilot
    + shape_bucket_mode=off、shape-bucket TriePilot + shape_bucket_mode=batch_max +
    bucket set=0/1,2,4,8,16。结果保存在
    /root/TriePilot/runs/20260513_2030_session5_ratio_sensitivity_seed20260512；
    combined summary 为 combined_session5_ratio_sensitivity.csv，comparison summary 为
    comparison_session5_ratio_sensitivity.csv，notes 为 notes.md。完整性检查确认
    batch_global 96 行、compact 24 行、shape-bucket 24 行；raw_step_events 与 bench outputs
    各 144 个；raw trace 预算违规=0；shape-bucket graph/shape bad events=0；
    successful_requests != 32 的 bench output 数=0；log 中 Traceback / CUDA out of memory /
    RuntimeError / server health error 均为 0；实验结束后 GPU 回到 1 MiB，且无残留
    sglang.launch_server / bench_serving 进程。
[x] 比例敏感性结论：shape-bucket TriePilot 的 p99 TPOT 在 24/24 个 pair×ratio×B_batch
    对比中优于 best tuned batch-global，mean TPOT 在 12/24 个对比中优于 best tuned
    batch-global；相对 compact TriePilot，shape-bucket 的 mean 与 p99 均为 22/24 胜出。
    分 pair 看，CNN/DailyMail+random 上 mean 胜出 3/6、p99 胜出 6/6，平均 mean/p99
    delta 为 -0.36ms / -4.43ms；InstructCoder+GSM8K 上 mean 胜出 2/6、p99 胜出
    6/6，平均 delta 为 +0.87ms / -2.29ms；InstructCoder+ShareGPT 上 mean 胜出
    4/6、p99 胜出 6/6，平均 delta 为 -0.91ms / -3.05ms；JSON/tool-call+ShareGPT
    上 mean 胜出 3/6、p99 胜出 6/6，平均 delta 为 -0.67ms / -2.15ms。结论延续
    50/50 小主表：shape-bucket 的稳定收益主要体现在 tail-latency 与 runtime-path 稳定性，
    mean TPOT 受 workload composition 影响，不能写成全场景 mean 都赢 tuned static。
[x] A100 上完成 Session 5 larger-budget / longer-output 增量验证：Qwen3-8B，
    四组 mixed pairs（InstructCoder+GSM8K、InstructCoder+ShareGPT、JSON/tool-call+ShareGPT、
    CNN/DailyMail+random），ratio=50/50、20/80、80/20，B_batch=100/160，
    NUM_PROMPTS=64、max_concurrency=8、request_rate=8、sharegpt_output_len=64、
    sharegpt_context_len=4096、CUDA_GRAPH_MAX_BS=8、MAX_RUNNING_REQUESTS=8。
    主对照为 batch_global_budget2/4/8/16 与 shape-bucket TriePilot
    （bucket set=0/1,2,4,8,16；shape_bucket_mode=batch_max），本轮不扩展已知慢的
    compact path。结果保存在
    /root/TriePilot/runs/20260513_2335_session5_larger_budget_out64_shape_bucket_seed20260512；
    summary 为 session5_sweep_summary.csv（120 行结果），comparison 为
    comparison_session5_larger_budget_out64_shape_bucket.csv，分析 notes 为 analysis_notes.md。
    完整性检查确认 rows=120/120、combos=24/24、missing_or_empty_files=0、
    budget_violations=0、driver log 无 Traceback / missing trace / B_batch exceeded /
    server health error / OOM / CUDA illegal memory access；所有方法
    min_cuda_graph_token_shape_ok_ratio=1.0，TriePilot min_cuda_graph_token_shape_ok_ratio=1.0；
    实验结束后 GPU 回到 1 MiB，且无残留 sglang.launch_server / Session 5 runner 进程。
[x] larger-budget / longer-output 结论：shape-bucket TriePilot 对 tuned static 的 p99 TPOT
    在 22/24 个 pair×ratio×B_batch 对比中胜出，且 22/24 个对比在 best-static p99
    的 5% 以内；mean TPOT 仅 2/24 个对比胜出，14/24 个对比在 best-static mean
    的 10% 以内。平均看，TriePilot 相对 p99-best static 的 p99 TPOT delta 为
    -1.81ms，相对 mean-best static 的 mean TPOT delta 为 +1.83ms；verified-node
    平均相对 p99-best static 减少 98.22%，相对 mean-best static 减少 99.34%。
    分 pair 看，CNN/DailyMail+random 的 p99 胜出 6/6、mean 胜出 2/6，平均 p99
    delta -5.12ms；InstructCoder+GSM8K、InstructCoder+ShareGPT 的 p99 都是
    6/6 胜出但 mean 0/6 胜出；JSON/tool-call+ShareGPT 的 p99 为 4/6 胜出且
    mean 0/6 胜出。该结果说明 shape-bucket 已解决 compact path 的 CUDA graph
    mismatch，并支持“显著降低 verifier waste、经常改善 p99”的论文叙述；不能写成
    mean TPOT 全面优于 tuned batch-global static。
[x] A100 上完成 compact TriePilot runtime ablation 小子集：Qwen3-8B，
    JSON/tool-call+ShareGPT 与 CNN/DailyMail+random 两组 mixed workload，ratio=50/50
    和 80/20，B_batch=100/160，NUM_PROMPTS=64、max_concurrency=8、
    request_rate=8、sharegpt_output_len=64、sharegpt_context_len=4096，
    CUDA_GRAPH_MAX_BS=8、MAX_RUNNING_REQUESTS=8。compact path 使用
    triepilot_allocation + shape_bucket_mode=off，并与既有
    20260513_2335 larger-budget / longer-output shape-bucket 结果按同配置对齐比较。
    结果保存在
    /root/TriePilot/runs/20260514_0030_session5_compact_ablation_out64_seed20260512；
    compact summary 为 combined_compact_ablation_out64.csv，comparison 为
    comparison_compact_vs_shape_bucket_out64.csv，分析 notes 为
    analysis_compact_vs_shape_bucket_out64.md。完整性检查确认 compact_rows=8、
    comparison_rows=8、missing_files=0、budget_violations=0，driver / server log
    无 Traceback / OOM / RuntimeError / server health error / B_batch exceeded /
    missing output；实验结束后 GPU 回到 1 MiB，且无残留 sglang.launch_server /
    bench_serving / Session 5 runner 进程。
[x] compact runtime ablation 结论：shape-bucket TriePilot 在 8/8 个对比中 mean TPOT
    与 p99 TPOT 均优于 compact path。compact-minus-shape 平均 mean TPOT delta 为
    +4.06ms，p99 TPOT delta 为 +4.47ms，target-forward delta 为 +18057.71us；
    compact min cuda_graph_ratio=0.00，而 shape-bucket min cuda_graph_ratio=1.00。
    分 workload 看，CNN/DailyMail+random 的 compact mean 比 shape-bucket 慢
    2.23-3.73ms、p99 慢 3.86-4.63ms；JSON/tool-call+ShareGPT 的 compact
    mean 慢 4.79-5.61ms、p99 慢 4.45-5.30ms。该结果确认大预算长输出下
    compact variable path 的主要劣势仍来自离开 CUDA graph 的 target forward 慢路径，
    shape-bucket 是后续主表更合理的 deployable runtime path。
[x] A100 上完成 Session 5 shape-bucket 多 seed 稳健性验证：Qwen3-8B，四组
    mixed pairs、ratio=50/50、B_batch=100/160、NUM_PROMPTS=64、max_concurrency=8、
    request_rate=8、sharegpt_output_len=64、sharegpt_context_len=4096，
    CUDA_GRAPH_MAX_BS=8、MAX_RUNNING_REQUESTS=8。新增 seed=20260513 与
    seed=20260514；每个 seed 下同配置跑 batch_global_budget2/4/8/16 与
    shape-bucket triepilot_allocation（bucket set=default，即 0/1、2、4、8、16）。
    结果保存在
    /root/TriePilot/runs/20260514_0735_session5_multiseed_shape_bucket_robustness；
    combined summary 为 combined_multiseed_shape_bucket_robustness.csv，
    comparison summary 为 comparison_multiseed_shape_bucket_robustness.csv，
    分析 notes 为 analysis_multiseed_shape_bucket_robustness.md。完整性检查确认
    batch-global seed20260513/seed20260514 各 32 行，shape-bucket seed20260513/
    seed20260514 各 8 行，combined=80 行、comparison=16 行；raw trace 预算违规=0；
    shape-bucket cuda_graph_ratio 与 cuda_graph_token_shape_ok_ratio 最小值均为 1.0；
    driver/server log 无 Traceback / OOM / RuntimeError / server health error /
    B_batch exceeded / missing output；实验结束后 GPU 回到 1 MiB 且无残留
    sglang.launch_server / bench_serving / multiseed driver 进程。
[x] 多 seed 稳健性结论：shape-bucket TriePilot 相对同 seed、同 pair、同 B_batch 下
    tuned best-static batch-global，p99 TPOT 在 15/16 个对比中胜出，平均 p99 delta
    为 -1.43ms；mean TPOT 在 2/16 个对比中胜出，平均 mean delta 为 +1.59ms；
    accepted_per_verified_node 相对 best-static APV 平均为 1.36x。分 pair 看，
    CNN/DailyMail+random 在 seed20260513 的 B=100/160 上 mean 与 p99 均胜出，
    seed20260514 上 p99 胜出但 mean 慢约 3.0ms；InstructCoder+GSM8K 两个 seed
    的 p99 均胜出、mean 均慢约 1.6-1.9ms；InstructCoder+ShareGPT 除
    seed20260513 B=100 外 p99 均胜出，mean 慢约 0.4-2.0ms；JSON/tool-call+ShareGPT
    p99 全部胜出，mean 慢约 1.8-3.2ms。JSON/tool-call+ShareGPT 在两个 seed、
    所有方法上 bench completed 均为 59/64，属于同 workload 下跨方法一致的
    bench/materialization caveat，后续正式主表需要记录或修正。
[x] A100 上完成 JSON/tool-call+ShareGPT completed=59/64 根因排查与 materialize 修正：
    bench_serving 的 ShareGPT loader 会在发送前过滤 prompt_len + output_len > context_len
    的样本；旧 materialize 只输出 64 条，导致 JSON/tool-call 的超长 schema prompt 被过滤后
    completed 固定掉到 59/64 或 62/64。新增 materialize_workload_to_sharegpt 的
    prompt_token_counter / max_prompt_tokens / fill_filtered 选项，runner 默认使用
    /root/anaconda3/envs/sglang/bin/python + Qwen3 tokenizer 按 SHAREGPT_CONTEXT_LEN 与
    SHAREGPT_OUTPUT_LEN 过滤并在同一 dataset 内向后补样，保持 mixed ratio 不漂移。TDD 在
    A100 上完成：RED 阶段 tests/test_workload_materialize.py 与 tests/test_launch_scripts.py
    按预期失败，补丁后 A100 上
    .venv/bin/python -m pytest tests/test_workload_materialize.py tests/test_launch_scripts.py
    -q 通过 6 passed / 2 subtests passed；后续完整 targeted 验证通过 8 passed /
    2 subtests passed，并通过 bash -n 与 py_compile。真实 materialize smoke 保存在
    /root/TriePilot/runs/20260514_materialize_filter_smoke，JSON/tool-call+ShareGPT
    r0.5 输出 64 条，过滤后 invalid=0，替换 5 条超长 json_tool 样本。
[x] A100 上完成 Session 5 ratio=20/80 与 80/20 的 shape-bucket 多 seed 稳健性扩展：
    Qwen3-8B，四组 mixed pairs、ratio=0.2/0.8、B_batch=100/160、NUM_PROMPTS=64、
    max_concurrency=8、request_rate=8、sharegpt_output_len=64、sharegpt_context_len=4096，
    CUDA_GRAPH_MAX_BS=8、MAX_RUNNING_REQUESTS=8。新增 seed=20260513 与 seed=20260514；
    每个 seed 下跑 batch_global_budget2/4/8/16（shape_bucket_mode=off）与
    shape-bucket triepilot_allocation（shape_bucket_mode=batch_max，bucket set=default）。
    结果保存在
    /root/TriePilot/runs/20260514_1120_session5_ratio_multiseed_shape_bucket_robustness；
    combined summary 为 combined_ratio_multiseed_shape_bucket_robustness.csv，
    comparison summary 为 comparison_ratio_multiseed_shape_bucket_robustness.csv，
    分析 notes 为 analysis_ratio_multiseed_shape_bucket_robustness.md。完整性检查确认
    batch-global seed20260513/seed20260514 各 64 行，shape-bucket seed20260513/
    seed20260514 各 16 行，combined=160 行、comparison=32 行；所有 bench completed=64，
    errors 为空；raw trace 预算违规=0；shape-bucket cuda_graph_ratio 与
    cuda_graph_token_shape_ok_ratio 最小值均为 1.0；driver log 无 Traceback / OOM /
    RuntimeError / server health error / B_batch exceeded / missing output / CUDA error；
    实验结束后 GPU 回到 1 MiB 且无残留 sglang.launch_server / bench_serving / driver 进程。
[x] ratio 多 seed 稳健性结论：shape-bucket TriePilot 相对同 seed、同 pair、同 ratio、
    同 B_batch 下 tuned best-static batch-global，p99 TPOT 在 28/32 个对比中胜出，
    平均 p99 delta 为 -2.19ms；mean TPOT 在 2/32 个对比中胜出，平均 mean delta
    为 +2.21ms；accepted_per_verified_node 相对 mean-tuned static 平均为 1.95x。
    分 ratio 看，20/80 的 p99 16/16 胜出、mean 0/16 胜出，平均 p99/mean delta
    为 -3.63ms / +2.71ms；80/20 的 p99 12/16 胜出、mean 2/16 胜出，平均
    p99/mean delta 为 -0.74ms / +1.71ms。分 pair 看，CNN/DailyMail+random 的
    p99 8/8 胜出且 mean 2/8 胜出，平均 p99 delta -5.45ms；InstructCoder+GSM8K
    与 InstructCoder+ShareGPT 的 p99 均 8/8 胜出但 mean 均 0/8；JSON/tool-call+ShareGPT
    的 p99 4/8 胜出，主要失败集中在 ratio=0.8，mean 全部慢约 3.5ms。该结果进一步支持
    “tail-latency / verifier waste 优势稳定，mean TPOT 需要 mean-aware utility 或 static fallback”
    的下一步方向。
[x] A100 上完成 mean-aware static fallback 负面诊断并回退实现：曾临时新增
    triepilot_mean_aware_allocation 作为诊断策略，在 dense predictable batch 中启用 capped
    static fallback=8，并跑通 A100 targeted pytest / py_compile / diagnostic sweep。实验确认该
    策略几乎退化为 batch_global_budget8 后，已从代码中回退该 policy、CLI choice、测试断言与
    telemetry allocation_modes 字段，避免进入后续测试和开发。
[x] A100 上完成 mean-aware policy diagnostic sweep：Qwen3-8B，三组代表性 mixed workload
    （InstructCoder+GSM8K、JSON/tool-call+ShareGPT、CNN/DailyMail+random），ratio=0.2/0.8，
    B_batch=100，NUM_PROMPTS=64，sharegpt_output_len=64，CUDA_GRAPH_MAX_BS=8、
    MAX_RUNNING_REQUESTS=8，比较 batch_global_budget2、batch_global_budget8、原
    triepilot_allocation 与 triepilot_mean_aware_allocation。结果保存在
    /root/TriePilot/runs/20260514_1600_session5_mean_aware_policy_diagnostic；
    session5_sweep_summary.csv 共 24/24 行，6/6 个 combo summary，raw trace 与 bench output
    均非空；driver log 无 Traceback / OOM / RuntimeError / server health error /
    B_batch exceeded / invalid choice；实验结束后 GPU 回到 1 MiB。
[x] mean-aware 负面诊断结论：capped static fallback 能显著追回 mean TPOT，但代价是几乎退化为
    batch_global_budget8。mean-aware 相对原 TriePilot mean TPOT 5/6 胜出，平均 delta=-2.49ms；
    但 p99 TPOT 0/6 胜出，平均 delta=+5.31ms；相对 budget8 的 mean 仅 1/6 胜出，
    平均慢 0.34ms，p99 3/6 胜出、平均快 0.29ms。verified nodes 平均是原 TriePilot 的
    150.0x、是 budget8 的 1.01x。结论：static fallback 只作为负面诊断证据保留，不能作为
    主方法、可选 baseline 或后续开发入口；后续要做 selective fallback / marginal utility，只在
    replay 证明接受收益足以抵消 shape cost 的 regime 上提高预算。
```

尚未完成：

```text
[ ] per-request budget 已接入最小 allocator baselines，正式 TriePilot regime encoder / Strategy Bank /
    utility estimator 第一版已能在线运行并写出 telemetry；shape bucket batch_max runtime path 已通过
    A100 smoke 与同构诊断 sweep，但尚未实现 Slow Explorer，也尚未接入更复杂 tree shape。
[ ] Step 1 已完成第一版 static tier library 与一组 match_window / bfs_breadth 对照；尚未做 match mode
    sweep、更长输出长度、更大样本数、多随机种子或正式主表规模复跑。
[ ] Session 5 已完成四组 mixed pairs 的 50/50 小主表、20/80 与 80/20 single-seed
    比例敏感性扩展，并已完成 B_batch=100/160、output_len=64、NUM_PROMPTS=64 的
    single-seed larger-budget / longer-output shape-bucket 增量验证、compact path 小子集
    ablation，以及 50/50、20/80、80/20、B_batch=100/160 的新增双 seed shape-bucket
    稳健性验证；尚未补更多输出长度/样本数或完整正式 baseline 主表。
[ ] mean TPOT 优化已完成 static fallback 负面诊断且已回退实现：当前 shape-bucket TriePilot
    已稳定降低 verified nodes 并改善 p99，fallback 退化路径能追回 mean 但 verified-node 与
    p99 代价过大，不能进入后续主线。后续仍需要基于 trace replay / oracle 做 selective fallback
    或 marginal utility model，显式纳入 accepted-token gain、verified-node cost、shape padding、
    CUDA graph shape 与 batch composition。
[ ] 尚未跑正式 baseline throughput / TPOT / wasted-node 主表；当前已有 AR、static NGRAM smoke 与
    random-ids / main workload static tier 第一版结果，以及 Session 5 allocator smoke，但还不是完整主表。
```

下一步：

```text
继续 Session 5：50/50、20/80、80/20 的 B_batch=100/160 新增双 seed 结果已经确认
shape-bucket 在 p99 TPOT 与 accepted_per_verified_node 上稳定优于 tuned batch-global，
但 mean TPOT 仍不是全面胜出；JSON/tool-call+ShareGPT 的 completed=59/64 已通过
tokenizer/context-aware materialize 补样修正，后续正式主表应默认启用 fill-filtered 并记录
替换样本数。暂不直接启动全量 mixed workload 主表。mean-aware static fallback 负面诊断已经证明：
mean TPOT 可以靠接近 budget8 的 fallback 追回，但这会基本放弃 verified-node 节省并损伤 p99；
该退化策略实现已回退，不能作为后续测试或开发入口。
下一步优先做 selective fallback / marginal allocator：基于已有 trace replay 估计每个 regime 的
边际 accepted-token gain 与 shape cost，只允许少数高置信 regime 从 off/tiny 升到 2/4/8，并把
fallback gate、shape-padding penalty、runtime-path penalty 和 workload-composition 特征写入
Strategy Bank / Utility Estimator。并行准备 Step 1 的 match mode / tree shape 与更长输出补充，
作为正式主表前的稳健性材料。
```

---

## 8. 实验步骤规划

### Step 0：工程可行性 microbenchmark

目标：确认 per-request budget cap 是否真的减少实际 verify cost。

实验：

```text
batch size 固定为 8 或 16

Case A: 所有 request 都 16 draft nodes
Case B: 一半 request 16 nodes，一半 0 nodes
Case C: 每个 request 不同 nodes
```

记录：

```text
actual verified draft nodes
forward latency
verify_time
padding 行为
CUDA graph / runtime shape
GPU memory
```

关键判断：

```text
如果减少 per-request nodes 不能降低实际 verify cost，则必须先改 packing / verification path，否则主论文点不成立。
```

当前验证（2026-05-12）：

```text
A100 feasibility probe 已确认：SGLang 0.5.6.post2 NGRAM 的 speculative_num_draft_tokens 是 server-level 固定值。
/set_internal_state 不能实际修改 NGRAM budget；请求级 sampling_params 中加入预算字段会失败。
因此未打补丁前只能跑 all-16 / all-8 这类全局预算对照，不能完成一半 16/一半 0 或 per-request heterogeneous nodes。
最小 SGLang NGRAM packing / verification path patch 后，A100 已完成 Case A/B/C：
Case A all-16 首轮 actual_draft_nodes=128、verify_input_tokens=128、CUDA graph=True；
Case B half16/half0 首轮 actual_draft_nodes=64、verify_input_tokens=68、CUDA graph=False；
Case C heterogeneous 首轮 actual_draft_nodes=54、verify_input_tokens=56、CUDA graph=False。
Case B/C 的 verify_time 明显低于 all-16，但 step_latency 不单调，说明后续实验必须区分 verify compute、
runtime shape/CUDA graph 行为和端到端 serving latency。
```

### Step 1：静态 NGRAM 与 tier library

目标：构建安全的 runtime budget set。

运行：

```text
AR
SGLang NGRAM default
static budgets: 0, 2, 4, 8, 16, 24, 32
不同 depth / breadth / match mode
```

输出：

```text
Pareto tiers
best static
oracle headroom
```

注意：tier library 只是基础设施，不是论文核心创新。

### Step 2：同质 workload 主结果

每个单独数据集运行所有主基线。

指标：

```text
output tokens/s
TPOT
ITL
p95/p99 latency
accepted tokens
verified nodes
wasted nodes
accepted per verified node
negative speedup ratio
```

目的：确认不同数据集的投机收益差异。

### Step 3：mixed-batch allocation 主实验

运行混合负载：

```text
InstructCoder + GSM8K
InstructCoder + ShareGPT
JSON + ShareGPT
CNN/DailyMail + random
```

比较：

```text
batch-global best tier
equal allocation
match-depth greedy
accept-EMA greedy
TriePilot
oracle
```

核心指标：

```text
wasted verified nodes
accepted per verified node
TPOT
p99 latency
negative speedup ratio
```

这是最重要的主实验。

### Step 4：B_batch sweep

控制总预算：

```text
B_batch = 0, 16, 32, 64, 100, 160
```

比较：

```text
equal
greedy
TriePilot
oracle
```

目的：证明 TriePilot 在不同预算规模下都优于简单分配。

### Step 5：Strategy Bank 冷启动与热启动

实验：

```text
Cold start：Strategy Bank 为空
Warm start：已有相似 regime 的策略
```

观察：

```text
前 N 个 step 的 goodput
恢复速度
exploration 次数
Strategy Bank hit rate
negative speedup ratio
```

目的：证明“策略沉淀和复用”有效。

### Step 6：Workload shift

阶段：

```text
代码 → 数学 → JSON → 代码再次出现
```

画图：

```text
goodput over time
budget distribution over time
regime id over time
strategy bank hit rate
exploration trigger
```

目的：证明再次遇到相似 regime 时可以直接复用策略。

### Step 7：消融实验

至少做：

```text
w/o trie features
w/o history feedback
w/o serving features
w/o Strategy Bank
w/o safe guard
batch-global only
request-level allocation only
```

目的：证明每个模块都有必要。

### Step 8：控制器开销实验

记录：

```text
Regime Encoder time
Strategy Bank lookup time
Allocator time
NGRAM query time
Verify time
End-to-end step latency
```

不要写 zero-cost。写：

```text
bounded overhead
O(1) policy lookup
no extra model forward
```

### Step 9：框架可迁移性实验

最低要求：

```text
用 vLLM N-Gram 或 TensorRT-LLM NGram 做 trace-driven replay
```

更好：

```text
实现简化版 allocator，控制 vLLM N-Gram 的 draft budget
```

目的：证明方法不是 SGLang-specific。

---

## 9. 实验数据保存规范

每次实验创建唯一目录：

```text
runs/YYYYMMDD_HHMM_method_dataset_workload_seed/
```

必须保存：

```text
config.yaml
env.json
git_commit.txt
dataset_sample_ids.json
raw_step_events.jsonl
request_metrics.jsonl
aggregate_metrics.csv
plots/
notes.md
```

`raw_step_events.jsonl` 至少包含：

```text
timestamp
framework
model
dataset
workload_id
request_id
batch_id
step_id
method
regime_id
allocated_budget
actual_draft_nodes
verified_nodes
accepted_tokens
wasted_nodes
match_depth
candidate_count
branch_entropy
top_branch_ratio
accept_len_ema
batch_size
queue_len
kv_usage
controller_time_us
ngram_query_time_us
verify_time_us
step_latency_us
TPOT
strategy_bank_hit
exploration_flag
```

聚合脚本必须输出：

```text
mean throughput
mean TPOT
p95 TPOT
p99 TPOT
accepted_per_verified_node
wasted_node_ratio
negative_speedup_ratio
controller_overhead_p50/p99
strategy_bank_hit_rate
oracle_gap
```

---

## 10. 论文撰写规划

### Section 1：Introduction

主线：

```text
Speculative decoding 已经很重要。
符号式 proposer 如 NGRAM/Trie 不需要额外草稿模型，部署面广。
但在 mixed batch serving 中，有限 verifier-side draft-node budget 不应平均分或全局统一分。
已有 batch allocation 方法多依赖 draft probabilities / speculator logits。
TriePilot 研究 probability-free setting。
```

### Section 2：Related Work

必须分组：

```text
Model-based speculative decoding
Draft-free / symbolic speculative decoding
Dynamic tree / budget control
Batch speculative decoding allocation
Serving-aware speculative decoding
```

强调：

```text
TETRIS/AdaServe 是强近邻。
TriePilot 不主张首次做 batch allocation。
TriePilot 研究 probability-free symbolic proposer 的变体。
```

### Section 3：Problem Definition

定义：

```text
symbolic proposer
draft forest
B_batch
per-request budget b_i
probability-free features
optimization objective
```

### Section 4：Method

包含：

```text
SymbolicProposer abstraction
Regime Encoder
Strategy Bank
Utility Estimator
Budget Allocator
Slow Explorer
Telemetry update
```

### Section 5：Implementation

写：

```text
主要实现于 SGLang 0.5.6
接口可迁移到 vLLM / TensorRT-LLM
per-request cache
per-round draft tree
batch-level draft forest
logging and instrumentation
```

### Section 6：Evaluation

回答问题：

```text
RQ1: 不同请求的 NGRAM 投机收益是否差异显著？
RQ2: TriePilot 是否减少 wasted verified nodes？
RQ3: TriePilot 是否改善 TPOT / p99 / negative speedup？
RQ4: Strategy Bank 是否能复用策略？
RQ5: 控制器开销是否足够低？
RQ6: 方法是否可迁移到其他框架？
```

### Section 7：Limitations

诚实写：

```text
第一版默认 per-request cache，不做全局可写 trie
不主张超过 model-based EAGLE/AdaServe
sampling correctness 可作为 future work
SGLang 实现可能受 padding / CUDA graph 约束
```

---

## 11. 多 session 推进计划

### Session 1：环境与基线

状态（2026-05-12）：部分完成。环境、模型路径、日志目录和 GitHub 管理已完成；AR 与 static NGRAM 端到端结果尚未跑。

任务：

```text
[x] 固定并记录 SGLang 0.5.6.post2 环境
[x] 跑通 AR/no-spec 端到端 smoke
[x] 跑通 static NGRAM default 端到端 smoke
[x] 准备并记录 Qwen3-8B / Llama-3.1-8B 本地模型候选
[x] 搭建日志目录结构
```

产出：

```text
[x] env.json
[x] baseline AR smoke result
[x] static NGRAM smoke result
```

### Session 2：数据集与 workload

状态（2026-05-12）：已完成第一版。主数据源已规范化，sample ids 已保存，homogeneous/mixed/shift workload 已生成。

任务：

```text
[x] 下载并清洗 InstructCoder / ShareGPT-format / GSM8K / CNN/DailyMail / random
[x] 下载并清洗 JSONSchemaBench，作为 JSON/tool-call 第一版数据源
[x] 构造 homogeneous / mixed / shift workload JSONL
[x] 保存 sample ids
```

产出：

```text
[x] data/*.jsonl
[x] workload configs
[x] source manifest
```

### Session 3：NGRAM 特征与 telemetry

状态（2026-05-12）：A100 已完成最小 read-only NGRAM step telemetry patch、单元测试、语法编译和
static NGRAM telemetry smoke；Step 0 probe 已确认未打补丁时无法表达 per-request budget；最小
per-request active budget mask/slice patch 已完成并通过 A100 Case A/B/C；Step 1 已补充 NGRAM tree mask
结构特征插桩并通过 A100 pytest / py_compile / static tiers 首轮验证，且已把 static tier library 扩展到
6 个主 workload 与一组 mw24/b8 tree shape 对照。

任务：

```text
[x] 本地插桩 accept_lens / actual_draft_nodes / verified_nodes / accepted_tokens / wasted_nodes / latency
[x] A100 static NGRAM telemetry sanity：验证 JSONL 写入与字段语义
[x] A100 per-request budget Case A/B/C sanity：验证同一 batch 内 all-16、half16/half0、heterogeneous nodes
[x] 插桩 match_depth / candidate_count / branch_entropy / top_branch_ratio / filled_nodes
[x] 插桩 target_forward_time_us 与 CUDA graph token-shape 字段：
    draft_token_num、cuda_graph_expected_tokens、cuda_graph_actual_tokens、cuda_graph_token_shape_ok
```

产出：

```text
[x] A100 raw_step_events.jsonl：
    /root/TriePilot/runs/20260512_telemetry_ngram_smoke_seed20260512/raw_step_events.jsonl
[x] Step 1 feature sanity / static tier trace：
    /root/TriePilot/runs/20260512_step1_static_ngram_tiers_random_ids_seed20260512/budget_*/raw_step_events.jsonl
    /root/TriePilot/runs/20260512_step1_static_ngram_tiers_random_ids_seed20260512/tier_summary.csv
    /root/TriePilot/runs/20260512_step1_static_ngram_tiers_random_ids_seed20260512/notes.md
[x] Step 1 main workload static tier library：
    /root/TriePilot/runs/20260512_step1_static_tier_library_driver_seed20260512/combined_tier_summary.csv
    /root/TriePilot/runs/20260512_step1_static_tier_library_driver_seed20260512/notes.md
[x] Session 5 CUDA graph / target-forward diagnostic smoke：
    /root/TriePilot/runs/20260513_1045_instrumentation_runtime_smoke_seed20260512/session5_sweep_summary.csv
    /root/TriePilot/runs/20260513_1045_instrumentation_runtime_smoke_seed20260512/*/raw_step_events.jsonl
```

### Session 4：per-request budget microbenchmark

状态（2026-05-12）：已完成未打补丁 feasibility probe、all-16/all-8 全局预算 smoke，以及最小 SGLang
per-request active budget mask/slice patch；A100 已重跑 Case A/B/C，确认 per-request nodes 可以减少
actual verified draft nodes，但 heterogeneous path 会离开 CUDA graph。

任务：

```text
[x] 验证不同 request draft nodes 是否减少实际 verify cost
[x] 确认 padding / shape / CUDA graph 行为
[x] 重跑 Case A all-16、Case B 一半 16/一半 0、Case C heterogeneous nodes
```

产出：

```text
[x] microbenchmark table：
    /root/TriePilot/runs/20260512_225000_step0_per_request_budget_cases_seed20260512/notes.md
[x] 决策：per-request budget 具备继续 V2/V3 的工程基础；后续需要优化/记录 non-CUDA-graph path 对 TPOT/p99 的影响
```

### Session 5：实现简单 allocator baselines

任务：

```text
[x] equal allocation
[x] random allocation
[x] match-depth greedy
[x] accept-EMA greedy
[x] TriePilot allocation 第一版：Regime Encoder / Strategy Bank / Utility Estimator / Greedy Budget Allocator
[x] batch-global tuned tier 第一版：batch_global_budget2/4/8/16
[x] 小型 B_batch 诊断 sweep：InstructCoder+GSM8K、CNN/DailyMail+random，B_batch=16/32/64
[x] shape bucket runtime path：将 per-request budget 量化到 0/1、2、4、8、16 等固定 bucket，
    按 bucket 构造 graph-compatible verify input，并记录 bucket padding / CUDA graph shape telemetry
[x] shape bucket 复跑诊断：复跑 20260513_0945_session5_bbatch_diagnostic_seed20260512 同构配置，
    对比 compact variable path、batch_global_budget2/4/8/16 与 shape-bucket TriePilot；默认
    CUDA_GRAPH_MAX_BS=32 shape-bucket 在长上下文 OOM，cgbs8 补跑完成并给出可用结果
[x] cgbs8 batch-global fairness smoke：在相同 CUDA_GRAPH_MAX_BS=8 / MAX_RUNNING_REQUESTS=8
    约束下复跑 batch_global_budget2/4/8/16，确认 shape-bucket 长上下文收益不是单纯来自
    max_running / cuda_graph cap
[x] 四组 mixed pairs 50/50 小主表：B_batch=16/32/64，固定 cgbs8 / max_running=8，
    对比 batch_global_budget2/4/8/16、compact TriePilot 与 shape-bucket TriePilot
[x] 四组 mixed pairs 20/80 与 80/20 比例敏感性：B_batch=16/32/64，固定 cgbs8 /
    max_running=8，对比 batch_global_budget2/4/8/16、compact TriePilot 与 shape-bucket TriePilot
[x] larger-budget / longer-output 增量验证：B_batch=100/160、NUM_PROMPTS=64、
    sharegpt_output_len=64，固定 cgbs8 / max_running=8，对比 batch_global_budget2/4/8/16
    与 shape-bucket TriePilot
[x] compact TriePilot runtime ablation 小子集：JSON/tool-call+ShareGPT 与 CNN/DailyMail+random，
    ratio=50/50 和 80/20，B_batch=100/160，NUM_PROMPTS=64、sharegpt_output_len=64，
    固定 cgbs8 / max_running=8，对比 compact path 与既有 shape-bucket 同配置结果
[x] shape-bucket 多 seed 稳健性：四组 mixed pairs、ratio=50/50、B_batch=100/160，
    NUM_PROMPTS=64、sharegpt_output_len=64，固定 cgbs8 / max_running=8，补
    seed=20260513 与 seed=20260514，对比 batch_global_budget2/4/8/16 与
    shape-bucket TriePilot
[x] materialize tokenizer/context-aware 补样：修正 JSON/tool-call+ShareGPT 因超长 prompt 被
    bench_serving 过滤导致 completed=59/64 或 62/64 的 caveat
[x] shape-bucket ratio 多 seed 稳健性：四组 mixed pairs、ratio=20/80 与 80/20、
    B_batch=100/160，NUM_PROMPTS=64、sharegpt_output_len=64，固定 cgbs8 / max_running=8，
    补 seed=20260513 与 seed=20260514，对比 batch_global_budget2/4/8/16 与
    shape-bucket TriePilot
[x] mean-aware static fallback 负面诊断与回退：临时实现过 triepilot_mean_aware_allocation 并完成
    A100 TDD、py_compile 与 3 组 workload × 2 ratio diagnostic sweep；因结果退化为
    batch_global_budget8，已回退 policy / CLI choice / 测试断言 / telemetry 字段，后续不作为
    可选策略或开发入口
[ ] selective mean-aware utility / cost model 优化：基于 trace replay 与 oracle 上界，加入
    selective fallback gate、shape-padding penalty、runtime-path penalty 与 workload-composition
    特征，目标是在尽量保持 p99 与 verified-node 优势的同时缩小 mean TPOT 差距
[ ] 正式 mixed workload 主表：更大样本、多 seed、完整主公平 baseline 与必要消融
```

产出：

```text
[x] allocator smoke：
    /root/TriePilot/runs/20260513_session5_allocator_baselines_smoke_seed20260512/allocator_summary.csv
[x] TriePilot post-move runtime smoke：
    /root/TriePilot/runs/20260513_session5_post_move_runtime_smoke_seed20260512/session5_sweep_summary.csv
[x] B_batch diagnostic sweep：
    /root/TriePilot/runs/20260513_0945_session5_bbatch_diagnostic_seed20260512/session5_sweep_summary.csv
[x] strong batch-global baseline：
    /root/TriePilot/runs/20260513_1030_session5_batch_global_baseline_seed20260512/session5_sweep_summary.csv
[x] CUDA graph / target-forward diagnostic：
    /root/TriePilot/runs/20260513_1045_instrumentation_runtime_smoke_seed20260512/session5_sweep_summary.csv
[x] shape bucket runtime smoke：
    /root/TriePilot/runs/20260513_shape_bucket_metadata_fix_smoke_seed20260512/session5_sweep_summary.csv
[x] shape bucket TriePilot c8 smoke：
    /root/TriePilot/runs/20260513_shape_bucket_triepilot_c8_smoke_seed20260512/session5_sweep_summary.csv
[x] shape bucket 同构诊断复跑：
    /root/TriePilot/runs/20260513_1254_session5_shape_bucket_diagnostic_seed20260512/compact_off/session5_sweep_summary.csv
    /root/TriePilot/runs/20260513_1254_session5_shape_bucket_diagnostic_seed20260512/shape_bucket_batch_max_cgbs8/session5_sweep_summary.csv
    /root/TriePilot/runs/20260513_1254_session5_shape_bucket_diagnostic_seed20260512/comparison_session5_shape_bucket_cgbs8.csv
[x] cgbs8 batch-global fairness smoke：
    /root/TriePilot/runs/20260513_1445_session5_cgbs8_batch_global_fairness_seed20260512/session5_sweep_summary.csv
    /root/TriePilot/runs/20260513_1445_session5_cgbs8_batch_global_fairness_seed20260512/comparison_shape_bucket_vs_batch_global_cgbs8_fairness.csv
[x] 四组 mixed pairs 50/50 小主表：
    /root/TriePilot/runs/20260513_1555_session5_mixed50_small_table_seed20260512/combined_session5_mixed50_small_table.csv
    /root/TriePilot/runs/20260513_1555_session5_mixed50_small_table_seed20260512/comparison_session5_mixed50_small_table.csv
    /root/TriePilot/runs/20260513_1555_session5_mixed50_small_table_seed20260512/notes.md
[x] 四组 mixed pairs 20/80 与 80/20 比例敏感性：
    /root/TriePilot/runs/20260513_2030_session5_ratio_sensitivity_seed20260512/combined_session5_ratio_sensitivity.csv
    /root/TriePilot/runs/20260513_2030_session5_ratio_sensitivity_seed20260512/comparison_session5_ratio_sensitivity.csv
    /root/TriePilot/runs/20260513_2030_session5_ratio_sensitivity_seed20260512/notes.md
[x] larger-budget / longer-output shape-bucket 增量验证：
    /root/TriePilot/runs/20260513_2335_session5_larger_budget_out64_shape_bucket_seed20260512/session5_sweep_summary.csv
    /root/TriePilot/runs/20260513_2335_session5_larger_budget_out64_shape_bucket_seed20260512/comparison_session5_larger_budget_out64_shape_bucket.csv
    /root/TriePilot/runs/20260513_2335_session5_larger_budget_out64_shape_bucket_seed20260512/analysis_notes.md
[x] compact TriePilot runtime ablation 小子集：
    /root/TriePilot/runs/20260514_0030_session5_compact_ablation_out64_seed20260512/combined_compact_ablation_out64.csv
    /root/TriePilot/runs/20260514_0030_session5_compact_ablation_out64_seed20260512/comparison_compact_vs_shape_bucket_out64.csv
    /root/TriePilot/runs/20260514_0030_session5_compact_ablation_out64_seed20260512/analysis_compact_vs_shape_bucket_out64.md
[x] shape-bucket 多 seed 稳健性：
    /root/TriePilot/runs/20260514_0735_session5_multiseed_shape_bucket_robustness/combined_multiseed_shape_bucket_robustness.csv
    /root/TriePilot/runs/20260514_0735_session5_multiseed_shape_bucket_robustness/comparison_multiseed_shape_bucket_robustness.csv
    /root/TriePilot/runs/20260514_0735_session5_multiseed_shape_bucket_robustness/analysis_multiseed_shape_bucket_robustness.md
[x] materialize filter smoke：
    /root/TriePilot/runs/20260514_materialize_filter_smoke/bench_json_tool_sharegpt_r0p5.json
[x] shape-bucket ratio 多 seed 稳健性：
    /root/TriePilot/runs/20260514_1120_session5_ratio_multiseed_shape_bucket_robustness/combined_ratio_multiseed_shape_bucket_robustness.csv
    /root/TriePilot/runs/20260514_1120_session5_ratio_multiseed_shape_bucket_robustness/comparison_ratio_multiseed_shape_bucket_robustness.csv
    /root/TriePilot/runs/20260514_1120_session5_ratio_multiseed_shape_bucket_robustness/analysis_ratio_multiseed_shape_bucket_robustness.md
[x] mean-aware static fallback negative diagnostic（结果保留，策略实现已回退）：
    /root/TriePilot/runs/20260514_1600_session5_mean_aware_policy_diagnostic/session5_sweep_summary.csv
    /root/TriePilot/runs/20260514_1600_session5_mean_aware_policy_diagnostic/comparison_mean_aware_policy_diagnostic.csv
    /root/TriePilot/runs/20260514_1600_session5_mean_aware_policy_diagnostic/analysis_mean_aware_policy_diagnostic.md
```

### Session 6：实现 TriePilot

任务：

```text
Regime Encoder
Strategy Bank
Utility Estimator
Greedy Budget Allocator
Telemetry update
```

产出：

```text
TriePilot 主方法结果
```

### Session 7：mixed-batch 主实验

任务：

```text
跑 4 组 mixed workload
跑所有主基线
保存完整日志
```

产出：

```text
主表数据
budget allocation 可视化
```

### Session 8：Strategy Bank 实验

任务：

```text
cold vs warm
workload shift
regime reuse
```

产出：

```text
strategy bank hit rate
recovery time 图
```

### Session 9：消融与开销

任务：

```text
w/o trie features
w/o history
w/o serving
w/o Strategy Bank
controller overhead
```

产出：

```text
ablation table
overhead table
```

### Session 10：可迁移性实验

任务：

```text
vLLM 或 TensorRT-LLM NGram trace-driven replay
映射同一套 allocator
```

产出：

```text
portability appendix result
```

### Session 11：论文初稿

任务：

```text
写 Introduction / Problem / Method / Evaluation skeleton
整理 related work 边界
```

产出：

```text
paper draft v0
```

### Session 12：图表与审稿防御

任务：

```text
完善 TETRIS/AdaServe/SuffixDecoding 对比
写 limitations
写 why not SGLang-specific
写 why probability-free
```

产出：

```text
paper draft v1
rebuttal notes
```

---

## 12. 最后执行原则

1. **主问题只保留一个**：无概率信号下，mixed batch 的 draft-node budget 怎么分。
2. **不要把 Strategy Bank 包装成玄学**：它就是 regime → allocation skill 的表。
3. **不要主张 SGLang 专用**：SGLang 是实现载体，方法抽象为 symbolic proposer + verifier + allocator。
4. **不要只看吞吐**：必须重点看 wasted verified nodes、accepted per verified node、p99 TPOT、negative speedup ratio。
5. **先验证工程关键点**：per-request budget 是否真的减少实际 verification cost。如果不能，先改 packing/verification path。
6. **相关工作必须诚实**：TETRIS 和 AdaServe 是强近邻，但它们依赖概率信号；TriePilot 研究 probability-free 变体。

[1]: https://aclanthology.org/2025.acl-long.1598/?utm_source=chatgpt.com "TETRIS: Optimal Draft Token Selection for Batch ..."
