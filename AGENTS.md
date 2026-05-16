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
[x] A100 上完成 selective fallback / cost-model trace replay 负面诊断：曾临时新增
    triepilot/replay/selective_fallback.py、scripts/analyze_selective_fallback.py 与
    tests/test_selective_fallback_replay.py，用于读取已有 Session 5 summary 与 shape-bucket
    raw_step_events，并输出 selective_fallback_decisions.csv、selective_fallback_summary.json、
    regime_profile.csv 与 analysis_selective_fallback.md。诊断完成后已删除这些临时代码与测试，
    只保留 A100 结果产物和 AGENTS.md 结论，避免后续把 selective static fallback 当作正式开发入口。
    已验证过的临时代码结果为：A100 tests/test_selective_fallback_replay.py 通过 5 passed，
    py_compile 覆盖临时 replay 模块与 CLI。
[x] selective fallback replay 结论：覆盖 20260513_2335 larger-budget/out64、20260514_0735
    50/50 multi-seed、20260514_1120 ratio multi-seed 共 72 个 pair×ratio×B_batch×seed
    场景。mean-best static 在 66/72 个场景 mean TPOT 优于 shape-bucket TriePilot，但在
    p99 regression≤1ms 且 verified_nodes≤4×shape 的严格 gate 下 0/72 允许 fallback；
    拒绝原因为 p99_regression 58、verified_cost 8、no_mean_gain 6。mean-best static 的
    verified-node multiplier 平均为 shape-bucket 的 134.07×。放宽到
    p99≤shape+1ms、verified≤256× 时仅 8/72 可 fallback，chosen verified multiplier 仍约
    14.94×；放宽到 p99≤shape+5ms、verified≤256× 时 54/72 可 fallback，但 chosen verified
    multiplier 约 98.87×。结论：static fallback oracle 只能作为负面上界，不能进入主方法；
    不再继续实现 selective static fallback / batch-level static fallback 方向；下一步应做
    per-regime marginal upgrade，而不是 batch-level fallback。
[x] A100 上完成 Session 6 per-regime marginal upgrade 最小实现：将 triepilot_allocation 从
    “按 preferred_budget 一次性贪心”改为小步边际升级 allocator，当前只允许 0→2 与少数
    高收益/high-confidence regime 的 2→4 升级，不再向 8/16 做 batch-level fallback。
    telemetry 新增 budget_caps、marginal_upgrade_steps、marginal_upgrade_gains，方便分析每个
    request 为什么被升级。A100 TDD RED 阶段 tests/test_sglang_triepilot_telemetry.py 预期失败
    2 failed / 15 passed；GREEN 后
    SGLANG_SOURCE_ROOT=/root/TriePilot/third_party/sglang_flex .venv/bin/python -m pytest
    tests/test_sglang_triepilot_telemetry.py tests/test_launch_scripts.py tests/test_workload_materialize.py -q
    通过 23 passed / 2 subtests passed；/root/anaconda3/envs/sglang/bin/python -m py_compile
    覆盖 triepilot/sglang_integration/budget.py、recorder.py、third_party/sglang_flex 的
    ngram_worker.py 与 server_args.py。
[x] A100 上完成 marginal allocator trace-level sanity 与小型 live diagnostic：结果保存在
    /root/TriePilot/runs/20260514_2030_session6_marginal_allocator_diagnostic。trace replay 使用
    /root/TriePilot/runs/20260514_1832_session5_selective_fallback_replay/regime_profile.csv，
    小步 gate 在 27 个 regime profile 行中激活 11 行，其中 cap=2 有 8 行、cap=4 有 3 行，
    输出 marginal_replay_summary.json。live sweep 使用 Qwen3-8B，InstructCoder+GSM8K 与
    CNN/DailyMail+random 两组 50/50，B_batch=16/64，NUM_PROMPTS=32，output_len=32，
    cgbs8 / max_running=8，shape_bucket_mode=batch_max，对比 batch_global_budget2/4/8/16、
    equal、match-depth、accept-EMA 与 triepilot_allocation。完整性检查确认 session5_sweep_summary.csv
    共 32 行、missing trace/output=0、budget_violations=0，实验结束 GPU 回到 1 MiB。
[x] marginal allocator diagnostic 结论：该版本非常保守，TriePilot verified_nodes 只有
    18-20 个，约为 p99-best static 的 1.04%-1.21%，APV 约 0.10-0.11，CUDA graph token
    shape ok ratio=1.0。InstructCoder+GSM8K 上 mean TPOT 比 mean-best static budget8 慢
    0.15-0.21ms，但 p99 TPOT 快 3.16-3.22ms；CNN/DailyMail+random 上 mean TPOT 比
    mean-best static budget8 快 5.87-5.96ms，p99 与 p99-best static budget2 基本持平
    （慢 0.06-0.49ms）。结论：per-regime marginal upgrade 可以在保持/改善 p99 与 verifier
    waste 的同时避免 static fallback 的 verified-node 爆炸，但当前 gate 明显偏保守，尚不能直接替代
    Session 5 larger-budget/out64 多 seed 结论。
[x] A100 上完成 out64 / multi-seed / multi-ratio marginal allocator 扩展诊断：Qwen3-8B，
    四组 mixed pairs（InstructCoder+GSM8K、InstructCoder+ShareGPT、JSON/tool-call+ShareGPT、
    CNN/DailyMail+random），ratio=50/50、20/80、80/20，B_batch=100/160，
    seed=20260513/20260514，NUM_PROMPTS=64、sharegpt_output_len=64，固定
    CUDA_GRAPH_MAX_BS=8 / MAX_RUNNING_REQUESTS=8，shape_bucket_mode=batch_max，只跑当前
    triepilot_allocation marginal path，并与既有 Session 5 batch_global_budget2/4/8/16 与旧
    shape-bucket TriePilot 结果拼表比较。结果保存在
    /root/TriePilot/runs/20260514_2230_session6_marginal_out64_multiratio_multiseed。
[x] marginal out64 扩展诊断完整性检查：combined_marginal_out64_multiratio.csv 共 48/48 行，
    comparison_marginal_vs_existing_out64.csv 共 48 行；raw trace 检查覆盖 26066 条 step events；
    missing trace=0、missing bench output=0、budget violation=0、min cuda_graph_token_shape_ok_ratio=1.0；
    24/24 个 seed×pair×ratio 的 B=100 与 B=160 verified_nodes 完全相同，说明当前大 B_batch
    不是限制项，marginal gate 才是限制项；实验结束 GPU 回到 1 MiB。
[x] marginal out64 扩展诊断结论：相对旧 shape-bucket TriePilot，当前 marginal path 平均
    mean TPOT 快 1.57ms、p99 TPOT 快 0.31ms，verified-node ratio=0.1622，mean wins=46/48、
    p99 wins=29/48；相对 tuned batch-global static，mean TPOT 平均慢 0.44ms，但 p99 TPOT
    快 2.24ms，verified-node ratio=0.00152，p99 wins=48/48、mean wins=16/48。结论：
    per-regime marginal upgrade 已能稳定保住 p99 与 verifier waste 优势，并改善旧 shape-bucket
    path；但当前 2→4 gate 仍偏保守，B_batch under-use 明显，下一轮应基于 APV/regime profile
    做 high-confidence regime 的小步放宽，而不是回到 batch-level static fallback。
[x] A100 上完成 low-accept cap4 gate 第一轮负面 live sweep：先临时实现 env-controlled
    低 accept / 非 low-match regime 的 cap4 gate（TRIEPILOT_LOW_ACCEPT_CAP4_SCORE_THRESHOLD=0.25，
    TRIEPILOT_LOW_ACCEPT_CAP4_MIN_CONFIDENCE=0.40）并在 A100 上完成 RED/GREEN targeted pytest；
    随后用 Qwen3-8B 跑 control vs relaxed 小型 live sweep，覆盖 InstructCoder+GSM8K 与
    CNN/DailyMail+random，ratio=0.2/0.5，B_batch=100，NUM_PROMPTS=64，sharegpt_output_len=64，
    固定 CUDA_GRAPH_MAX_BS=8 / MAX_RUNNING_REQUESTS=8 / shape_bucket_mode=batch_max，seed=20260515。
    结果保存在 /root/TriePilot/runs/20260515_0025_session6_cap4_gate_live_sweep。
[x] cap4 gate live sweep 完整性与结论：8 个 summary row、4 个 paired comparison，raw trace
    budget violation=0、cuda_graph_token_shape_ok_ratio=1.0、实验结束 GPU 回到 1 MiB。relaxed gate
    确实触发 6 个 cap4 event（主要是 R_high_match_high_branch_low_accept_low_load），但 4/4
    都增加 verified nodes（平均 +3），mean TPOT 仅 1/4 胜出且平均 +0.020ms，p99 TPOT 0/4
    胜出且平均 +0.137ms，APV 平均 +0.0041。结论：仅靠 score/confidence 打开 low-accept
    cap4 不安全；临时 env gate 代码与测试已回退，A100 回退后 targeted pytest 23 passed /
    2 subtests passed，py_compile 覆盖 budget.py、recorder.py、ngram_worker.py、server_args.py。
[x] A100 上完成 Step 1 match-type / tree-shape / out64 补充验证的 runner 加固：scripts/run_server.sh
    支持 NGRAM_MATCH_TYPE 并透传 --speculative-ngram-match-type；a100_step1_static_tiers.sh
    将 ngram_match_type 写入 config / tier_summary；a100_step1_static_tier_library_driver.sh
    支持 TRIEPILOT_STEP1_MATCH_TYPES 与 TRIEPILOT_STEP1_RUN_ID_PREFIX。A100 targeted pytest
    tests/test_launch_scripts.py 通过 5 passed / 2 subtests passed；bash -n 覆盖 run_server.sh、
    a100_step1_static_tiers.sh、a100_step1_static_tier_library_driver.sh；sglang env 下 py_compile
    覆盖 third_party/sglang_flex/python/sglang/srt/server_args.py。
[x] A100 上完成 Step 1 match-type / tree-shape / out64 小型补充 sweep：Qwen3-8B，
    datasets=json_tool/shared_prefix，match types=PROB/BFS，shapes=mw12_b4 与 mw24_b8，
    budgets=0/2/4/8/16，NUM_PROMPTS=32，sharegpt_output_len=64，max_concurrency=8，
    request_rate=8，seed=20260515。结果保存在
    /root/TriePilot/runs/20260515_0135_step1_match_type_tree_shape_out64_seed20260515。
[x] Step 1 match-type / tree-shape / out64 完整性与结论：combined_tier_summary.csv 共 40/40 行，
    completed_runs=8/8，missing budget group=0，missing bench/trace file=0，非零 budget empty trace=0，
    driver log 无 Traceback / RuntimeError / CUDA OOM / missing output，实验结束 GPU 回到 1 MiB。
    PROB 与 BFS 在本轮两个 workload、两个 shape 上 accepted_per_verified_node 与 verified_nodes 完全一致，
    非零预算 paired comparison 中 PROB mean-TPOT wins=8/16、p99 wins=8/16，平均 PROB-BFS
    delta 仅 mean +0.0038ms、p99 -0.0036ms，属于噪声级；暂不把 PROB 作为默认 match mode。
    json_tool 上 best APV budget=2，AR/budget0 仍是 mean 与 p99 TPOT 最优；shared_prefix 上
    best APV budget=4，best mean TPOT budget=8，p99 TPOT 仍由 AR/budget0 最优。mw24_b8
    对 shared_prefix budget4 有正向信号（APV +0.0187，mean TPOT -0.34~-0.44ms，p99 -0.76~-1.12ms），
    但跨全部非零 pair 平均 APV -0.0026、mean TPOT +0.265ms；更复杂 tree shape 需要按
    workload/regime 条件使用，不能全局替换 mw12_b4。
[x] A100 上完成 Session 7 full baseline preflight：Qwen3-8B，四组 mixed pairs
    （InstructCoder+GSM8K、InstructCoder+ShareGPT、JSON/tool-call+ShareGPT、CNN/DailyMail+random），
    ratio=50/50、20/80、80/20，B_batch=100/160，seed=20260515，NUM_PROMPTS=64，
    sharegpt_output_len=64，固定 CUDA_GRAPH_MAX_BS=8 / MAX_RUNNING_REQUESTS=8，
    shape_bucket_mode=batch_max，materialize fill-filtered=1。方法覆盖
    batch_global_budget2/4/8/16、equal_budget_allocation、random_budget_allocation、
    match_depth_greedy、accept_ema_greedy 与当前 triepilot_allocation marginal path。结果保存在
    /root/TriePilot/runs/20260515_0730_session7_full_baseline_preflight_seed20260515。
[x] Session 7 preflight 完整性与结论：session5_sweep_summary.csv 共 216/216 行，
    24/24 个 combo、每个 combo 9 个方法齐全；raw bench output=216、raw step trace=216，
    trace events=88321，missing/empty file=0，bench bad success=0，budget violation=0，
    min cuda_graph_token_shape_ok_ratio=1.0，min cuda_graph_ratio=1.0，driver log 无
    Traceback / RuntimeError / CUDA OOM / missing trace / B_batch exceeded，实验结束后 GPU 回到
    1 MiB 且无残留 SGLang/bench 进程。p99 TPOT winner：TriePilot 24/24；mean TPOT winner：
    batch_global_budget8 为 17/24，batch_global_budget4 为 3/24，TriePilot 为 4/24（均在
    CNN/DailyMail+random）。TriePilot 相对 best-mean baseline 的 mean TPOT delta 平均
    +1.025ms、最大 +2.724ms；相对 best-p99 baseline 的 p99 delta 为 0；相对最省 verified-node
    static tier 的 verified-node ratio 平均 0.00389、最大 0.00630。结论：当前 marginal
    TriePilot 在 single-seed 全 baseline preflight 中稳定占优 p99 和 verifier waste，但 mean
    TPOT 仍主要落后 tuned batch-global budget8/4；该结果是正式主表前的完整 baseline 预检，
    不是最终 multi-seed 主表。
[x] A100 上完成 Session 7 AR/no-spec supplement：Qwen3-8B，四组 mixed pairs、ratio=50/50、
    20/80、80/20，seed=20260515，NUM_PROMPTS=64，sharegpt_output_len=64，固定
    CUDA_GRAPH_MAX_BS=8 / MAX_RUNNING_REQUESTS=8，使用与 preflight 相同的
    materialize fill-filtered=1 配置。结果保存在
    /root/TriePilot/runs/20260515_1330_session7_ar_no_spec_supplement_seed20260515。
    完整性：raw bench output=12/12，ar_no_spec_summary.csv=12 行，completed 全部为 64，
    实验结束 GPU 回到 1 MiB，且无残留 SGLang/bench 进程。AR 平均 mean TPOT=17.65ms，
    平均 p99 TPOT=21.24ms；与同 seed Session 7 preflight 的 24 个 pair×ratio×B 对照相比，
    AR 相对 best-mean online baseline 的 mean TPOT 平均快 0.40ms、16/24 胜出；
    相对 best-p99 online baseline 的 p99 TPOT 平均快 3.17ms、24/24 胜出。结论：
    AR/no-spec 必须作为强公平基线进入正式主表；当前 TriePilot 的 p99 优势只相对
    NGRAM / allocator online baselines 成立，不能在未纳入 AR 时表述为全局 p99 最优。
[x] A100 上完成 Session 7 focused multi-seed online baseline 对齐第一块：Qwen3-8B，
    seed=20260516，四组 mixed pairs、ratio=50/50、20/80、80/20，B_batch=100/160，
    NUM_PROMPTS=64，sharegpt_output_len=64，固定 CUDA_GRAPH_MAX_BS=8 /
    MAX_RUNNING_REQUESTS=8，shape_bucket_mode=batch_max，materialize fill-filtered=1。
    为避免重复 4.5 小时级 full baseline，本轮采用 AGENTS 允许的精简 multi-seed 对齐：
    只跑强静态 baseline batch_global_budget2/4/8 与 triepilot_allocation。结果保存在
    /root/TriePilot/runs/20260515_1348_session7_focused_multiseed_seed20260516。
[x] Session 7 focused seed=20260516 完整性与结论：session5_sweep_summary.csv 共 96/96 行，
    24/24 个 combo、每个 combo 4 个方法齐全；missing/empty file=0，bench bad success=0，
    budget violation=0，trace events=43852，min cuda_graph_ratio=1.0，
    min cuda_graph_token_shape_ok_ratio=1.0，实验结束 GPU 回到 1 MiB。p99 TPOT winner：
    TriePilot 20/24，batch_global_budget4 为 2/24，batch_global_budget2 为 2/24；
    mean TPOT winner：batch_global_budget8 为 14/24，batch_global_budget4 为 4/24，
    TriePilot 为 6/24。TriePilot 相对同组 best static 的 mean TPOT delta 平均 +0.759ms，
    p99 TPOT delta 平均 -1.693ms、最差 +0.357ms、最好 -9.244ms；相对最省 verified-node
    static tier 的 verified-node ratio 平均 0.00446、最大 0.00688。将 seed=20260515 full
    preflight 过滤到同四个方法并与 seed=20260516 合并后，TriePilot 在 48 个组合中
    p99 胜出 44/48，mean 胜出 10/48，平均 p99 delta=-2.000ms，平均 mean delta=+0.758ms。
    结论：第二个 seed 支持 TriePilot 的 p99/waste 主张，但不能写成 p99 绝对全胜或 mean 全胜。
[x] A100 上完成 Session 7 focused multi-seed online baseline 第三块：Qwen3-8B，
    seed=20260517，四组 mixed pairs、ratio=50/50、20/80、80/20，B_batch=100/160，
    NUM_PROMPTS=64，sharegpt_output_len=64，固定 CUDA_GRAPH_MAX_BS=8 /
    MAX_RUNNING_REQUESTS=8，shape_bucket_mode=batch_max，materialize fill-filtered=1。
    方法覆盖 batch_global_budget2/4/8 与 triepilot_allocation。结果保存在
    /root/TriePilot/runs/20260515_1715_session7_focused_multiseed_seed20260517。
[x] Session 7 focused seed=20260517 完整性与结论：session5_sweep_summary.csv 共 96/96 行，
    24/24 个 combo、每个 combo 4 个方法齐全；missing/empty file=0，bench bad success=0，
    budget violation=0，trace events=43108，raw bench output=96，raw step trace=96，
    min cuda_graph_ratio=1.0，min cuda_graph_token_shape_ok_ratio=1.0，driver log 无
    Traceback / RuntimeError / CUDA OOM / missing trace / B_batch exceeded / server health error，
    实验结束 GPU 回到 1 MiB 且无残留 SGLang/bench 进程。p99 TPOT winner：TriePilot 24/24；
    mean TPOT winner：batch_global_budget8 为 11/24，batch_global_budget4 为 9/24，
    TriePilot 为 4/24。TriePilot 相对同组 best static 的 mean TPOT delta 平均 +0.526ms，
    p99 TPOT delta 平均 -2.993ms；相对最省 verified-node static tier 的 verified-node ratio
    平均 0.00382。将 seed=20260515 full preflight 过滤到同四个方法并与 seed=20260516/20260517
    合并后，TriePilot 在 72 个组合中 p99 胜出 68/72，mean 胜出 14/72，
    平均 p99 delta=-2.331ms，平均 mean delta=+0.681ms，平均 verified-node ratio=0.00406。
[x] A100 上完成 Session 7 AR/no-spec multi-seed supplement：复用 seed=20260515 的同配置
    AR/no-spec 入口，补跑 seed=20260516 与 seed=20260517；四组 mixed pairs、ratio=50/50、
    20/80、80/20，NUM_PROMPTS=64，sharegpt_output_len=64，固定 CUDA_GRAPH_MAX_BS=8 /
    MAX_RUNNING_REQUESTS=8，materialize fill-filtered=1。seed=20260516 结果保存在
    /root/TriePilot/runs/20260515_1935_session7_ar_no_spec_supplement_seed20260516；
    seed=20260517 结果保存在
    /root/TriePilot/runs/20260515_2005_session7_ar_no_spec_supplement_seed20260517；
    三 seed 汇总保存在 /root/TriePilot/runs/20260515_2010_session7_ar_no_spec_multiseed_summary。
    注意：复用脚本在解析 cnn_dailymail_random_r0pX 时曾用 split("_r") 误切 random，
    已基于原始 bench output 用 rsplit("_r", 1) 重新生成 seed16/17 的 ar_no_spec_summary.csv。
[x] Session 7 AR/no-spec multi-seed 完整性与结论：三 seed 共 36/36 行，所有 completed=64，
    missing/empty output=0，实验结束 GPU 回到 1 MiB 且无残留 SGLang/bench 进程。
    AR 平均 mean TPOT：seed20260515 为 17.65ms、seed20260516 为 17.42ms、
    seed20260517 为 17.33ms；平均 p99 TPOT 分别为 21.24ms、21.12ms、20.60ms。
    与同 seed focused online baselines（batch_global_budget2/4/8 + TriePilot，按 B=100/160
    展开共 72 个组合）相比，AR mean TPOT 胜出 54/72、平均 delta=-0.662ms；
    AR p99 TPOT 胜出 72/72、平均 delta=-3.314ms。相对 TriePilot，AR mean TPOT 胜出
    68/72、平均 delta=-1.612ms；AR p99 TPOT 胜出 72/72、平均 delta=-3.326ms。
    结论：AR/no-spec 是当前 out64/cgbs8 设置下必须纳入正式主表的强基线；TriePilot 的
    稳健主张应表述为相对 NGRAM / allocator online baselines 的 p99 与 verifier-node waste 改善，
    不能表述为相对 AR 的端到端 TPOT 胜出。
[x] A100 上完成 Session 7 paper-ready replay/main-table 验证包：复用三 seed focused online
    baselines（batch_global_budget2/4/8 + TriePilot）与三 seed AR/no-spec supplement，在远端
    /root/TriePilot 生成第一版正式主表汇总、metric-wise observed replay upper bound、p99-vs-verified
    Pareto 点表/图和 TriePilot budget distribution 图。结果保存在
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary。
    覆盖 focused online rows=288、online scenarios=72、AR replicate rows=72、combined rows=360；
    missing method scenarios=0、duplicate method scenarios=0；扫描 TriePilot raw trace 72 个文件、
    39052 条 step events，budget violations=0，CUDA graph token-shape bad events=0。
    NGRAM-only p99 winners：TriePilot 68/72、batch_global_budget2 2/72、batch_global_budget4 2/72；
    NGRAM-only mean winners：batch_global_budget8 42/72、batch_global_budget4 16/72、
    TriePilot 14/72。TriePilot 相对 best static budget2/4/8 的平均 p99 delta=-2.331ms，
    平均 mean delta=+0.681ms，verified-node ratio 相对最低 verified static 平均 0.0041。
    加入 AR 后，AR/no-spec p99 72/72 胜出、mean 54/72 胜出。注意：
    observed_replay_oracle_ngram_metricwise.csv 是现有 online NGRAM 结果上的 metric-wise
    observed replay upper bound，不是可部署 oracle allocator，也不包含 AR/no-spec。
[x] A100 上完成 Session 7.5 near-paper baseline feasibility scan：覆盖 TETRIS、AdaServe、
    SuffixDecoding、SAM-Decoding、Token Recycling，并按同一 A100、同一 Qwen3-8B、
    同一 TriePilot mixed workload、同一 SGLang 0.5.6.post2 serving stack 的强公平条件判断
    是否能进入主公平表。结果保存在
    /root/TriePilot/runs/20260515_2115_session75_near_paper_feasibility。
    产物包含 near_paper_feasibility_matrix.csv/json、analysis_near_paper_feasibility.md、
    integrity_near_paper_feasibility.json、network_probe.txt。完整性：rows=5、
    fair_same_stack_count=0、main_table_runnable_count=0。A100 网络诊断显示 arxiv.org 可访问，
    但 github.com 与 huggingface.co 从远端超时；因此 repo_reachable 只表示 A100 侧网络探测结果，
    不表示公开仓库不存在。结论：没有方法满足主公平表的同栈在线复现条件；SuffixDecoding /
    SAM-Decoding 是最近的 model-free 邻近工作，但官方实现分别偏 vLLM plugin / standalone
    Transformers-SpecBench 路径，适合 discussion、future port 或 separate-stack appendix；
    TETRIS / AdaServe 作为强近邻边界讨论，Token Recycling 可作 train-free 近邻讨论但不是
    probability-free。
[x] A100 上完成 Session 7.5 外部源码同步与小范围验证：本机从 GitHub 拉取并同步到 A100 的
    公开源码包括 TETRIS、ArcticInference/SuffixDecoding、SAM-Decoding、Token Recycling，
    本地 manifest 记录 commit，远端源码位于
    /root/TriePilot/external/near_paper_sources_20260515，验证产物保存在
    /root/TriePilot/runs/20260515_2130_session75_external_code_smoke。
    commit 分别为 TETRIS acb77de80152、ArcticInference fba641f8ffba、SAM-Decoding
    aaf939819223、Token Recycling 1b4c05cc642d。远端只做小范围 source/import/proposer-level
    smoke，不改动现有 sglang/vllm conda env；Arctic 编译所需 cmake/nanobind 安装在 run-local
    pydeps 目录。
[x] 外部源码小验证结论：selected py_compile 通过；Arctic/SuffixDecoding 的 suffix-decoding
    native extension 用 run-local deps 编译通过，合成 suffix-cache micro 中 8/8 draft tokens
    被 synthetic response 接受，match_len=5，latency≈14.47us；SAM-Decoding 直接加载
    StaticSAM 单文件绕开旧 Transformers 包级导入，合成 StaticSAM micro 中 accepted prefix=2，
    states=27，match_len=5，latency≈908.94us；Token Recycling tree template 2.2.2
    验证为 80 draft nodes、max_depth=6，topk=8 的候选矩阵在 Vicuna vocab 下约 1.95MB，
    在 Qwen3 vocab 下约 9.27MB。
[x] 外部源码小验证边界：TETRIS import 在当前 vLLM 0.16 / Triton 环境下失败，原因是该
    vLLM fork 期望旧 API / build artifacts（vllm._C、triton.runtime.cache.default_cache_dir）；
    Token Recycling full entry 需要 FastChat/SpecBench/自定义 LLaMA KV 路径，当前环境未安装
    FastChat，tiny random LLaMA forward 还触发 PyTorch/Transformers mask dtype 兼容问题；
    SAM 包级导入依赖旧 Transformers StaticCache 路径。结论：这些结果能证明近邻 model-free
    proposer 在隔离小合成场景下可以产生 draft，但仍不是同一 SGLang serving stack 的
    end-to-end 公平复现；主表仍不应混入这些外部方法，除非后续做正式 port。
[x] A100 上完成 baseline 专属 conda 环境第一轮搭建：按用户要求为外部 baseline 建隔离环境，
    不污染现有 /root/anaconda3/envs/sglang 与 /root/anaconda3/envs/vllm。环境路径为
    /root/TriePilot/baseline_envs/arctic_suffix、/root/TriePilot/baseline_envs/sam_decoding、
    /root/TriePilot/baseline_envs/token_recycling、/root/TriePilot/baseline_envs/tetris；
    过程和日志保存在 /root/TriePilot/runs/20260515_2145_session75_baseline_conda_envs。
    Arctic/SuffixDecoding、SAM-Decoding、Token Recycling 的 env-level import / entry smoke
    与 proposer/tiny-forward micro 已跑通；TETRIS 的独立 Python 3.11 env 安装 torch 2.5.1、
    torch-scatter、Triton 3.1 与 repo common requirements 后，vLLM fork editable build
    已成功，`vllm._C` 可 import，`select_proposals_no_priority` 可 import。
[ ] 下一步继续 TETRIS 独立环境小范围验证：在
    /root/TriePilot/baseline_envs/tetris 中先跑 synthetic proposal-selection micro 的修正版结果检查，
    再尝试 repo 自带 benchmark/DSD 脚本的最小化 smoke（优先 1-2 条 prompt、短 output、单 GPU、
    不改 TriePilot/SGLang 环境）。如果 vLLM engine 真实启动受模型、FastChat、数据集或旧接口阻塞，
    记录为 TETRIS separate-stack boundary；如果能启动，则只做小范围 end-to-end，对照口径明确为
    separate-stack smoke，不放入 SGLang 主公平表。
[x] A100 上完成 Session 8 Strategy Bank workload-shift 第一轮 smoke：Qwen3-8B，单个
    SGLang server 不重启，按 InstructCoder → GSM8K → JSON/tool-call → InstructCoder
    四阶段顺序运行，每阶段 16 prompts、out64、max_concurrency=8、request_rate=8、
    B_batch=100、shape_bucket_mode=batch_max、cgbs8。结果保存在
    /root/TriePilot/runs/20260516_session8_strategy_bank_shift_smoke。
    完整性检查：raw_step_events.jsonl 共 795 条，phase_count=4，四阶段 bench 均
    completed=16/16，budget violation=0，min cuda_graph_token_shape_ok_ratio=1.0，
    server.log 无 Traceback / CUDA error，实验结束 GPU 回到 1 MiB 且无残留
    SGLang/bench 进程。
[x] Session 8 Strategy Bank smoke 结论：在线 regime reuse 链路已跑通，cold_code 首阶段
    发现 9 个新 regime，shift_math 再发现 6 个新 regime，shift_json 与 warm_code_reuse
    均为 0 个新 regime；strategy_bank_hit_rate 从 cold_code 0.993 提升到 warm_code_reuse
    1.000，warm_code_reuse 的 first5 hit rate 相对 cold_code +0.20。但当前反馈策略会在
    连续低接受后把 preferred budget 压到 0，shift_json 与 warm_code_reuse 的 verified_nodes
    均为 0，说明 Strategy Bank 复用机制可用但当前 warm path 过度保守，尚不能作为最终
    recovery-time 收益图。下一步 Session 8 应做带 exploration floor / 最小验证预算或
    per-regime 冷却恢复的对照，再扩大样本和 seed。
[x] A100 上完成 Session 8 exploration floor / 最小验证预算 proxy 小型对照：复用同样
    InstructCoder → GSM8K → JSON/tool-call → InstructCoder 四阶段 shift，每阶段 16 prompts、
    out64、max_concurrency=8、request_rate=8、shape_bucket_mode=batch_max、cgbs8、seed=20260516；
    对比当前 triepilot_allocation B_batch=100、per-request equal floor proxy B_batch=16
    （约每轮总 floor=16）与 batch_global_budget2。结果保存在
    /root/TriePilot/runs/20260516_session8_floor_proxy_diagnostic。
    完整性检查：floor_proxy_phase_summary.csv 共 12 行，raw step events 共 2016 条，
    三个 policy × 四个 phase 的 bench 均 completed=16/16，budget violation=0，
    min cuda_graph_token_shape_ok_ratio=1.0，server log 无 Traceback / RuntimeError /
    CUDA error / illegal memory access，实验结束 GPU 回到 1 MiB。
[x] Session 8 floor proxy 结论：当前 Strategy Bank 在 warm_code_reuse 阶段 verified_nodes=0、
    accepted_tokens=0、mean/p99 TPOT=17.97/18.75ms，确认预算塌缩会错过重复代码阶段的
    spec 机会。equal_floor_B16 在 warm_code_reuse 验证 1232 nodes、接受 928 tokens、
    APV=0.7532、mean/p99 TPOT=1.95/3.44ms；batch_global_budget2 验证 1056 nodes、
    接受 511 tokens、APV=0.4839、mean/p99 TPOT=9.01/9.59ms。结论：最小验证预算能恢复
    warm regime 的接受收益，但全局/均分 floor 会显著增加 verified nodes，不能直接作为最终
    Strategy Bank recovery 机制；下一步应做 selective per-regime floor / cooldown recovery，
    只对曾经正收益或结构稳定的 regime 保留少量探测预算。
[x] A100 上完成 Session 8 selective per-regime floor / cooldown recovery 第一轮机制对照
    与负面验证：新增 env-controlled selective recovery（默认关闭，仅当
    TRIEPILOT_SELECTIVE_RECOVERY=1 时启用），StrategyRecord 记录 positive_observations、
    max_observed_gain_per_node、zero_gain_streak、last_recovery_probe_step，recorder 写出
    recovery_probe_flags / recovery_probe_count 等 telemetry；A100 targeted pytest
    tests/test_sglang_triepilot_telemetry.py 通过 20 passed，sglang env py_compile 与
    scripts/remote/a100_session8_strategy_bank_recovery.sh bash -n 均通过。
    第一轮 live 对照保存在
    /root/TriePilot/runs/20260516_session8_selective_recovery_seed20260516：
    no-recovery、selective-recovery、equal_floor_B16、batch_global_budget2 四个 policy
    均完成四阶段 shift，summary_rows=16、trace_events_total=2510、completed=16/16、
    budget violation=0、min cuda_graph_token_shape_ok_ratio=1.0、server_log_error_policies=[]。
    该轮发现 selective recovery 的门过窄，recovery_probe_count=0，行为与 no-recovery
    完全等价。
[x] A100 上完成 selective recovery v2 小型复跑：将 recovery 条件放宽为 high_match 且该
    regime 曾有正收益，结果保存在
    /root/TriePilot/runs/20260516_session8_selective_recovery_v2_seed20260516。
    完整性检查：summary_rows=8、trace_events_total=1417、completed=16/16、budget violation=0、
    min cuda_graph_token_shape_ok_ratio=1.0、server_log_error_policies=[]，实验结束 GPU 回到
    1 MiB 且无残留 SGLang / bench 进程。v2 的 probes 已触发：
    cold_code/shift_math/shift_json/warm_code_reuse 的 recovery_probe_count 分别为
    9/11/11/6；但 warm_code_reuse 只从 no-recovery 的 verified=4、accepted=1、
    mean/p99 TPOT=18.17/19.01ms 变为 verified=16、accepted=2、mean/p99 TPOT=18.11/18.97ms，
    而第一轮 equal_floor_B16 的 warm_code_reuse 为 verified=1266、accepted=809、
    mean/p99 TPOT=4.58/7.82ms，batch_global_budget2 为 verified=1074、accepted=502、
    mean/p99 TPOT=10.08/12.74ms。
[x] Session 8 selective recovery 结论：仅依赖 “regime 历史正收益 + high_match +
    cooldown 探针” 的选择性恢复太弱，虽然能以极低 verified-node 成本触发 probes，但无法
    恢复 warm_code_reuse 的投机收益；不能把该机制作为最终 Strategy Bank recovery，也不要
    基于当前结果画正式 recovery-time 图。下一步应转向更强的恢复信号：例如基于 raw trace
    replay 学习 phase/regime 的 positive accept 触发条件，或引入连续 match/branch 稳定性、
    request-local early accept probe、以及 capped-but-denser warm-start floor，再做 A100 小型
    live sweep。
[x] A100 上完成 Session 8 capped-but-denser warm-start floor 小型 live sweep：复用同一
    InstructCoder → GSM8K → JSON/tool-call → InstructCoder 四阶段 shift，每阶段 16 prompts、
    out64、max_concurrency=8、request_rate=8、shape_bucket_mode=batch_max、cgbs8、
    seed=20260516；对 selective recovery 使用 RECOVERY_BUDGET=8、
    RECOVERY_COOLDOWN_STEPS=1、RECOVERY_MIN_GAIN=0.20，并同跑 no-recovery、
    equal_floor_B16、batch_global_budget2。结果保存在
    /root/TriePilot/runs/20260516_session8_dense_recovery_floor_seed20260516。
    完整性检查：phase_summary.csv 共 16 行，trace_events_total=2502，四个 policy × 四个
    phase 的 bench 均 completed=16/16，budget violation=0，
    min cuda_graph_token_shape_ok_ratio=1.0，server_log_error_policies=[]，实验结束 GPU 回到
    1 MiB 且无残留 SGLang / bench 进程。
[x] capped-but-denser warm-start floor 结论：更密的 selective probes 确实显著提高验证量，
    warm_code_reuse 从 no-recovery 的 verified=4、accepted=1、mean/p99 TPOT=18.22/19.24ms，
    提升到 verified=136、accepted=26、mean/p99 TPOT=18.02/19.52ms，recovery_probe_count=31；
    但仍远低于 equal_floor_B16 的 verified=1266、accepted=811、mean/p99 TPOT=4.51/7.74ms，
    也低于 batch_global_budget2 的 verified=1078、accepted=500、mean/p99 TPOT=10.04/13.02ms。
    说明仅靠“历史正收益 high_match regime 的更密探针”仍无法恢复 warm reuse；下一步不要继续只调
    recovery budget / cooldown，应转向 request-local early accept probe 或 trace-driven
    positive-trigger replay，将恢复预算给到已经在当前 request 上证明有连续接受的请求。
[x] A100 上完成 Session 8 request-local early accept probe 机制设计、TDD 验证与小型 live sweep：
    新增默认关闭的 TRIEPILOT_REQUEST_LOCAL_RECOVERY 机制；当 high-match request 所属 regime
    已塌缩到低 score / preferred_budget=0 时，允许每个 request 本地最多少量 early probes，
    如果当前 request 后续 accept_ema 转正，则由 request-local 信号接回后续预算，而不是继续
    依赖 regime-level cooldown。A100 RED/GREEN 验证覆盖新增 collapsed high-match request
    探针测试与 recorder telemetry 字段；远端 targeted pytest
    tests/test_sglang_triepilot_telemetry.py / tests/test_launch_scripts.py 通过 26 passed /
    2 subtests passed，sglang env py_compile 覆盖 triepilot/sglang_integration/budget.py 与
    recorder.py，scripts/remote/a100_session8_strategy_bank_recovery.sh bash -n 通过。
    live sweep 仍使用 InstructCoder → GSM8K → JSON/tool-call → InstructCoder 四阶段 shift，
    每阶段 16 prompts、out64、max_concurrency=8、request_rate=8、shape_bucket_mode=batch_max、
    cgbs8、seed=20260516；对比 no-recovery、request-local recovery、equal_floor_B16、
    batch_global_budget2。结果保存在
    /root/TriePilot/runs/20260516_session8_request_local_recovery_seed20260516。
    完整性检查：phase_summary.csv 共 16 行，trace_events_total=2456，四个 policy × 四个
    phase 的 bench 均 completed=16/16，budget violation=0，min cuda_graph_token_shape_ok_ratio=1.0，
    server_log_error_policies=[]，实验结束 GPU 回到 1 MiB 且无残留 SGLang / bench 进程。
[x] request-local early accept probe 结论：这是第一轮能显著恢复 warm_code_reuse 的
    Strategy Bank recovery 机制。warm_code_reuse 从 no-recovery 的 verified=4、accepted=1、
    mean/p99 TPOT=18.36/19.44ms，提升到 request-local recovery 的 verified=988、
    accepted=394、APV=0.3988、mean/p99 TPOT=12.43/15.41ms，request_local_probe_count=37；
    相比 no-recovery，mean TPOT 改善 5.93ms、p99 改善 4.04ms。它仍弱于
    equal_floor_B16（verified=1266、accepted=809、mean/p99 TPOT=4.57/7.79ms）和
    batch_global_budget2（verified=1074、accepted=502、mean/p99 TPOT=10.14/12.81ms），但
    verified nodes 低于二者且 APV 高于 batch_global_budget2 的部分前置阶段。下一步应围绕
    request-local probe/sustain 预算做多 seed recovery 图和 trace-driven positive-trigger replay，
    而不是继续调 regime-level recovery cooldown。
[x] A100 上完成 Session 8 request-local recovery probe/sustain 参数扫：Qwen3-8B，
    InstructCoder → GSM8K → JSON/tool-call → InstructCoder 四阶段 shift，seed=20260516，
    每阶段 24 prompts、out64、max_concurrency=8、request_rate=8、B_batch=100、
    shape_bucket_mode=batch_max、cgbs8。三组配置均只对比 triepilot_no_recovery_B100 与
    triepilot_request_local_recovery_B100：probe1_thr075_sustain2、
    probe2_thr050_sustain2、probe4_thr050_sustain2。结果保存在
    /root/TriePilot/runs/20260516_session8_request_local_param_sweep_seed20260516。
    完整性检查：config_count=3、phase summary=24 行、recovery comparison=12 行，
    all_bench_completed=true、all_traces_nonempty=true、budget_violation_count=0、
    min_cuda_graph_token_shape_ok_ratio=1.0、server_log_error_configs=[]，实验结束 GPU 回到
    1 MiB 且无残留 SGLang / bench 进程。A100 远端 targeted pytest / py_compile 在实验前
    通过：28 passed / 2 subtests passed。
[x] request-local recovery 参数扫结论：probe2_thr050_sustain2 是当前最稳的正式多 seed
    候选。相对 no-recovery，它在 4 个 phase 上平均增加 verified_nodes=622、accepted=213.75，
    APV delta=+0.2468，mean TPOT delta=-1.98ms，p99 TPOT delta=-0.52ms，mean/p99 wins
    分别为 3/4 与 4/4；warm_code_reuse 上 mean/p99 TPOT delta=-6.26ms/-1.65ms，
    request_local_probe_count=76。probe1_thr075_sustain2 探测较少但 p99 平均回退
    +0.47ms；probe4_thr050_sustain2 继续增加 probes 到 417，但 p99 平均回退 +0.08ms，
    说明过探测开始吃掉尾部收益。下一步正式 Strategy Bank recovery 图优先使用
    probe2_thr050_sustain2，保留 probe4_thr050_sustain2 作为 over-probing 对照，不再继续
    只调 regime-level recovery cooldown。
[x] A100 上完成 Session 8 probe2 request-local recovery 正式多 seed 扩展与 recovery-time /
    positive-trigger replay 汇总：复用 seed=20260516 的 probe2 参数扫结果，并新增
    seed=20260518、seed=20260519 live sweep；两组新增 seed 均使用 Qwen3-8B、
    InstructCoder → GSM8K → JSON/tool-call → InstructCoder 四阶段 shift、每阶段
    24 prompts、out64、max_concurrency=8、request_rate=8、B_batch=100、
    shape_bucket_mode=batch_max、cgbs8，对比 triepilot_no_recovery_B100、
    triepilot_request_local_recovery_B100、equal_floor_B16、batch_global_budget2。
    request-local 配置固定为 probe_budget=2、max_probes=2、accept_threshold=0.50、
    sustain_budget=2。新增 live 结果保存在
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_probe2_seed20260518 和
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_probe2_seed20260519；
    三 seed 汇总与图表保存在
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary。
    远端实验前验证：A100 targeted pytest 通过 26 passed / 2 subtests passed，
    scripts/remote/a100_session8_strategy_bank_recovery.sh bash -n 通过，sglang conda
    env 下 py_compile 覆盖 budget.py、recorder.py、ngram_worker.py、server_args.py。
    汇总完整性：all_source_integrity_clean=true、combined_rows=40、delta_rows=12、
    trace_replay_summary_rows=12、budget_violation_count=0、min_cuda_graph_token_shape_ok_ratio=1.0、
    server_log_error_policies/configs 为空；实验结束 GPU 回到 1 MiB，未发现残留
    SGLang / bench 进程。
[x] Session 8 正式 recovery 结论：probe2 request-local recovery 在 warm_code_reuse 上
    三 seed 全胜 no-recovery，平均 verified_delta=1527.3、accepted_delta=593.0、
    APV delta=+0.3882、mean TPOT delta=-6.31ms、p99 TPOT delta=-1.33ms，
    mean/p99 wins 均为 3/3。shift_json 也稳定改善 mean/p99（-1.43ms/-0.41ms，
    wins=3/3）；shift_math mean 稳定小幅改善（-0.30ms，wins=3/3）但 p99 基本持平
    （+0.003ms，wins=2/3）；cold_code 有轻微冷启动成本（mean/p99 +0.53/+0.26ms）。
    trace-driven positive-trigger replay 已生成 recovery_time_tpot.png、recovery_tpot_delta.png、
    positive_trigger_probe_sensitivity.png；seed=20260516 的 probe sensitivity 继续支持
    probe2_thr050_sustain2 作为默认配置，probe4_thr050_sustain2 仅作为 over-probing
    appendix/control，不作为正式默认策略。
```

尚未完成：

```text
[ ] per-request budget 已接入最小 allocator baselines，正式 TriePilot regime encoder / Strategy Bank /
    utility estimator 第一版已能在线运行并写出 telemetry；shape bucket batch_max runtime path 已通过
    A100 smoke 与同构诊断 sweep，但尚未实现 Slow Explorer，也尚未接入更复杂 tree shape。
[ ] Step 1 已完成第一版 static tier library、一组 match_window / bfs_breadth 对照，以及
    json_tool/shared_prefix 上的 match-type / tree-shape / out64 小型补充；尚未做全数据集
    match mode sweep、更大样本数、多随机种子或正式主表规模复跑。
[ ] Session 5 已完成四组 mixed pairs 的 50/50 小主表、20/80 与 80/20 single-seed
    比例敏感性扩展，并已完成 B_batch=100/160、output_len=64、NUM_PROMPTS=64 的
    single-seed larger-budget / longer-output shape-bucket 增量验证、compact path 小子集
    ablation，以及 50/50、20/80、80/20、B_batch=100/160 的新增双 seed shape-bucket
    稳健性验证；Session 7 已补 single-seed full baseline preflight、三 seed focused online
    baseline 对齐与 AR/no-spec 三 seed supplement，并已形成第一版 metric-wise observed replay
    upper bound、main table、Pareto 图与 budget distribution 图；仍需把这些结果裁剪成论文最终
    主表/图表口径，并补必要消融。
[ ] mean TPOT 优化已完成 static fallback 负面诊断、回退实现、selective fallback replay、
    per-regime marginal upgrade 最小实现、out64 / multi-seed / multi-ratio 扩展诊断，以及
    low-accept cap4 gate 第一轮负面 live sweep。
    当前 marginal gate 能显著降低 verified nodes、改善旧 shape-bucket path 并稳住 p99，但仍会
    under-use B_batch；已确认不能只用 score/confidence 放宽 low-accept cap4。后续若继续优化
    mean TPOT，应引入更强的 per-regime online 信号（例如连续 positive accept、branch/match
    稳定性、或 tree-shape 维度），而不是 batch-level fallback 或粗粒度 cap4 gate。
[ ] paper-ready 正式 baseline throughput / TPOT / wasted-node 主表第一版已经通过 A100 replay
    汇总生成，包含 AR 强基线、batch_global_budget2/4/8、TriePilot、observed replay upper bound、
    p99-vs-verified Pareto 图与 budget distribution 图；但论文最终版还需要明确主表取舍、
    图表排版、caption 口径和必要消融。
[ ] Session 8 已完成 Strategy Bank workload-shift 第一轮 smoke、floor proxy 小型对照、
    selective per-regime floor / cooldown recovery 第一轮负面机制对照，
    确认 hit/reuse telemetry、phase trace 切片、budget 约束与 CUDA graph shape 仍然可靠；
    也确认最小验证预算能恢复 warm regime 的接受收益，但全局/均分 floor verified-node 成本较高。
    当前 selective recovery probes 与 capped-but-denser warm-start floor 都能触发但恢复收益不足；
    request-local early accept probe 已在小型 live sweep 中显著恢复 warm_code_reuse，但仍未追上
    equal floor 或 batch-global budget2。request-local 参数扫已筛出 probe2_thr050_sustain2
    作为当前最稳候选；正式多 seed recovery 图与 trace-driven positive-trigger replay 已完成，
    证明 probe2 能稳定修复 warm_code_reuse 的 Strategy Bank 预算塌缩，但 cold_code 有小幅
    探测成本，且 equal floor / batch_global_budget2 仍是必须保留的强对照。Session 8 后续
    只建议整理成论文图表口径，不再继续调 recovery budget / cooldown。
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
下一步不要继续 score/confidence-only cap4 gate；本轮 A100 live sweep 已证明该放宽会增加
verified nodes 并损伤 p99。后续若继续 mean TPOT 优化，应先基于 raw trace 做更细粒度
per-regime online signal 设计（连续 positive accept、branch/match 稳定性、tree-shape 维度），
再做小型 live sweep。Step 1 match-type / tree-shape / out64 小型补充已显示 PROB 与 BFS
接受收益基本同构，mw24_b8 只在 shared_prefix budget4 有条件收益；后续不要全局切 PROB 或
宽树 shape。Session 7 single-seed full baseline preflight 已证明 runner、fill-filtered、
预算约束和完整 baseline 汇总链路可用。Session 7 AR/no-spec supplement 已补齐同 seed
非投机强基线，并显示 AR 在当前 out64/cgbs8 配置下是 p99 与 mean 都很强的对照；下一步不要
重复 single-seed preflight 或只在 NGRAM baselines 内宣称 p99 最优。seed=20260516 focused
multi-seed 对齐已经补齐强静态 budget2/4/8 与 TriePilot，证明第二个 seed 上 TriePilot 的
p99/waste 主张仍成立但不是 p99 全胜、也不是 mean 全胜。seed=20260517 focused online baseline
和 AR/no-spec 三 seed supplement 已补齐；AR 在三 seed 下相对 focused online baselines 的
p99 TPOT 72/72 胜出，因此正式主表必须同时报告 AR 强基线，并把 TriePilot 的优势限定在
NGRAM/allocator online baselines 的 p99/waste 维度。Session 7 paper-ready replay/main-table
验证包已生成：它补齐了 observed replay upper bound、main table、p99-vs-verified Pareto 图与
budget distribution 图，但 oracle 口径只能写成 metric-wise observed upper bound，不能写成
可部署 oracle allocator。Session 7.5 near-paper feasibility scan 已完成：当前没有近邻方法满足
同一 A100 / 同一模型 / 同一 workload / 同一 SGLang serving stack 的主公平表复现条件，
因此不要强行把 TETRIS、AdaServe、SuffixDecoding、SAM-Decoding 或 Token Recycling 塞入
主表；应写成 external reference / discussion，SuffixDecoding 与 SAM-Decoding 可作为
future-port 或 separate-stack appendix 备选。Session 8 Strategy Bank 正式 recovery 图已补齐后，
下一步优先进入 Session 9 消融与 controller overhead，focused tree-shape 稳健性仅作为附录材料。
Session 8 第一轮 A100 smoke 已完成：hit/reuse 很快达到 1.0，但当前反馈会把连续低接受
regime 的 preferred budget 压到 0，导致后续 phase 几乎停止验证 draft nodes。下一步不要直接
把该结果画成最终 recovery 图；应先做 exploration floor / 最小验证预算 / per-regime 恢复机制的
小型对照，再决定 Strategy Bank 正式实验口径。Session 8 floor proxy 与 selective recovery
第一轮小型对照已完成：全局/均分 floor 能恢复 warm_code_reuse，但 verified-node 成本高；
当前 selective recovery probes 虽然能触发，但 warm_code_reuse 只增加 12 个 verified nodes
与 1 个 accepted token，TPOT 基本不变。因此下一步不要继续只调 regime-level cooldown 阈值；
capped-but-denser warm-start floor 小型 live sweep 也已完成：warm_code_reuse 只恢复到
verified=136、accepted=26，mean/p99 TPOT=18.02/19.52ms，仍明显落后 equal_floor_B16 与
batch_global_budget2。request-local early accept probe 小型 live sweep 已完成：warm_code_reuse
恢复到 verified=988、accepted=394、APV=0.3988、mean/p99 TPOT=12.43/15.41ms，
明显优于 no-recovery 与 denser floor，但仍弱于 equal_floor_B16 和 batch_global_budget2。
request-local probe/sustain 参数扫与正式三 seed recovery 图已完成，probe2_thr050_sustain2
在 warm_code_reuse 上三 seed mean/p99 全胜 no-recovery，平均 mean/p99 TPOT 分别改善
6.31ms/1.33ms，并已补 trace-driven positive-trigger replay 作为机制解释。下一步不要继续
只调 recovery budget / cooldown；Session 8 应进入论文图表整理与 caption 口径收敛，
工程实验优先转向 Session 9 消融与 controller overhead（w/o trie features、w/o history、
w/o serving、w/o Strategy Bank、controller overhead），或补 TETRIS separate-stack
appendix smoke。
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
[x] selective mean-aware utility / cost model 负面诊断：基于 trace replay 与 oracle 上界评估
    selective fallback gate、shape-padding penalty、runtime-path penalty 与 workload-composition
    特征；本轮结论是 batch-level static fallback 在 p99 与 verified-node 约束下不安全。
    临时代码已删除，后续不再做 selective static fallback / batch-level fallback，转向
    per-regime marginal upgrade
[x] per-regime marginal upgrade 最小实现与小型 diagnostic：triepilot_allocation 改为 0→2/2→4
    小步边际升级，新增 budget_caps / marginal_upgrade_steps / marginal_upgrade_gains telemetry；
    A100 RED/GREEN targeted pytest、py_compile 与两组 50/50 小型 live sweep 已完成
[x] out64 / multi-seed / multi-ratio marginal allocator 扩展诊断：四组 mixed pairs、ratio=50/50、
    20/80、80/20，B_batch=100/160，seed=20260513/20260514，NUM_PROMPTS=64、
    sharegpt_output_len=64，固定 cgbs8 / max_running=8，只跑当前 triepilot_allocation
    marginal path，并与既有 Session 5 batch-global 和旧 shape-bucket TriePilot 结果拼表比较
[x] low-accept cap4 gate 负面 live sweep：临时 env gate 在 A100 上完成 RED/GREEN 后跑
    control vs relaxed 小型 live sweep；结论是 score/confidence-only cap4 会增加 verified nodes
    且损伤 p99，临时代码已回退
[x] Step 1 match-type / tree-shape / out64 补充验证：runner 支持 NGRAM_MATCH_TYPE /
    TRIEPILOT_STEP1_MATCH_TYPES；A100 上覆盖 json_tool/shared_prefix、PROB/BFS、mw12_b4/mw24_b8、
    budgets=0/2/4/8/16、NUM_PROMPTS=32、sharegpt_output_len=64。结论是 PROB 与 BFS
    accepted/verified 同构，mw24_b8 只在 shared_prefix budget4 呈现条件收益，暂不全局替换默认
    BFS + mw12_b4
[x] Session 7 full baseline preflight：四组 mixed pairs、ratio=50/50、20/80、80/20、
    B_batch=100/160、seed=20260515、NUM_PROMPTS=64、sharegpt_output_len=64，固定
    cgbs8 / max_running=8 / shape_bucket_mode=batch_max / fill-filtered=1；覆盖
    batch_global_budget2/4/8/16、equal、random、match-depth、accept-EMA 与 TriePilot。
    完整性 216/216 行、24/24 combo、budget violation=0、cuda graph shape ok=1.0。
[x] 正式 mixed workload 主表第一版：已有三 seed focused online baseline、三 seed AR/no-spec
    强基线、metric-wise observed replay upper bound、最终 baseline 汇总、p99-vs-verified Pareto 图
    与 budget distribution 图；仍需论文最终表格口径、caption 与必要消融
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
[x] selective fallback / cost-model replay：
    /root/TriePilot/runs/20260514_1832_session5_selective_fallback_replay/selective_fallback_decisions.csv
    /root/TriePilot/runs/20260514_1832_session5_selective_fallback_replay/selective_fallback_summary.json
    /root/TriePilot/runs/20260514_1832_session5_selective_fallback_replay/regime_profile.csv
    /root/TriePilot/runs/20260514_1832_session5_selective_fallback_replay/analysis_selective_fallback.md
    /root/TriePilot/runs/20260514_1832_session5_selective_fallback_replay/relaxed_p99_1_verified_256/selective_fallback_summary.json
    /root/TriePilot/runs/20260514_1832_session5_selective_fallback_replay/relaxed_p99_5_verified_256/selective_fallback_summary.json
[x] per-regime marginal upgrade diagnostic：
    /root/TriePilot/runs/20260514_2030_session6_marginal_allocator_diagnostic/session5_sweep_summary.csv
    /root/TriePilot/runs/20260514_2030_session6_marginal_allocator_diagnostic/marginal_replay_summary.json
    /root/TriePilot/runs/20260514_2030_session6_marginal_allocator_diagnostic/analysis_marginal_allocator_diagnostic.md
[x] out64 / multi-seed / multi-ratio marginal allocator 扩展诊断：
    /root/TriePilot/runs/20260514_2230_session6_marginal_out64_multiratio_multiseed/combined_marginal_out64_multiratio.csv
    /root/TriePilot/runs/20260514_2230_session6_marginal_out64_multiratio_multiseed/comparison_marginal_vs_existing_out64.csv
    /root/TriePilot/runs/20260514_2230_session6_marginal_out64_multiratio_multiseed/pair_summary_marginal_out64_multiratio.csv
    /root/TriePilot/runs/20260514_2230_session6_marginal_out64_multiratio_multiseed/integrity_marginal_out64_multiratio.json
    /root/TriePilot/runs/20260514_2230_session6_marginal_out64_multiratio_multiseed/analysis_marginal_out64_multiratio.md
[x] low-accept cap4 gate 负面 live sweep：
    /root/TriePilot/runs/20260515_0025_session6_cap4_gate_live_sweep/combined_cap4_gate_live_sweep.csv
    /root/TriePilot/runs/20260515_0025_session6_cap4_gate_live_sweep/comparison_cap4_gate_live_sweep.csv
    /root/TriePilot/runs/20260515_0025_session6_cap4_gate_live_sweep/integrity_cap4_gate_live_sweep.json
    /root/TriePilot/runs/20260515_0025_session6_cap4_gate_live_sweep/analysis_cap4_gate_live_sweep.md
[x] Step 1 match-type / tree-shape / out64 补充验证：
    /root/TriePilot/runs/20260515_0135_step1_match_type_tree_shape_out64_seed20260515/combined_tier_summary.csv
    /root/TriePilot/runs/20260515_0135_step1_match_type_tree_shape_out64_seed20260515/best_budget_by_shape_match_type.csv
    /root/TriePilot/runs/20260515_0135_step1_match_type_tree_shape_out64_seed20260515/prob_vs_bfs_pairwise.csv
    /root/TriePilot/runs/20260515_0135_step1_match_type_tree_shape_out64_seed20260515/integrity_match_type_tree_shape_out64.json
    /root/TriePilot/runs/20260515_0135_step1_match_type_tree_shape_out64_seed20260515/analysis_match_type_tree_shape_out64.md
[x] Session 7 full baseline preflight：
    /root/TriePilot/runs/20260515_0730_session7_full_baseline_preflight_seed20260515/session5_sweep_summary.csv
    /root/TriePilot/runs/20260515_0730_session7_full_baseline_preflight_seed20260515/integrity_session7_full_baseline_preflight.json
    /root/TriePilot/runs/20260515_0730_session7_full_baseline_preflight_seed20260515/analysis_session7_full_baseline_preflight.md
[x] Session 7 AR/no-spec supplement：
    /root/TriePilot/runs/20260515_1330_session7_ar_no_spec_supplement_seed20260515/ar_no_spec_summary.csv
    /root/TriePilot/runs/20260515_1330_session7_ar_no_spec_supplement_seed20260515/analysis_ar_no_spec_supplement.md
    /root/TriePilot/runs/20260515_1330_session7_ar_no_spec_supplement_seed20260515/comparison_ar_vs_session7_preflight.json
[x] Session 7 paper-ready replay/main-table：
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/paper_ready_main_table_with_ar.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/paper_ready_ngram_only_table.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/observed_replay_oracle_ngram_metricwise.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/triepilot_vs_best_static_by_scenario.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/ar_vs_online_by_scenario.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/p99_vs_verified_pareto_points.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/budget_distribution_triepilot.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/integrity_session7_paper_ready_replay.json
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/analysis_session7_paper_ready_replay.md
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/plots/p99_vs_verified_pareto.png
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/plots/budget_distribution_triepilot.png
```

### Session 6：实现 TriePilot

任务：

```text
[x] Regime Encoder
[x] Strategy Bank
[x] Utility Estimator
[x] Greedy Budget Allocator 第一版
[x] per-regime marginal upgrade allocator 第一版
[x] Telemetry update
```

产出：

```text
TriePilot 主方法结果第一版：Session 5 shape-bucket allocation + Session 6 marginal allocator diagnostic
```

### Session 7：mixed-batch 主实验

任务：

```text
[x] single-seed full baseline preflight：跑 4 组 mixed workload、3 个比例、B_batch=100/160，
    覆盖 batch-global tuned tiers 与 request-level allocator baselines，并保存完整日志
[x] AR/no-spec supplement：同 seed / 同 workload / 同 NUM_PROMPTS=64 / 同 out64 / 同 cgbs8，
    跑 4 组 mixed workload × 3 个比例，作为非投机强公平基线
[x] focused multi-seed online baseline 对齐第一块：seed=20260516，跑强静态
    batch_global_budget2/4/8 与 TriePilot，覆盖 4 组 mixed workload × 3 个比例 × B_batch=100/160
[x] focused multi-seed online baseline 对齐第二块：seed=20260517，跑强静态
    batch_global_budget2/4/8 与 TriePilot，覆盖 4 组 mixed workload × 3 个比例 × B_batch=100/160
[x] AR/no-spec 多 seed supplement：补 seed=20260516/20260517，并和 seed=20260515 拼成
    三 seed AR 强基线汇总
[x] multi-seed paper-ready 主表第一版：补 metric-wise observed replay upper bound、最终 baseline
    汇总、p99-vs-verified Pareto 图与 budget distribution 图；注意该 replay upper bound
    不是可部署 oracle allocator
```

产出：

```text
single-seed preflight 数据：
    /root/TriePilot/runs/20260515_0730_session7_full_baseline_preflight_seed20260515
AR/no-spec supplement 数据：
    /root/TriePilot/runs/20260515_1330_session7_ar_no_spec_supplement_seed20260515
focused seed=20260516 online baseline 对齐数据：
    /root/TriePilot/runs/20260515_1348_session7_focused_multiseed_seed20260516/session5_sweep_summary.csv
    /root/TriePilot/runs/20260515_1348_session7_focused_multiseed_seed20260516/integrity_session7_focused_multiseed_seed20260516.json
    /root/TriePilot/runs/20260515_1348_session7_focused_multiseed_seed20260516/analysis_session7_focused_multiseed_seed20260516.md
    /root/TriePilot/runs/20260515_1348_session7_focused_multiseed_seed20260516/comparison_focused_seed20260515_20260516.csv
focused seed=20260517 online baseline 对齐数据：
    /root/TriePilot/runs/20260515_1715_session7_focused_multiseed_seed20260517/session5_sweep_summary.csv
    /root/TriePilot/runs/20260515_1715_session7_focused_multiseed_seed20260517/integrity_session7_focused_multiseed_seed20260517.json
    /root/TriePilot/runs/20260515_1715_session7_focused_multiseed_seed20260517/analysis_session7_focused_multiseed_seed20260517.md
    /root/TriePilot/runs/20260515_1715_session7_focused_multiseed_seed20260517/comparison_focused_seed20260515_20260516_20260517.csv
AR/no-spec multi-seed 数据：
    /root/TriePilot/runs/20260515_1935_session7_ar_no_spec_supplement_seed20260516/ar_no_spec_summary.csv
    /root/TriePilot/runs/20260515_2005_session7_ar_no_spec_supplement_seed20260517/ar_no_spec_summary.csv
    /root/TriePilot/runs/20260515_2010_session7_ar_no_spec_multiseed_summary/combined_ar_no_spec_seed20260515_20260516_20260517.csv
    /root/TriePilot/runs/20260515_2010_session7_ar_no_spec_multiseed_summary/comparison_ar_vs_focused_online_seed20260515_20260516_20260517.csv
    /root/TriePilot/runs/20260515_2010_session7_ar_no_spec_multiseed_summary/analysis_ar_no_spec_multiseed.md
paper-ready replay/main-table 数据：
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/paper_ready_main_table_with_ar.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/paper_ready_ngram_only_table.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/single_seed_full_baseline_preflight_table.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/triepilot_vs_best_static_by_scenario.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/observed_replay_oracle_ngram_metricwise.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/ar_vs_online_by_scenario.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/integrity_session7_paper_ready_replay.json
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/analysis_session7_paper_ready_replay.md
budget allocation 可视化 / p99-vs-verified Pareto 图：
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/p99_vs_verified_pareto_points.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/budget_distribution_triepilot.csv
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/plots/p99_vs_verified_pareto.png
    /root/TriePilot/runs/20260515_2027_session7_paper_ready_replay_summary/plots/budget_distribution_triepilot.png
```

### Session 7.5：near-paper baselines

任务：

```text
[x] 近邻论文 feasibility scan：TETRIS、AdaServe、SuffixDecoding、SAM-Decoding、Token Recycling
[x] 判断每个方法是否能在同一 A100、同一模型、同一 workload、同一 serving stack 下公平复现
[x] 对可公平复现的方法跑小规模对齐实验，优先 model-free / draft-free / symbolic 相近方法；
    本轮 fair_same_stack_count=0，因此没有启动外部方法在线小表
[x] 对依赖 draft logits / speculator probability / SLO probability model 且无法公平复现的方法，
    写成 external reference 或 discussion，不放主公平表
```

优先级：

```text
1. SuffixDecoding / SAM-Decoding：model-free，和 symbolic proposer 更近
2. Token Recycling：train-free / draft-free，可作为近邻对比或 appendix
3. TETRIS / AdaServe：强近邻但依赖概率信号，优先做边界讨论；只有公平实现可行时才跑表
```

产出：

```text
[x] near-paper baseline feasibility matrix：
    /root/TriePilot/runs/20260515_2115_session75_near_paper_feasibility/near_paper_feasibility_matrix.csv
    /root/TriePilot/runs/20260515_2115_session75_near_paper_feasibility/near_paper_feasibility_matrix.json
[x] 小规模复现实验结果（如果公平可跑）：
    本轮没有方法满足同栈强公平条件，故未启动外部 baseline 在线小表
[x] related-work boundary notes：
    /root/TriePilot/runs/20260515_2115_session75_near_paper_feasibility/analysis_near_paper_feasibility.md
[x] external-reference / discussion 段落素材：
    /root/TriePilot/runs/20260515_2115_session75_near_paper_feasibility/integrity_near_paper_feasibility.json
    /root/TriePilot/runs/20260515_2115_session75_near_paper_feasibility/network_probe.txt
[x] 外部源码同步与小范围验证：
    /root/TriePilot/external/near_paper_sources_20260515/source_manifest.json
    /root/TriePilot/runs/20260515_2130_session75_external_code_smoke/source_manifest.json
    /root/TriePilot/runs/20260515_2130_session75_external_code_smoke/analysis_external_code_smoke.md
    /root/TriePilot/runs/20260515_2130_session75_external_code_smoke/integrity_external_code_smoke.json
    /root/TriePilot/runs/20260515_2130_session75_external_code_smoke/arctic_suffix_micro_result.json
    /root/TriePilot/runs/20260515_2130_session75_external_code_smoke/sam_static_micro_result.json
    /root/TriePilot/runs/20260515_2130_session75_external_code_smoke/token_recycling_micro_result.json
[x] baseline 专属 conda 环境：
    /root/TriePilot/baseline_envs/arctic_suffix
    /root/TriePilot/baseline_envs/sam_decoding
    /root/TriePilot/baseline_envs/token_recycling
    /root/TriePilot/baseline_envs/tetris
    /root/TriePilot/runs/20260515_2145_session75_baseline_conda_envs/status_after_micro_fixed.json
    /root/TriePilot/runs/20260515_2145_session75_baseline_conda_envs/tetris_env_import_after_build.log
[ ] TETRIS 独立环境下一步：
    /root/TriePilot/baseline_envs/tetris 中继续做最小 end-to-end / repo benchmark smoke，
    结果单独进入 separate-stack appendix 证据，不进入 SGLang 主公平表。
```

### Session 8：Strategy Bank 实验

任务：

```text
[x] cold vs warm / workload shift / regime reuse 第一轮 A100 smoke
[x] exploration floor / 最小验证预算 proxy 小型对照
[x] selective per-regime floor / cooldown recovery 第一轮小型机制对照（负面）
[x] capped-but-denser warm-start floor 小型 live sweep（负面）
[x] request-local early accept probe 机制设计与 A100 小型 live sweep
[x] request-local recovery probe/sustain 参数扫
[x] trace-driven positive-trigger replay
[x] 正式多 seed Strategy Bank recovery 图
```

产出：

```text
[x] strategy bank hit rate 第一轮 smoke：
    /root/TriePilot/runs/20260516_session8_strategy_bank_shift_smoke/phase_trace_summary.csv
    /root/TriePilot/runs/20260516_session8_strategy_bank_shift_smoke/regime_reuse_summary.csv
    /root/TriePilot/runs/20260516_session8_strategy_bank_shift_smoke/recovery_summary.json
    /root/TriePilot/runs/20260516_session8_strategy_bank_shift_smoke/integrity_session8_strategy_bank_shift_smoke.json
    /root/TriePilot/runs/20260516_session8_strategy_bank_shift_smoke/analysis_session8_strategy_bank_shift_smoke.md
[x] floor proxy 小型对照：
    /root/TriePilot/runs/20260516_session8_floor_proxy_diagnostic/floor_proxy_phase_summary.csv
    /root/TriePilot/runs/20260516_session8_floor_proxy_diagnostic/floor_proxy_comparison.csv
    /root/TriePilot/runs/20260516_session8_floor_proxy_diagnostic/integrity_session8_floor_proxy_diagnostic.json
    /root/TriePilot/runs/20260516_session8_floor_proxy_diagnostic/analysis_session8_floor_proxy_diagnostic.md
[x] selective recovery 第一轮机制对照（门过窄，probes=0）：
    /root/TriePilot/runs/20260516_session8_selective_recovery_seed20260516/phase_summary.csv
    /root/TriePilot/runs/20260516_session8_selective_recovery_seed20260516/recovery_comparison.csv
    /root/TriePilot/runs/20260516_session8_selective_recovery_seed20260516/integrity_session8_selective_recovery.json
    /root/TriePilot/runs/20260516_session8_selective_recovery_seed20260516/analysis_session8_selective_recovery.md
[x] selective recovery v2 小型复跑（probes 触发但收益不足）：
    /root/TriePilot/runs/20260516_session8_selective_recovery_v2_seed20260516/phase_summary.csv
    /root/TriePilot/runs/20260516_session8_selective_recovery_v2_seed20260516/recovery_comparison.csv
    /root/TriePilot/runs/20260516_session8_selective_recovery_v2_seed20260516/integrity_session8_selective_recovery.json
    /root/TriePilot/runs/20260516_session8_selective_recovery_v2_seed20260516/analysis_session8_selective_recovery.md
[x] capped-but-denser warm-start floor 小型复跑（更密 probes 仍收益不足）：
    /root/TriePilot/runs/20260516_session8_dense_recovery_floor_seed20260516/phase_summary.csv
    /root/TriePilot/runs/20260516_session8_dense_recovery_floor_seed20260516/recovery_comparison.csv
    /root/TriePilot/runs/20260516_session8_dense_recovery_floor_seed20260516/integrity_session8_selective_recovery.json
    /root/TriePilot/runs/20260516_session8_dense_recovery_floor_seed20260516/analysis_session8_selective_recovery.md
[x] request-local early accept probe 小型 live sweep：
    /root/TriePilot/runs/20260516_session8_request_local_recovery_seed20260516/phase_summary.csv
    /root/TriePilot/runs/20260516_session8_request_local_recovery_seed20260516/recovery_comparison.csv
    /root/TriePilot/runs/20260516_session8_request_local_recovery_seed20260516/integrity_session8_selective_recovery.json
    /root/TriePilot/runs/20260516_session8_request_local_recovery_seed20260516/analysis_session8_selective_recovery.md
[x] request-local recovery 参数扫：
    /root/TriePilot/runs/20260516_session8_request_local_param_sweep_seed20260516/param_sweep_phase_summary.csv
    /root/TriePilot/runs/20260516_session8_request_local_param_sweep_seed20260516/param_sweep_recovery_comparison.csv
    /root/TriePilot/runs/20260516_session8_request_local_param_sweep_seed20260516/param_sweep_config_summary.csv
    /root/TriePilot/runs/20260516_session8_request_local_param_sweep_seed20260516/integrity_session8_request_local_param_sweep.json
    /root/TriePilot/runs/20260516_session8_request_local_param_sweep_seed20260516/analysis_session8_request_local_param_sweep.md
[x] 正式多 seed Strategy Bank recovery / trace-driven positive-trigger replay：
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_probe2_seed20260518/phase_summary.csv
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_probe2_seed20260518/recovery_comparison.csv
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_probe2_seed20260518/integrity_session8_selective_recovery.json
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_probe2_seed20260519/phase_summary.csv
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_probe2_seed20260519/recovery_comparison.csv
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_probe2_seed20260519/integrity_session8_selective_recovery.json
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/combined_phase_summary.csv
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/probe2_recovery_comparison_multiseed.csv
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/probe2_seed_phase_delta.csv
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/probe2_phase_delta_aggregate.csv
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/positive_trigger_replay_summary.csv
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/recovery_time_trace_points.csv
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/positive_trigger_config_comparison_seed20260516.csv
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/integrity_session8_recovery_multiseed.json
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/analysis_session8_recovery_multiseed.md
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/plots/recovery_time_tpot.png
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/plots/recovery_tpot_delta.png
    /root/TriePilot/runs/20260516_session8_recovery_multiseed_summary/plots/positive_trigger_probe_sensitivity.png
```

### Session 9：消融与开销

任务：

```text
[x] Session 9 proxy ablation / controller overhead preflight：
    A100 online serving，Qwen3-8B，NUM_PROMPTS=64，out64，cgbs8/max_running=8，
    shape_bucket_mode=batch_max，seed=20260518；覆盖 instructcoder+gsm8k 与
    cnn_dailymail+random 两组 50/50 mixed workload、B_batch=100/160。
    对比 batch_global_budget2、batch_global_budget8、equal_budget_allocation、
    match_depth_greedy、accept_ema_greedy、triepilot_allocation。
[x] true w/o trie features hard-switch ablation hooks + A100 online smoke
[x] true w/o history hard-switch ablation hooks + A100 online smoke
[x] true w/o serving-pressure hard-switch ablation hooks + A100 online smoke
[x] true w/o Strategy Bank hard-switch ablation hooks + A100 online smoke
[x] formal hard-switch ablation table seed slice（seed=20260521）
[x] formal hard-switch ablation table seed slice（seed=20260520）
[x] formal hard-switch ablation table seed slice（seed=20260522）
[x] formal multi-seed ablation table after hard-switch seed slices + AR/no-spec are completed
```

产出：

```text
proxy ablation / overhead preflight：
    /root/TriePilot/runs/20260516_session9_ablation_overhead_preflight_seed20260518/session5_sweep_summary.csv
    /root/TriePilot/runs/20260516_session9_ablation_overhead_preflight_seed20260518/session9_ablation_proxy_table.csv
    /root/TriePilot/runs/20260516_session9_ablation_overhead_preflight_seed20260518/session9_controller_overhead_table.csv
    /root/TriePilot/runs/20260516_session9_ablation_overhead_preflight_seed20260518/integrity_session9_ablation_overhead_preflight.json
    /root/TriePilot/runs/20260516_session9_ablation_overhead_preflight_seed20260518/analysis_session9_ablation_overhead_preflight.md

完整性检查：
    summary_rows=24 / expected_rows=24，missing=[]，extra=[]，trace_errors=[]，
    budget_violations=[]，server_log_hits=[]；实验后 GPU 回到 1 MiB，无残留
    sglang.launch_server / bench_serving / Session 5 runner 进程。

第一轮结论：
    这轮是 proxy ablation，不是最终组件 hard-switch 消融。match_depth_greedy 可作为
    trie-structure-only proxy（w/o history / w/o Strategy Bank），accept_ema_greedy 可作为
    history-only proxy（w/o trie structural features / w/o Strategy Bank），equal_budget_allocation
    可作为 w/o utility estimator 的负面对照。TriePilot controller overhead 在 4 个
    scenario/B 设置中 p99 均 < 120us，平均 p50/p99 约 76.3us / 108.4us。
    InstructCoder+GSM8K 上 TriePilot verified_nodes=24，p99 TPOT 17.95/17.99ms，
    优于单信号 request-level baselines 与 batch_global_budget2，但 mean TPOT 仍不如
    batch_global_budget8。CNN/DailyMail+random 上 TriePilot verified_nodes=24，
    p99 TPOT 34.48/33.35ms，mean TPOT 22.98/22.61ms，均为本轮最优，并显著少于
    batch_global_budget2 的 6312 verified nodes。

下一步：
    如果论文需要组件因果 claim，先实现 true hard-switch ablations（w/o trie features、
    w/o history、w/o serving-pressure、w/o Strategy Bank），再扩展到选定 paper scenarios
    与多 seed；不要把本轮 proxy table 直接写成最终组件消融。TETRIS separate-stack
    appendix smoke 仍可补，但不进入 SGLang 主公平表。

hard-switch ablation hook / smoke（2026-05-16）：
    已新增 TRIEPILOT_ABLATION_MODE hard-switch：no_trie_features、no_history、
    no_serving_pressure、no_strategy_bank。runner 支持方法名 triepilot_wo_trie_features、
    triepilot_wo_history、triepilot_wo_serving_pressure、triepilot_wo_strategy_bank，
    并在 raw_step_events 与 summary 中记录 ablation_modes。A100 RED/GREEN 验证：
    先同步测试到远端，tests/test_sglang_triepilot_telemetry.py 出现预期 4 failed /
    21 passed；tests/test_launch_scripts.py 出现预期 1 failed / 4 passed /
    2 subtests passed。实现后 A100 targeted pytest：
    SGLANG_SOURCE_ROOT=/root/TriePilot/third_party/sglang_flex
    .venv/bin/python -m pytest tests/test_sglang_triepilot_telemetry.py
    tests/test_launch_scripts.py -q 通过 30 passed / 2 subtests passed；
    bash -n 覆盖 scripts/remote/a100_session5_allocator_baselines.sh；
    /root/anaconda3/envs/sglang/bin/python -m py_compile 覆盖
    triepilot/sglang_integration/budget.py、recorder.py、third_party/sglang_flex
    的 ngram_worker.py 与 server_args.py。
    A100 online smoke 使用 Qwen3-8B，NUM_PROMPTS=32，out64，cgbs8/max_running=8，
    shape_bucket_mode=batch_max，seed=20260520，覆盖 instructcoder+gsm8k 与
    cnn_dailymail+random 两组 50/50 mixed workload、B_batch=100；方法为
    triepilot_allocation + 四个 true hard-switch ablation。结果保存在：
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_smoke_seed20260520/session5_sweep_summary.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_smoke_seed20260520/session9_hardswitch_ablation_smoke_table.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_smoke_seed20260520/integrity_session9_hardswitch_ablation_smoke.json
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_smoke_seed20260520/analysis_session9_hardswitch_ablation_smoke.md
    完整性：summary_rows=10/10，missing=[]，extra_methods=[]，mode_mismatch_count=0，
    budget_violation_count=0，trace_errors=[]，server_log_hits=[]，
    min_cuda_graph_ratio=1.0，min_cuda_graph_token_shape_ok_ratio=1.0，
    no_strategy_bank_hit_rates=[0.0,0.0]，实验后 GPU 回到 1 MiB。
    smoke 结论：hard-switch 已经进入真实 A100 online serving path，不再只是 proxy baseline。
    instructcoder+gsm8k 上 TriePilot verified_nodes=20、mean/p99 TPOT=16.97/17.72ms；
    w/o trie features verified_nodes=0；w/o history 与 w/o Strategy Bank 均 verified_nodes=1256，
    mean/p99 TPOT 分别约 17.90/19.10ms 与 17.96/19.16ms。cnn_dailymail+random 上
    TriePilot verified_nodes=22、mean/p99 TPOT=23.24/33.38ms；w/o trie features verified_nodes=0；
    w/o history 与 w/o Strategy Bank 均 verified_nodes=1214，mean/p99 TPOT 分别约
    26.98/34.72ms 与 26.97/34.67ms。该轮证明组件开关和 trace 口径可用，但样本仍是 smoke，
    不能写成最终多 seed 组件消融结论。

formal hard-switch ablation seed slice（2026-05-16，seed=20260521）：
    A100 online serving，Qwen3-8B，NUM_PROMPTS=64，out64，cgbs8/max_running=8，
    shape_bucket_mode=batch_max，ratio=50/50，B_batch=100/160。覆盖
    instructcoder+gsm8k、cnn_dailymail+random、json_tool+sharegpt 三组代表场景；
    方法为 batch_global_budget2、batch_global_budget8、triepilot_allocation 与
    四个 true hard-switch ablation（triepilot_wo_trie_features、triepilot_wo_history、
    triepilot_wo_serving_pressure、triepilot_wo_strategy_bank）。结果保存在：
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260521/session5_sweep_summary.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260521/session9_hardswitch_formal_seed20260521_table.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260521/session9_hardswitch_formal_seed20260521_winners.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260521/integrity_session9_hardswitch_formal_seed20260521.json
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260521/analysis_session9_hardswitch_formal_seed20260521.md
    完整性：summary_rows=42/42，missing=[]，extra=[]，duplicates=[]，trace_errors=0，
    missing_trace=0，empty_trace=0，mode_mismatch_count=0，budget_violation_count=0，
    server_log_hit_count=0，min_cuda_graph_ratio=1.0，
    min_cuda_graph_token_shape_ok_ratio=1.0；no_strategy_bank_hit_rates 全为 0.0；
    实验后 GPU 回到 1 MiB，残留进程 residual_processes=[]。
    seed slice 结果：6 个 scenario/B cell 中，p99 winner 分别为
    triepilot_wo_serving_pressure 2 个、triepilot_wo_trie_features 2 个、
    triepilot_allocation 2 个；mean winner 为 triepilot_wo_serving_pressure 2 个、
    batch_global_budget8 4 个。TriePilot 相对 best static 的 p99 TPOT 平均 delta=-1.554ms
    （min=-2.518，max=-1.061），mean TPOT 平均 delta=+0.836ms
    （min=-0.067，max=+1.708）。w/o history 与 w/o Strategy Bank 在 6 个 cell 中稳定把
    verified_nodes 拉高到约 2.3K-2.7K，而主 TriePilot 仅 18-26 个 verified_nodes，
    说明 history / Strategy Bank 复用确实在抑制低价值投机；但 w/o trie features 与
    w/o serving-pressure 当前表现为极保守诊断开关，经常验证 0 或极少 draft nodes，
    可在部分低价值 regime 中匹配或优于 tail latency，因此最终论文不能简单写成
    “去掉该组件一定变差”，应解释为 hard-switch diagnostic。
    该轮是 formal seed slice，不是最终 multi-seed 表；仍缺 AR/no-spec 行和至少另两个 seed。

formal hard-switch ablation seed slice（2026-05-16，seed=20260520）：
    A100 online serving，Qwen3-8B，NUM_PROMPTS=64，out64，cgbs8/max_running=8，
    shape_bucket_mode=batch_max，ratio=50/50，B_batch=100/160。覆盖
    instructcoder+gsm8k、cnn_dailymail+random、json_tool+sharegpt 三组代表场景；
    方法为 batch_global_budget2、batch_global_budget8、triepilot_allocation 与
    四个 true hard-switch ablation（triepilot_wo_trie_features、triepilot_wo_history、
    triepilot_wo_serving_pressure、triepilot_wo_strategy_bank）。结果保存在：
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260520/session5_sweep_summary.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260520/session9_hardswitch_formal_seed20260520_table.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260520/session9_hardswitch_formal_seed20260520_winners.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260520/integrity_session9_hardswitch_formal_seed20260520.json
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260520/analysis_session9_hardswitch_formal_seed20260520.md
    完整性：summary_rows=42/42，missing=[]，extra=[]，duplicates=[]，trace_errors=0，
    missing_trace=0，empty_trace=0，mode_mismatch_count=0，budget_violation_count=0，
    server_log_hit_count=0，min_cuda_graph_ratio=1.0，
    min_cuda_graph_token_shape_ok_ratio=1.0；no_strategy_bank_hit_rates 全为 0.0；
    实验后 GPU 回到 1 MiB，残留进程 residual_processes=[]。
    seed slice 结果：6 个 scenario/B cell 中，p99 winner 为 triepilot_allocation 3 个、
    triepilot_wo_serving_pressure 3 个；mean winner 为 batch_global_budget8 4 个、
    triepilot_wo_serving_pressure 2 个。TriePilot 相对 best static 的 p99 TPOT 平均
    delta=-3.316ms（min=-5.036，max=-1.282），mean TPOT 平均 delta=+0.864ms
    （min=-1.161，max=+1.942）。w/o history 与 w/o Strategy Bank 在 6 个 cell 中
    稳定把 verified_nodes 拉高到约 2.2K-2.7K，而主 TriePilot 仅 26-34 个
    verified_nodes；这继续支持 history / Strategy Bank 复用主要负责抑制低价值投机。
    w/o trie features 与 w/o serving-pressure 仍表现为保守诊断开关，部分 cell 可赢 p99，
    因此组件消融表应解释为 hard-switch diagnostic，而不是“去掉组件必然变差”的单调因果。

formal hard-switch ablation seed slice（2026-05-16，seed=20260522）：
    A100 online serving，Qwen3-8B，NUM_PROMPTS=64，out64，cgbs8/max_running=8，
    shape_bucket_mode=batch_max，ratio=50/50，B_batch=100/160。覆盖
    instructcoder+gsm8k、cnn_dailymail+random、json_tool+sharegpt 三组代表场景；
    方法为 batch_global_budget2、batch_global_budget8、triepilot_allocation 与
    四个 true hard-switch ablation。结果保存在：
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260522/session5_sweep_summary.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260522/session9_hardswitch_formal_seed20260522_table.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260522/session9_hardswitch_formal_seed20260522_winners.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260522/integrity_session9_hardswitch_formal_seed20260522.json
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_formal_seed20260522/analysis_session9_hardswitch_formal_seed20260522.md
    完整性：summary_rows=42/42，missing=[]，extra=[]，duplicates=[]，
    missing_trace_count=0，empty_trace_count=0，trace_errors_count=0，
    trace_event_count_mismatch_count=0，mode_mismatch_count=0，
    budget_violation_count=0，server_log_hit_count=0，min_cuda_graph_ratio=1.0，
    min_cuda_graph_token_shape_ok_ratio=1.0；no_strategy_bank_hit_rates 全为 0.0；
    实验后 GPU 回到 1 MiB，残留进程 residual_processes=[]。
    seed slice 结果：6 个 scenario/B cell 中，NGRAM-only p99 winner 为
    triepilot_allocation 2 个、triepilot_wo_trie_features 3 个、
    triepilot_wo_serving_pressure 1 个；mean winner 为 triepilot_allocation 2 个、
    batch_global_budget8 4 个。TriePilot 相对 best static 的 p99 TPOT 平均
    delta=-1.681ms（min=-3.819，max=-0.473），mean TPOT 平均 delta=+0.721ms
    （min=-2.217，max=+2.779）。w/o history 与 w/o Strategy Bank 在 6 个 cell 中
    仍把 verified_nodes 拉高到约 2.3K-2.7K，而主 TriePilot 仅 18-30 个
    verified_nodes；该 seed 继续支持 history / Strategy Bank 复用负责抑制低价值投机。

AR/no-spec formal 对齐行（2026-05-16，seed=20260520/20260521/20260522）：
    A100 online serving，Qwen3-8B，NUM_PROMPTS=64，out64，max_concurrency=8，
    request_rate=8，ratio=50/50。每个 seed 使用一个 no-spec server 跑三组代表场景，
    AR 行不进入 NGRAM ablation_modes 检查。结果保存在：
    /root/TriePilot/runs/20260516_session9_ar_no_spec_formal_seed20260520/ar_no_spec_summary.csv
    /root/TriePilot/runs/20260516_session9_ar_no_spec_formal_seed20260521/ar_no_spec_summary.csv
    /root/TriePilot/runs/20260516_session9_ar_no_spec_formal_seed20260522/ar_no_spec_summary.csv
    /root/TriePilot/runs/20260516_session9_ar_no_spec_formal_multiseed_summary/session9_ar_no_spec_formal_multiseed.csv
    /root/TriePilot/runs/20260516_session9_ar_no_spec_formal_multiseed_summary/integrity_session9_ar_no_spec_formal_multiseed.json
    /root/TriePilot/runs/20260516_session9_ar_no_spec_formal_multiseed_summary/analysis_session9_ar_no_spec_formal_multiseed.md
    完整性：rows=9/9，missing=[]，extra=[]，duplicates=[]，completed_values=[64]，
    server_log_hit_count=0，GPU 回到 1 MiB，残留进程 residual_processes=[]。
    AR/no-spec 三 seed 平均 mean/p99 TPOT 为 18.652ms / 22.435ms，是 Session 9
    正式消融表中必须单独报告的强非投机参照。

formal hard-switch ablation multi-seed summary（2026-05-16）：
    已合并 seed=20260520/20260521/20260522 的 hard-switch 表，并将 AR/no-spec
    按 scenario/seed/B_batch 对齐复制为参照行。结果保存在：
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_multiseed_summary/session9_hardswitch_formal_multiseed_table.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_multiseed_summary/session9_hardswitch_formal_multiseed_winners.csv
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_multiseed_summary/integrity_session9_hardswitch_formal_multiseed.json
    /root/TriePilot/runs/20260516_session9_hardswitch_ablation_multiseed_summary/analysis_session9_hardswitch_formal_multiseed.md
    完整性：combined_rows=144，其中 hard_rows=126、AR-aligned rows=18；
    missing=[]，extra=[]，duplicates=[]，per_seed_hard_integrity_ok=[true,true,true]，
    all_seed_hard_integrity_ok=true；GPU 回到 1 MiB，残留进程 residual_processes=[]。
    合表结论：加入 AR 后，p99 winner 为 AR/no-spec 18/18；mean winner 为
    AR/no-spec 8/18、batch_global_budget8 6/18、triepilot_wo_serving_pressure 2/18、
    triepilot_allocation 2/18。NGRAM-only p99 winner 为 triepilot_allocation 7/18、
    triepilot_wo_serving_pressure 6/18、triepilot_wo_trie_features 5/18；NGRAM-only
    mean winner 为 batch_global_budget8 12/18、triepilot_wo_serving_pressure 4/18、
    triepilot_allocation 2/18。TriePilot 相对 best static 的平均 p99/mean delta 为
    -2.184ms / +0.807ms；相对 AR 的平均 p99/mean delta 为 +3.259ms / +1.200ms。
    verified_nodes 均值为 TriePilot 26.1、w/o history 2449.8、w/o Strategy Bank 2449.8，
    因此正式写法应是：history / Strategy Bank 复用显著抑制低价值投机；w/o trie features
    与 w/o serving-pressure 是偏保守 hard-switch diagnostic，不能写成“去掉组件必然变差”。

下一步：
    1. 基于 Session 9 multi-seed hard-switch summary 写 paper-ready ablation 表述：
       区分 NGRAM-only component diagnostic 与 AR/no-spec 强参照，不把 proxy table 或
       保守 hard-switch 写成单调因果结论。
    2. 正式表必须报告 mean/p99 TPOT、verified_nodes、APV、strategy_bank_hit_rate、
       controller_overhead_p99_us、ablation_modes 完整性、CUDA graph/token-shape ratio、
       budget violation 检查与残留进程/GPU 清理状态。
    3. 可选补 controller overhead 的 paper-ready 小表：从 multi-seed table 中拆出
       triepilot family p99 controller_time_us，配合已有 preflight overhead table 写正文或附录。
    4. TETRIS separate-stack appendix smoke 仍可补，但继续标注为 separate-stack boundary，
       不进入 SGLang 主公平表。
    5. Session 10 可迁移性实验可作为下一阶段：做 vLLM 或 TensorRT-LLM NGram trace-driven replay，
       验证 allocator 抽象不是 SGLang-only。
```

Session 10 vLLM NGram portability replay（2026-05-16）：
    A100 远端只读/trace-driven 验证已完成，未在本地跑实验。使用 /root/anaconda3/envs/vllm
    中的 vLLM 0.16.0，直接导入
    vllm.v1.spec_decode.ngram_proposer._find_longest_matched_ngram_and_propose_tokens，
    对 Qwen3-8B tokenizer 下的三组代表性 50/50 mixed workload 做 prompt-token trace replay：
    instructcoder+gsm8k、cnn_dailymail+random、json_tool+sharegpt；seed=20260520/20260521/20260522；
    B_batch=100/160；方法为 batch_global_budget2、batch_global_budget8、equal_budget_allocation、
    match_depth_greedy、accept_ema_greedy、triepilot_allocation。每个 workload 使用 64 prompts，
    每个请求取 0.50/0.65/0.80/0.90 四个 prefix cut，用后续 prompt token 只作为 proxy acceptance，
    因此该实验只能作为 portability appendix / abstraction-boundary 证据，不能写成在线 serving throughput。
    结果保存在：
    /root/TriePilot/runs/20260516_session10_vllm_ngram_portability_replay_seed20260520_22/session10_vllm_ngram_portability_summary.csv
    /root/TriePilot/runs/20260516_session10_vllm_ngram_portability_replay_seed20260520_22/session10_vllm_ngram_state_summary.csv
    /root/TriePilot/runs/20260516_session10_vllm_ngram_portability_replay_seed20260520_22/vllm_ngram_replay_events.jsonl
    /root/TriePilot/runs/20260516_session10_vllm_ngram_portability_replay_seed20260520_22/integrity_session10_vllm_ngram_portability.json
    /root/TriePilot/runs/20260516_session10_vllm_ngram_portability_replay_seed20260520_22/analysis_session10_vllm_ngram_portability.md
    完整性：summary_rows=108/108，state_rows=9，raw_event_count=3456，missing=[]，
    budget_violation_count=0，vLLM NGram import 正常，tokenizer=Qwen2TokenizerFast；实验前后 GPU
    均为 1 MiB，posthoc_residual_processes=[]。由于脚本内置 residual 检查发生在 Python 进程退出前，
    integrity 中 residual_processes 原字段包含脚本自身；以 posthoc_residual_processes=[] 为清理结论。
    汇总结果：TriePilot 在 vLLM linear NGram proposer 上平均 actual_draft_tokens=10.7，
    proxy_apv=0.3201，controller_overhead_p99_us 均值约 130.18us，strategy_bank_hit_rate 均值 0.9883；
    batch_global_budget8 平均 actual_draft_tokens=922.1、proxy_apv=0.1503；
    batch_global_budget2 平均 actual_draft_tokens=231.9、proxy_apv=0.3367。TriePilot 相比
    batch_global_budget8 的平均 actual draft delta=-911.44、proxy APV delta=+0.1697。
    结论：同一套 probability-free structural features + Strategy Bank allocator 可以消费 vLLM NGram
    的符号 proposal 并满足 B_batch 约束，支持“方法不是 SGLang-only”的附录论据；但 vLLM NGram
    当前是 linear prompt-lookup proposer，不是 SGLang tree verification path，不能和 SGLang
    在线主表混写为公平吞吐对比。
    下一步：
    1. 把 Session 10 整理成 portability appendix 小表，明确写 trace-driven proxy acceptance 与
       separate-stack / non-throughput boundary。
    2. 如果还需要更强可迁移性证据，再补 true vLLM online NGram smoke 或 TensorRT-LLM NGram replay；
       仍只作为 separate-stack appendix，不进入 SGLang 主公平表。
    3. 主线进入 Session 11 论文初稿：写 Introduction / Problem / Method / Evaluation skeleton，
       并把 Session 7 主表、Session 8 recovery、Session 9 hard-switch diagnostic 与 Session 10
       portability 边界统一到同一套 claim 口径。

Session 10.5：Related Baseline Bridge + TriePilot targeted optimization（下一步执行）：
    背景判断：
    当前不建议增加到 2-4 张 A100 来改变主实验口径。Qwen3-8B 在 1xA100 上已能稳定运行，
    多卡 tensor parallel 可能引入 NCCL/all-reduce 通信和 latency 噪声，不能解决 symbolic proposer
    接受率不稳定与 TriePilot under-use B_batch 的根因。若未来临时有多卡，只用于并行跑实验或更大模型附录，
    主公平表仍保持 1xA100。
    本阶段不要直接进入最终写稿，也不要无限调参；目标是在进入 Session 11 前补齐评审会关心的
    model-free / draft-free 论文基线边界，并给 TriePilot 一次小范围、机制明确的性能修正机会。

    任务 A：Related baseline bridge（优先执行）
    1. 复查 A100 上已有外部源码和隔离环境：
       /root/TriePilot/external/near_paper_sources_20260515，
       /root/TriePilot/baseline_envs/{arctic_suffix,sam_decoding,token_recycling,tetris}。
    2. 优先尝试 SuffixDecoding、SAM-Decoding、Token Recycling 的 separate-stack smoke / proxy table；
       能 online 就跑极小 online smoke，不能 online 就做 proposer-level / trace-driven proxy，并记录边界。
    3. TETRIS 在独立 env 中继续 synthetic proposal-selection micro 与最小 benchmark/DSD smoke；
       若旧 vLLM/FastChat/数据集入口阻塞，则记录为 probability-signal / separate-stack boundary。
    4. 每个相关论文基线都必须记录：
       是否同 A100、是否同模型、是否同 tokenizer、是否同 serving stack、是否 online serving、
       是否支持 mixed batch、是否支持 request-level budget allocation、是否使用概率信号、
       是否依赖外部 corpus/global suffix structure、能否进入主公平表。
    5. 产出一个 Related Baseline Bridge matrix：
       fair_same_stack、separate_stack_online、trace_or_proxy_only、main_table_allowed、appendix_allowed、
       blocked_reason、paper_claim_boundary。

    任务 B：TriePilot positive-evidence budget upgrade（窄优化）
    1. 只做一个小优化，不做 batch-level static fallback、不继续 score/confidence-only cap4 gate、
       不继续调 recovery cooldown / budget。
    2. 设计 positive-evidence upgrade：只有 request/regime 最近出现 accepted tokens、match_depth 稳定、
       proposal coverage 非零、negative_gain_count 低、Strategy Bank confidence 足够时，才谨慎从
       off/tiny/small 升级预算。
    3. 成功标准：
       mean TPOT 相比 current TriePilot 改善；p99 TPOT 不明显退化；verified_nodes 仍显著低于
       batch_global_budget8；APV 不崩；budget violation=0。
    4. 若失败，不继续调参；将 TriePilot 定位为 tail/waste-oriented allocator，进入写稿。

    任务 C：Favorable-regime / diagnostic benchmark
    1. 可以构造对 TriePilot 有利的 stress/diagnostic 场景，但不能伪装成主结果。
    2. 候选场景：代码编辑长片段复用、JSON/tool-call schema 重复、RAG/long-context 复用、
       shared-prefix/repeated-output、workload shift 中高收益 regime 消失后再次出现。
    3. 主表仍保留普通 mixed workload 与 AR/no-spec 强参照；favorable-regime 只用于机制展示和附录。

    下一步立即执行顺序：
    1. A100 上复查 related baseline env/source 状态，生成 Session 10.5 bridge preflight 结果。
    2. 根据 preflight 选择 1-2 个最可能跑通的 model-free 基线做 separate-stack smoke/proxy 小表。
    3. 再进入 TriePilot positive-evidence upgrade 的小型实现与 A100 live sweep。
    4. 完成后更新 AGENTS.md，并决定是否正式进入 Session 11 论文初稿。

Session 10.5 related baseline bridge preflight（2026-05-17）：
    A100 远端测试验证已完成，未在本地跑实验。使用已有外部源码：
    /root/TriePilot/external/near_paper_sources_20260515，以及隔离环境：
    /root/TriePilot/baseline_envs/{arctic_suffix,sam_decoding,token_recycling,tetris}。
    本轮产物保存在：
    /root/TriePilot/runs/20260517_session105_related_baseline_bridge_preflight/preflight_session105_bridge.json
    /root/TriePilot/runs/20260517_session105_related_baseline_bridge_preflight/related_baseline_bridge_matrix.csv
    /root/TriePilot/runs/20260517_session105_related_baseline_bridge_preflight/analysis_session105_related_baseline_bridge.md
    以及 suffix_smoke_result.json、sam_smoke_result.json、token_recycling_proxy_result.json、
    tetris_selection_smoke_result.json。
    完整性：env_import_ok=4/4，smoke_or_proxy_ok=4/4，main_table_allowed=0/4，
    appendix_allowed=4/4；GPU before/after 均为 1 MiB / 40960 MiB / 0%，posthoc
    residual_processes=[]。源码 commit：SuffixDecoding/Arctic fba641f8ffba，
    SAM-Decoding aaf939819223，Token Recycling 1b4c05cc642d，TETRIS acb77de80152。
    验证结果：SuffixDecoding proposer-level suffix-cache smoke 通过，match_len=3，
    accepted_prefix=3，latency≈14.50us；SAM StaticSAM proposer-level smoke 通过，
    states=20，match_len=3，accepted_prefix=3，latency≈115.25us；Token Recycling
    fixed tree-template proxy/import 通过，template version 2.2.2，nodes=80，max_depth=6；
    TETRIS synthetic CPU proposal-selection smoke 通过，capacity=4 后 proposal_lens=[2,2,0]，
    并明确依赖 draft proposal logprobs/probability scores。
    结论：四个近邻 baseline 均可作为 related-work / appendix boundary 证据，但都不满足
    同 A100 + 同 Qwen3-8B + 同 tokenizer + 同 SGLang serving stack + online mixed-batch
    request-level draft-node allocation 的主公平表条件，因此不能进入 SGLang 主吞吐表。
    下一步：
    1. 进入 TriePilot positive-evidence budget upgrade 的小型实现与 A100 live sweep；
       若 mean TPOT 未改善或 p99/APV/verified_nodes 退化，则不继续调参。
    2. 将本轮 related baseline bridge matrix 整理成论文 related work / appendix 小表，
       明确 separate-stack、proxy-only 与 probability-signal 边界。
    3. 若 positive-evidence upgrade 失败或收益不足，正式进入 Session 11 论文初稿，
       将 TriePilot 定位为 tail / verifier-waste-oriented allocator。

Session 10.5 positive-evidence budget upgrade live sweep（2026-05-17）：
    A100 远端测试验证已完成，未在本地跑实验。本轮在远端临时使用
    triepilot_positive_evidence_upgrade 方法别名，复用 Session 8 筛出的 request-local
    probe2 / accept_threshold=0.50 / sustain_budget=2 配置，并用 posthoc analysis 生成
    comparison / integrity / analysis。由于该优化未达标，稳定主线不发布该方法别名，
    默认主路径仍是 triepilot_allocation。A100 上 targeted validation 通过：
    SGLANG_SOURCE_ROOT=/root/TriePilot/third_party/sglang_flex .venv/bin/python -m pytest
    tests/test_launch_scripts.py tests/test_sglang_triepilot_telemetry.py -q 通过
    30 passed / 2 subtests passed；/root/anaconda3/envs/sglang/bin/python -m py_compile
    覆盖 triepilot/sglang_integration/budget.py、recorder.py、third_party/sglang_flex 的
    ngram_worker.py、server_args.py；bash -n 覆盖 a100_session5_allocator_baselines.sh。
    正式小型 live sweep 使用 Qwen3-8B，seed=20260523，NUM_PROMPTS=64、out64、
    max_concurrency=8、request_rate=8、CUDA_GRAPH_MAX_BS=8、MAX_RUNNING_REQUESTS=8、
    shape_bucket_mode=batch_max，覆盖三组代表性 50/50 mixed workload：
    instructcoder+gsm8k、cnn_dailymail+random、json_tool+sharegpt；B_batch=100/160；
    方法为 batch_global_budget2、batch_global_budget8、triepilot_allocation 与
    triepilot_positive_evidence_upgrade。结果保存在：
    /root/TriePilot/runs/20260517_session105_positive_evidence_upgrade_seed20260523/session5_sweep_summary.csv
    /root/TriePilot/runs/20260517_session105_positive_evidence_upgrade_seed20260523/positive_evidence_comparison.csv
    /root/TriePilot/runs/20260517_session105_positive_evidence_upgrade_seed20260523/integrity_session105_positive_evidence.json
    /root/TriePilot/runs/20260517_session105_positive_evidence_upgrade_seed20260523/analysis_session105_positive_evidence.md。
    完整性：summary_rows=24/24，missing=[]，extra=[]，duplicates=[]，trace_event_total=11498，
    missing_trace_count=0，empty_trace_count=0，budget_violation_count=0，
    server_log_hit_count=0，min_cuda_graph_ratio=1.0，
    min_cuda_graph_token_shape_ok_ratio=1.0；实验后 GPU 回到
    1 MiB / 40960 MiB / 0%，residual_processes=[]。
    结果结论：positive-evidence upgrade 在 6 个 scenario/B cell 中 mean TPOT 胜出 current
    TriePilot 为 0/6，p99 TPOT 胜出为 0/6；平均 mean TPOT delta=+2.223ms，
    平均 p99 TPOT delta=+1.288ms。request-local probe 确实触发，平均
    request_local_probe_count=139.17，accepted tokens 与 APV 在 json_tool+sharegpt 等场景中
    明显上升（例如 B=100 时 accepted 2→281、APV 0.0909→0.2739），但 verified_nodes
    平均变为 current TriePilot 的 23.61x；即便如此，positive verified nodes 仍只有
    batch_global_budget8 的约 2.91%，说明该方向的问题不是 budget explosion，而是额外 probe
    带来的 runtime/latency tradeoff 不达标。根据预设成功标准，该优化失败；不继续调
    request-local positive-evidence probe / sustain / threshold。
    下一步：
    1. 正式进入 Session 11 论文初稿：写 Introduction / Problem / Method / Evaluation skeleton。
    2. 将 TriePilot 主张收敛为 NGRAM/allocator online baselines 下的 tail latency 与
       verifier-waste-oriented allocator；端到端 TPOT 必须保留 AR/no-spec 强基线边界。
    3. 把 Session 10.5 related baseline bridge matrix 和 positive-evidence 负面验证写入
       related work / limitations / appendix，不把 request-local probe 作为主方法继续调参。
    4. 可选只做 favorable-regime diagnostic appendix，但不能伪装成主结果；默认不再开新的
       allocator 调参 session。

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
