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

任务：

```text
固定 SGLang 0.5.6 commit
跑通 AR
跑通 static NGRAM default
准备 Qwen/Llama 8B 模型
搭建日志目录结构
```

产出：

```text
env.json
baseline AR result
static NGRAM result
```

### Session 2：数据集与 workload

任务：

```text
下载并清洗 InstructCoder / ShareGPT / GSM8K / CNN/DailyMail / random
构造 mixed workload JSONL
保存 sample ids
```

产出：

```text
data/*.jsonl
workload configs
```

### Session 3：NGRAM 特征与 telemetry

任务：

```text
插桩 match_depth / candidate_count / branch_entropy
记录 accept_len / verified_nodes / latency
```

产出：

```text
raw_step_events.jsonl
feature sanity report
```

### Session 4：per-request budget microbenchmark

任务：

```text
验证不同 request draft nodes 是否减少实际 verify cost
确认 padding / shape 行为
```

产出：

```text
microbenchmark table
是否继续 V2/V3 的决策
```

### Session 5：实现简单 allocator baselines

任务：

```text
equal allocation
random allocation
match-depth greedy
accept-EMA greedy
batch-global best tier
```

产出：

```text
baseline allocation results
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
