# 2026-08-11 A100 HotpotQA 拓扑与调度优化

## 1. 目标与有效性边界

本轮在 6x A100-SXM4-80GB 单机上调试 HotpotQA DAG-aware replay。目标不是单纯提高
planner coverage，而是在 prefill/decode 均存在实际压力时比较 baseline 与 HexAgenT 的
workflow tail SLO ratio。

有效实验必须同时满足：

- 使用 20 个不同 HotpotQA workflows、636 calls，不复制 workflow；
- 固定复用 profile 中的 input/output token length；
- SLO 分母为相同 Llama-3.1-8B、A100、TP1、禁用 radix cache 的独占 profile DAG
  critical path；
- baseline 与 HexAgenT 各自在测量前执行 32 requests、并发度 8 的独立 warmup；
- HexAgenT 运行存在非零 schedule revision coverage；
- 所有 636 calls 成功完成。

没有逐模式 warmup、sidecar 重启后 router revision watcher 已退出、或只完成单个 mode 的
结果，只能作为 debug evidence，不能用于性能结论。

## 2. 资产与环境

- 模型：Llama-3.1-8B-Instruct BF16，TP1 worker。
- 拓扑：1 prefill worker + 5 decode workers。
- Prefill：`max_running_requests=32`、`prefill_max_requests=32`。
- Decode：每个 worker `max_running_requests=128`。
- KV：Mooncake intra-node NVLink；planner transfer model 为 61 GB/s + 3ms startup。
- 输入：20 workflows / 636 calls；独占 profile 与 replay 长度一一对齐。
- `0.18 workflow/s` 下，prefill GPU utilization P95 为 100%；五个 decode worker 的
  utilization P95 也均为 100%。DAG barrier 会造成短时 idle，因此均值低于峰值。

旧 Pod 的有效 profile、代码和结果已归档为
`real_trace_profiles/a100_hexgent_final_valid_assets_20260810.tar.gz`。远端与本地 SHA-256
均为 `9732893bbf171a12e130941f02cb866b92e76ab5c9d5ab06815c5f0858206a4b`。

## 3. 直接观测到的问题

### 3.1 运行顺序与冷启动偏差

旧 runner 只在两种 mode 之前做一次很小的 readiness warmup。第二个运行的 mode 因模型、
decode kernel 和 Mooncake 路径已经热起来，P95 常出现约 5% 的系统性优势，正反顺序会得到
相反结论。

runner 已改为每个 mode 测量前分别执行 32 个 warmup requests，并使用并发度 8。该 warmup
不计入 replay summary。

### 3.2 Planner 求解并不是主要 CPU 瓶颈

逐模式 warmup 的有效运行中，planner invocation P95 约为 1.3--1.5ms。被 planner 覆盖的
calls 会在 global admission queue 中等待 prefill credit，admission wait 可达到数百毫秒；
该时间不是 solver CPU time。Baseline 将相同排队放在 worker-local queue，HexAgenT 则保留
一部分请求在 global queue，允许后续 plan 修改顺序和 decode placement。

### 3.3 Fan-out workflow 会错误回退到短串行排序

代码审计确认：scheduler 虽然会识别 sibling fan-out，但 fan-out 后的短 call 仍可能重新进入
`small_sequential_batch`，使用 call laxity/reveal order，而不是 projected SLO ratio。这样会让
HotpotQA workflow 在 root fan-out 后丢失统一的风险优先语义。

## 4. 本轮代码改动

### 4.1 持久化 fan-out 语义

`AgenticPDExactScheduler` 维护已经 reveal fan-out 的 workflow ID。只要一个 workflow 曾出现
fan-out，其后续 prefill/decode calls：

- 不再执行低压力 pairwise local search；
- 不再回退到短串行 laxity/reveal-order 分支；
- 继续使用 high-first projected SLO ratio 排序。

纯 ShareGPT/BFCL 串行 workflow 的原有短串行行为不变。

### 4.2 实验 runner 的逐模式 warmup

`run_h100_formal_comparison.sh` 新增：

- `WARMUP_REQUESTS`，默认 32；
- `WARMUP_CONCURRENCY`，默认 8；
- 每个 trace/mode 测量前独立执行 warmup。

### 4.3 Replay runtime ID 隔离

多轮实验复用同一批 profile workflow IDs，而 planner 的 runtime state 原先不会在同 ID 的
新 arrival 上自动重置。这会使后续轮次继承旧 revealed/completed DAG。Replayer 现在接受
`--run-namespace`：发送给 router/planner 的 workflow、call 和 parent IDs 带有唯一 run
namespace，结果 JSONL 仍保存原始 IDs，因此 profile lookup 与 SLO 汇总不变。

## 5. 测试

- 本地 scheduler/planner/trace/collector：143 passed。
- A100 Pod 同一组测试：143 passed。
- 新增测试先在旧实现上得到 2 个预期失败，再在最小修复后通过。
- 测试明确覆盖“fan-out 已 reveal，后续短 call 仍按 projected ratio 排序”的 prefill 和
  decode 路径。

## 6. `0.19 workflow/s` 有效结果

两轮分别采用 baseline-first 和 HexAgenT-first。每个 mode 独立 warmup，且每个 replay
使用唯一 runtime namespace。run-level 指标取两轮算术平均：

| 指标 | Baseline | HexAgenT | 相对变化 |
| --- | ---: | ---: | ---: |
| Mean SLO ratio | 1.8815x | 1.9389x | +3.05% |
| Median SLO ratio | 1.9537x | 2.0060x | +2.68% |
| P95 SLO ratio | 2.4674x | 2.4530x | -0.58% |
| Max SLO ratio | 2.6376x | 2.5468x | -3.44% |

将两轮的 40 个 workflow-runs 合并后，归档脚本当时用线性插值得到 baseline/HexAgenT 的
P95 ratio `2.5193x/2.4874x`、P99 `2.7074x/2.6214x`，改善 `1.27%/3.18%`。为统一
ShareGPT、BFCL 和 HotpotQA 的统计口径，2026-08-12 总报告改用 runner 的 nearest-rank
定义；同一批 40 个样本对应 P95 `2.5138x/2.4859x`、P99 `2.7614x/2.6893x`，改善
`1.11%/2.61%`。HexAgenT 改善 pooled P95/P99，但牺牲 mean/median，符合 tail-ratio
objective 的方向；收益幅度仍小且单轮方差较大，不能表述为对每个 workflow 都更快。

两轮 HexAgenT planner coverage 分别为 11.3% 和 10.2%，planner invocation P95 为
1.18--1.29ms。第一轮 prefill baseline/HexAgenT utilization 均值为 75.8%/74.9%；第二轮为
76.2%/70.3%；全部 P/D GPUs 的 P95 utilization 均为 100%。该压力点既能形成 prefill 和
decode batch，又保留约 10% calls 供 planner 改写，优于此前没有 queue pressure 的轻载点。

逐 workflow 对照显示，HexAgenT 并不是普遍降低 latency，而是重新分配尾部风险。第一轮最差
baseline workflows `hotpotqa_11` 和 `hotpotqa_13` 的 ratio 分别降低约 0.54x 和 0.22x，
同时部分中等风险 workflows 变慢。第二轮极端 workflow 集合发生变化，说明 20-workflow
样本仍有较高运行方差。

## 7. 待验证事项

- 增加独立 repetitions 或更多不同 HotpotQA workflows，给 P95/P99 改善计算置信区间。
- 继续检查 admission wait 与 critical-path call 的关联，在不降低 prefill utilization 的前提下
  减少 global queue 中的无效等待。
- 修复 router revision watcher：sidecar 重启后 watcher 应重连，而不是永久退出。正式实验
  当前通过同时重启 sidecar 和两个 router 规避该问题。
- `0.20 workflow/s` 曾出现 request 卡住，不能作为正常饱和点；当前推荐压力点为 1P+5D、
  `0.19 workflow/s`。

## 8. 最终归档

正式结果已保存为：

- Pod：`/workspace/hexgent_e2e/a100_hotpotqa_hexgent_valid_20260811.tar.gz`
- 本地：`real_trace_profiles/a100_hotpotqa_hexgent_valid_20260811.tar.gz`
- SHA-256：`879bb6a0e14b1460f0f31be14edc70d513576dad43227655755fd8a8ec8c53ae`

归档只包含两轮 runtime namespace 隔离后的有效结果、输入/profile、GPU/snapshot/planner
telemetry、warmup、aggregate summary 和对应源码；受 state reuse 污染或未完成的调试结果未纳入。

## 9. 早期 `2.1740x` P95 结果的解释

关闭 pairwise local search、保持 high-first projected ratio 后得到的 `2.1740x`，是
`0.17 workflow/s` 下的一次 HexAgenT-only 消融运行。它曾被临时与其他运行中的 baseline
`2.2130x` 和 `2.3290x` 比较，用于形成“local search 可能扰乱 fan-out barrier”的调试假设，
但不属于可发布的 paired A/B，原因如下：

- HexAgenT 与 baseline 不在同一 runner invocation 中，无法控制运行时波动；
- 当时尚未为每个 mode 分别执行 32-request warmup，后运行的 mode 存在约 5% 热状态优势；
- 当时 replay 尚无 runtime namespace，多轮复用同一 workflow ID 可能继承 planner 的旧 DAG
  state；
- 它只运行 HexAgenT 一次，20-workflow 样本的 P95 对最慢两三个 workflows 很敏感；
- 到达率为 0.17，而最终实验为 0.19，排队压力和绝对 ratio 本来就不同。

因此，`2.1740x` 仍可作为关闭 local search 的调试证据，但不能用于计算最终提升。正式数字只
采用逐模式 warmup、runtime namespace 隔离、两种 mode 顺序都完成的 `0.19` 两轮结果。
“planner CPU 不是主要瓶颈”的判断仍然成立：早期 planner P95 为 2.76ms，最终有效运行进一步
降至 1.18--1.29ms；差异主要来自调度顺序、global admission 和运行方差，而不是 solver CPU。

统计口径补充：本文件归档时 pooled P95/P99 使用线性插值，因此报告 `1.27%/3.18%`；
2026-08-12 三 trace 总报告统一改用 runner 的 nearest-rank 定义，对应改善
`1.11%/2.61%`。两者使用相同 40 个 workflow-runs，差异仅来自分位数定义。
