# 2026-08-04 Admission Credit and Greedy Semantics

## 1. 调查问题

本轮回答两个问题：

1. 降低 router prefill admission credit 会怎样影响 prefill 与 decode？
2. 当前 HexAgenT 是否按 projected SLO ratio 全局排序，再按顺序 greedy 选择预计最早完成的 worker？

本轮只核对代码和已有 telemetry，没有修改 scheduler。

## 2. Admission credit 的实际含义

`max_inflight_per_prefill` 统计已经由 router dispatch、但 prefill HTTP response 尚未返回的
requests。统计范围通过 snapshot 与 router state 合并，包含 prefill worker 的
bootstrap、waiting、inflight 和 running。Prefill response 返回后 credit 释放；后续 KV
transfer/decode 生命周期不继续占用 prefill credit。

因此降低 credit 的直接效果是限制每个 prefill worker 同时持有的未完成 prefill requests。
它不会直接修改 decode `max_running_requests`，但会间接影响 decode：

- credit 太低：prefill 可选工作不足，batch token 数降低或出现空闲，decode 得不到持续输入；
- credit 合理：保留足够 prefill work，同时让其余 requests 留在 global queue 等待新 plan；
- credit 太高：prefill 未必更快，但更多 requests 提前锁定 P/D placement，planner 可修订空间减少；
- 如果 prefill 供给显著快于 decode，适度限制 credit 还能减少集中 KV transfer 与 decode burst。

## 3. Compute-bound prefill 是否需要大量 local requests

Compute-bound 不代表 batch size 不重要。较大的 prefill token batch 可提高 GEMM/tensor-core
利用率并摊薄 launch/scheduler overhead，但真正需要准备的是“足够的 input tokens”，而不是
固定数量的 requests。

当前实现的 credit 仍按 request 数计算；worker 的实际 prefill batch 同时受
`max_prefill_tokens=16384` 和 `prefill_max_requests=32` 约束。最新 trace 的真实分布为：

| Trace | 平均 input tokens | P50 | P95 | 按平均长度填满 16K token 所需 requests |
|---|---:|---:|---:|---:|
| ShareGPT | 771 | 696 | 1795 | 约 21 |
| BFCL | 6658 | 6963 | 8093 | 约 3 |
| HotpotQA | 1772 | 1658 | 2085 | 约 9 |

所以不能给三条 trace 统一套用很小的 credit：

- credit 8 对 ShareGPT 平均只能准备约 6.2K tokens，容易 underfill；
- credit 8 对 BFCL 已远超 16K token budget；
- credit 8 对 HotpotQA 平均约 14.2K，接近一批；
- credit 16 对 BFCL/HotpotQA 通常充足，对 ShareGPT 平均约 12.3K，仍未必填满。

推荐方向是 token-aware credit：根据 fresh snapshot 中 running/waiting input tokens，允许每个
prefill worker 至少保留一批 `max_prefill_tokens` 加小量 headroom，其余 requests 留在 global
queue。正式实现前应先做 credit 8/16/32 ablation，同时记录实际 prefill batch tokens、GPU
utilization、decode running batch 和 planner revision coverage。

## 4. 当前 HexAgenT 的通用 greedy 逻辑

用户描述的总体方向正确，但不是简单的“一次排序 + 一次 placement”。通用
`projected_finish` 分支执行以下逻辑：

1. 对每个全局可调度 call，枚举可选 prefill/decode pair。
2. 使用当前虚拟 worker availability 估计 prefill end、transfer ready、decode start/end。
3. 取该 call 最早完成的 P/D pair，计算 projected workflow completion 与 horizon 的 ratio。
4. 优先选择 projected ratio 更高、即 SLO pressure 更大的 call。
5. 将该 call 的 P/D assignment 写入虚拟 availability/reservation。
6. 对剩余 calls 重复计算，因此后一个 call 会看到前面 assignments 占用的 worker 时间。

当 pending calls 不超过 `GREEDY_RECOMPUTE_LIMIT=32` 时，排序和 placement 按上述过程动态
交替进行。超过 32 时，代码先按 priority key 排序，再逐个选择预计最早完成的 pair，避免
solver 成本过高。

## 5. 不是所有分支都只按 projected ratio

- 普通 calls：projected ratio 优先。
- 已 reveal fan-out siblings：critical-path pressure 优先，再比较 projected ratio、critical
  path service 和 child count。
- 短串行 batch：使用 call laxity，再按 reveal sequence tie-break。
- 部分 large short-decode cluster：使用额外的 decode virtual-load balancing 分支。

因此如果要验证“统一 projected ratio”是否更好，需要作为单独 ablation，而不能假设当前所有
calls 已经采用同一种 priority key。

## 6. Planner 顺序如何落实到 worker

Planner 为排序后的 calls 写入 `priority_score=rank`，rank 0 最高。当前真实 worker 启动参数
启用了 priority scheduling，并设置 low-priority-values-first，因此：

- router global queue 优先 grant 较小 rank；
- 同一 worker 中尚未执行的 waiting requests 按较小 rank 优先，再用 FCFS tie-break；
- safe `priority_update` 可以修改尚未执行的 local waiting request；
- prefill/KV transfer/decode 已开始后不再跨 worker 迁移；
- decode running batch 不是按 rank 串行逐个执行，continuous batching 会共同推进 active requests。

所以“进入同一个 worker 的全局顺序代表 local priority”在 waiting/admission 阶段成立，但不能
把它理解为 decode running batch 中严格的一条串行队列。

## 7. 与最新实验的关系

最新 4P+4D 运行所有成功 HexAgenT requests 都是 revision-0 fallback，所以本节描述的
projected-ratio/greedy decisions 没有在该运行中被执行。必须先获得非零 planner revision
覆盖率，才能根据真实 performance 验证上述策略。

