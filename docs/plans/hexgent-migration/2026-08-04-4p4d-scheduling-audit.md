# 2026-08-04 4P+4D Scheduling Audit

## 1. 调查目标

确认最新 Llama-8B H100 4P+4D 实验为何在 BFCL 和 HotpotQA 上未优于 baseline，回答：

- 实际使用的 arrival rate；
- HexAgenT planner 是否真正影响了 placement 和 global order；
- baseline 是否等价于 FCFS；
- 当前性能差异是否足以评价调度策略。

本轮只进行证据审计，不修改 scheduler 算法。

## 2. 证据来源

- Archive：`artifacts/20260804/hexgent_h100_4p4d_llama8b_50_50_20_20260804_r1.tar.gz`
- Replay driver：`scripts/hexgent_real_trace_collectors/run_h100_formal_comparison.sh`
- Deployment：`scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_4p4d.sh`
- Router admission：`sgl-model-gateway/src/routers/http/pd_router.rs`
- Planner telemetry、calls、worker snapshots 和 GPU CSV 均来自上述 archive。

## 3. 实验配置

- 模型：`neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`
- 硬件：8x H100，4 个 TP1 prefill workers + 4 个 TP1 decode workers
- Prefill `max_running_requests=32`，总 admission credit 上限为 128
- Decode `max_running_requests=128`
- Arrival rate：ShareGPT 3、BFCL 0.6、HotpotQA 0.2 workflows/s
- 顺序：每条 trace 都先 baseline，后 HexAgenT
- Baseline 与 HexAgenT 使用相同 trace、DAG、length、arrival seed 和 workers

## 4. 原始结果

| Trace | Baseline P95 SLO ratio | HexAgenT-labeled P95 ratio | P95 workflow latency baseline / Hex |
|---|---:|---:|---:|
| ShareGPT | 1.324x | 1.265x | 10282 / 9902 ms |
| BFCL | 1.501x | 1.690x | 6495 / 6283 ms |
| HotpotQA | 1.185x | 1.195x | 8924 / 9131 ms |

ShareGPT 两侧各有 3 个相同的空消息 HTTP 400，因此其数字本身仍是暂定值。

## 5. 关键发现

### 5.1 Planner plan 没有被执行

全量 calls 审计结果：

| Trace | 成功 Hex calls | `schedule_revision=0` | `schedule_revision>0` |
|---|---:|---:|---:|
| ShareGPT | 221 | 221 | 0 |
| BFCL | 487 | 487 | 0 |
| HotpotQA | 636 | 636 | 0 |

`pd_router.rs::hexgent_fallback_decision` 将 native fallback 明确标记为 revision 0。
同时三条 trace 的 planner telemetry 均满足：

- 所有 invocation 的 `admission_count=0`；
- 所有 invocation 的 `result_count=0`。

因此不是 planner 计算出较差计划，而是 queued admission 在 coordinator 开始求解前就已被
router 按 fallback dispatch。

### 5.2 负载没有进入 global scheduling 区间

| Trace | 请求区间最大并发 | Prefill waiting 峰值 | Decode running 峰值 |
|---|---:|---:|---:|
| ShareGPT | 23 | 0 | 9 |
| BFCL | 8 | 0 | 2 |
| HotpotQA | 20 | 0 | 4 |

这些值远低于 128 个总 prefill credits。`HexgentGlobalAdmissionQueue::next_grant_index`
允许在真实 worker credit 可用时直接选择 fallback，因此 admission 没有停留到 revision plan
发布。该行为满足“不因 planner 阻塞空闲 worker”，但意味着低负载实验不会触发 HexAgenT
global ordering。

### 5.3 Baseline 不是纯 FCFS

Baseline router 启动命令没有设置 `--policy`、`--prefill-policy` 或 `--decode-policy`。
SGLang Rust router 的默认主 policy 是 `cache_aware`，PD-specific policy 缺失时继承主 policy。
因此 baseline 是 cache-aware P/D placement 加 SGLang worker-local scheduling，不是纯 FCFS。

本轮 worker snapshot 的 local waiting 峰值为 0，所以几乎没有观察到 local FCFS/priority
顺序竞争。两次运行的 worker token 分布也都较均衡，无法把 BFCL 差值归因于明显的 placement
热点。

## 6. 结论

本轮不能得出“HexAgenT 调度策略不如 FCFS”或“BFCL 上 projected-SLO-ratio 退化”的结论。
更准确的描述是：两次运行都使用 native cache-aware fallback，差异来自独立运行中的 worker
selection、执行抖动、依赖完成时间和共享 cache 状态。固定 baseline-first 顺序还可能让第二次
运行继承不同的 prefix-cache 状态。

当前最重要的问题是实验没有覆盖 planner plan，而不是已经证明 planner objective 错误。

## 7. 下一步验证

1. 先完成无空消息 ShareGPT 50-workflow 独占 profile。
2. 每次运行强制统计 `schedule_revision>0`、dispatch reason、fallback/override 和 global queue depth。
3. 从 3/0.6/0.2 开始做 arrival-rate staircase；若 plan coverage 和 queue pressure 都为 0，
   以固定倍数增加，直到出现真实 global backlog。
4. 同时验证 prefill GPU/batch 未因 admission credit 受限，避免为了触发 planner 人为降低吞吐。
5. 只有 planner plan 覆盖率显著非零后，才比较 projected-ratio 与 baseline。
6. 使用交替运行顺序或重启/清空 cache，并做重复实验，隔离 run-order 和 cache 影响。

## 8. 本轮代码改动

为避免后续 summary 再次把 fallback-only run 当成 scheduler 对比，已修改：

- `scripts/hexgent_real_trace_collectors/summarize_scheduling_replay.py`
- `test/unit/utils/test_hexgent_scheduling_summary.py`

新增输出 `planner_plan_execution`，包括：

- `successful_hexgent_calls`
- `planner_revision_calls`
- `fallback_revision_calls`
- `missing_revision_calls`
- `planner_revision_coverage`
- `planner_effect_observed`

修改采用 TDD：新增测试首先因缺少 `planner_plan_execution` 失败，完成最小实现后该测试文件
结果为 `6 passed`。本轮没有调整 projected-SLO-ratio、critical-path priority 或 placement
策略，因为现有实验尚未执行这些 planner decisions。
