# A100 Live Snapshot Alignment 与 Planner Coverage

## 1. 本轮目标

本轮验证两个 serving 语义：

1. idle prefill worker 必须允许 native PD fallback 立即 dispatch，planner 不得阻塞 serving；
2. 当 fallback worker 已无 admission credit、请求本来就必须等待时，planner 必须读取真实 P/D queue state，并可在 bootstrap 创建前覆盖 placement。

## 2. 根因与代码修复

此前 router 为 admission payload 接受最长 `snapshot_max_staleness_ms=1000` 的 cache。BFCL burst 中，planner 可能在约一秒内反复看到同一个 worker 的旧低负载状态，从而持续向该 worker 分配请求。另一个问题是 `refresh_hexgent_admission_state` 会用部分刷新结果替换完整 snapshot map，导致未返回 worker 的状态丢失。

修改文件：

- `sgl-model-gateway/src/routers/http/pd_router.rs`

新路径如下：

```text
request arrives
  -> native fallback selected
  -> global admission checks live credit
  -> credit available: dispatch immediately, no planner reservation
  -> credit unavailable: asynchronously refresh all unique P/D worker snapshots
  -> refresh router credit and merge snapshots into planner payload
  -> credit became available: dispatch fallback and stop
  -> still pending: enqueue one sidecar admission using the fresh cluster state
```

刷新复用 25ms 内的 cache；过期 worker 并发请求 `/hexgent/pd_snapshot`，每个请求最多 50ms。部分失败不会清除其他 worker 的有效状态。刷新完成后再次检查 global queue identity，已 dispatch 的请求不会创建陈旧 sidecar reservation。

测试：

- `test_hexgent_admission_poll_refreshes_live_cluster_state`
- `test_hexgent_pending_snapshot_refresh_covers_unique_worker_urls`
- Pod Linux Rust：`cargo test hexgent --lib -- --nocapture`，36 passed。

## 3. BFCL 4P+2D 结果

模型为 Llama-3.1-8B-Instruct BF16，硬件为 6x A100-SXM4-80GB，Mooncake 使用 intra-node NVLink。输入与输出长度固定复用权威 BFCL trace，SLO 使用同模型、同硬件的 A100 独占 profile。

| Rate | Scheduler | Makespan | P95 latency | Mean ratio | P95 ratio | Coverage | Prefill placement |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.3 wf/s | Baseline | 168.66s | 38,262ms | 2.5790x | 3.7298x | - | 142/133/114/98 |
| 0.3 wf/s | HexAgenT | 167.17s | 33,961ms | 2.2364x | 3.2403x | 7.6% | 110/121/138/118 |
| 0.4 wf/s | Baseline | 156.08s | 83,461ms | 4.8160x | 8.2418x | - | 111/131/114/131 |
| 0.4 wf/s | HexAgenT | 141.35s | 52,540ms | 3.7712x | 6.2291x | 29.0% | 114/120/131/122 |

0.3 workflow/s 是当前更合理的可持续推荐点。0.4 已出现明显 queueing amplification，适合作为压力点而不是稳态配置。两点中 HexAgenT 均改善 P95 ratio，同时 placement 没有重现旧实验的 253/197/274/489 skew。

## 4. ShareGPT 2P+4D 诊断

在 24 workflows/s 下将每个 prefill 的 router credit 降至 16 requests，仍保留一个 16-request batch。结果仍为 0% planner coverage，因为两个 prefill 始终有 admission credit；四个 decode 才是瓶颈。

| 模式 | Makespan | P95 ratio | Coverage | Decode placement |
| --- | ---: | ---: | ---: | --- |
| Baseline | 34.30s | 1.4128x | - | 115/123/121/118 |
| Pending-only planner | 36.05s | 1.4358x | 0% | 105/114/151/107 |
| Planner-first diagnostic | 34.98s | 1.8189x | 100% | 102/142/171/62 |

强制 planner-first 没有改善结果，反而暴露 decode model 的偏斜。生产实现不得把 coverage 当作目标，也不得为了 planner 人为阻塞 idle prefill。ShareGPT 当前仍以已验证的 1P+5D、24 workflows/s 作为更合适拓扑；该配置此前把 P95 ratio 从 2.0327x 降至 1.8451x，coverage 为 28.3%。

## 5. 证据路径

本地：

- `real_trace_profiles/a100_llama8b_4p2d_bfcl_live_snapshot_rate03_20260805`
- `real_trace_profiles/a100_llama8b_4p2d_bfcl_live_snapshot_rate04_20260805`
- `real_trace_profiles/a100_llama8b_2p4d_sharegpt_live_snapshot_credit16_rate24_20260805`
- `real_trace_profiles/a100_llama8b_2p4d_sharegpt_planner_first_rate24_20260805`

Pod：

- `/workspace/hexgent_e2e/topology_4p2d_live_snapshot_20260805`
- `/workspace/hexgent_e2e/topology_4p2d_live_snapshot_rate04_20260805`
- `/workspace/hexgent_e2e/topology_2p4d_live_snapshot_credit16_20260805`
- `/workspace/hexgent_e2e/topology_2p4d_planner_first_20260805`

## 6. 下一步

优先校准 decode worker 的 running-batch occupancy、remaining output tokens 和 planner reservation，使 planner 的 decode queue work 与 snapshot 对齐。完成后再重复 ShareGPT planner-first 作为诊断；production 路径仍保持 pending-only planning。HotpotQA 则需要在相同 live-snapshot 语义下重新验证 fan-out critical-path tail。
