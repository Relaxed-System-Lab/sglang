# A100 三类 Trace 实验汇总

## 1. 范围与有效性

本文汇总单机 6 张 A100-SXM4-80GB 上的最终有效实验。模型为
Llama-3.1-8B-Instruct BF16、每个 worker TP1、关闭 radix cache，KV 由 Mooncake
intra-node NVLink 传输。Baseline 与 HexAgenT replay 完全相同的 DAG 及固定
input/output token 长度；SLO 分母来自同模型、同 GPU、同 dtype、同 cache 设置的独占
profile。所有结果均为零请求错误。

三个 trace 的最佳已验证配置不同，因为其阶段服务特征不同：

| Trace | Workflows / calls | Input P50 | Output P50 | TTFT P50 | Decode P50 | Topology / rate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ShareGPT | 100 / 477 | 742 | 229 | 65.33 ms | 2,468.95 ms | 1P+5D / 24 wf/s |
| BFCL-v3 | 50 / 487（A/B 子集） | 7,679 | 24 | 569.31 ms | 267.42 ms | 4P+2D / 0.3 wf/s |
| HotpotQA/LATS | 20 / 636 | 1,658 | 54 | 126.21 ms | 578.84 ms | 1P+5D / 0.19 wf/s |

完整 BFCL 独占 profile 仍包含 100 workflows / 1,213 calls。HotpotQA 的正式数值
合并两轮相反 mode 顺序的 40 个 workflow-runs；ShareGPT/BFCL 为单轮 clean A/B，
因此尚不能给出跨重复置信区间。

## 2. Performance 对比

本节统一采用 runner 的 nearest-rank 分位数定义：对排序后长度为 `n` 的样本，Pq 取
`ceil(q*n)` 对应的观测值，不在相邻样本间插值。这样 ShareGPT/BFCL 单轮结果与 HotpotQA
pooled 结果使用同一统计口径。

### 2.1 Workflow SLO ratio

| Trace | Metric | Baseline | HexAgenT | 相对变化 |
| --- | --- | ---: | ---: | ---: |
| ShareGPT | Mean | 1.5198x | 1.4650x | -3.61% |
| ShareGPT | P95 | 2.0327x | 1.8451x | **-9.23%** |
| ShareGPT | P99 | 2.3700x | 2.1543x | **-9.10%** |
| BFCL-v3 | Mean | 2.5790x | 2.2364x | -13.28% |
| BFCL-v3 | P95 | 3.7298x | 3.2403x | **-13.12%** |
| BFCL-v3 | P99 | 4.4703x | 4.4981x | +0.62% |
| HotpotQA | Mean | 1.8815x | 1.9389x | +3.05% |
| HotpotQA | P95 | 2.5138x | 2.4859x | **-1.11%** |
| HotpotQA | P99 | 2.7614x | 2.6893x | **-2.61%** |

HexAgenT 在 ShareGPT 和 BFCL 的 mean/P95 均优于 baseline；HotpotQA 改善
P95/P99/max，但牺牲 mean/median，符合 tail-ratio objective 的方向。BFCL 只有 50 个
workflow，P99 几乎由最慢单个 workflow 决定；其 P99/max 未改善，不能用 P95 结论外推。

### 2.2 Workflow latency

| Trace | Metric | Baseline | HexAgenT | 相对变化 |
| --- | --- | ---: | ---: | ---: |
| ShareGPT | Mean | 19,314.9 ms | 18,609.0 ms | -3.66% |
| ShareGPT | P95 | 32,543.4 ms | 31,349.0 ms | -3.67% |
| ShareGPT | P99 | 34,002.9 ms | 32,734.7 ms | -3.73% |
| BFCL-v3 | Mean | 20,126.5 ms | 17,637.6 ms | -12.37% |
| BFCL-v3 | P95 | 38,262.1 ms | 33,960.9 ms | -11.24% |
| BFCL-v3 | P99 | 49,558.8 ms | 37,931.9 ms | -23.46% |
| HotpotQA | Mean | 29,260.0 ms | 30,108.5 ms | +2.90% |
| HotpotQA | P95 | 42,359.9 ms | 42,383.2 ms | +0.05% |
| HotpotQA | P99 | 43,766.3 ms | 44,787.5 ms | +2.33% |

HotpotQA 的 latency P95 未改善但 SLO-ratio P95 改善，原因是两者的尾部 workflow
集合并不相同：ratio 按每个 workflow 自己的 critical-path SLO 归一化，而绝对 latency
只取最慢完成时间。

### 2.3 HotpotQA 早期结果与正式结果的口径差异

调试阶段曾得到 HexAgenT P95 ratio `2.1740x`，并与另外两次 baseline 的
`2.2130x` 和 `2.3290x` 横向比较。该比较看起来比本报告中的最终改善更明显，但它不是
paired A/B，不能作为最终性能结论：

- `2.1740x` 来自 `0.17 workflow/s` 的一次 HexAgenT-only 消融；正式实验使用
  `0.19 workflow/s`。
- 三个数字来自不同 runner invocation，未控制瞬时系统波动和 mode 执行顺序。
- 早期实验没有为每个 mode 分别进行 32-request warmup，后运行的 mode 可能获得约 5%
  的热状态优势。
- 早期 replay 没有 runtime namespace，多轮复用 workflow ID 时可能继承旧 planner DAG
  state。
- 只有 20 个 workflows，P95 由最慢的少数 workflows 决定，单轮方差较大。

最终 HotpotQA 结论只使用 `0.19 workflow/s` 下两轮顺序互换的 paired A/B。若先对每轮
P95 分别计算再取均值，baseline/HexAgenT 为 `2.4674x/2.4530x`，改善约 `0.58%`；若将
两轮共 40 个 workflow-runs 合并并按本报告的 nearest-rank 口径计算，P95 为
`2.5138x/2.4859x`，改善 `1.11%`。归档内早期 `aggregate_summary.json` 使用线性插值，
对应 `2.5193x/2.4874x` 和 `1.27%`；差异仅来自分位数定义。本报告表格采用统一后的
nearest-rank pooled 口径。`2.1740x` 仅支持“关闭 pairwise local search 值得进一步验证”
这一调试判断，不支持宣称更大的正式提升。

## 3. GPU 资源利用率

下表是 replay 窗口内 `nvidia-smi utilization.gpu` 的时间平均值。P95 在所有正式
运行、所有 P/D GPU 上均为 100%，说明都形成过满载 batch；均值较低不等于
`max_running_requests` 限制，尤其 HotpotQA barrier 会产生自然空档。

| Trace / scheduler | Prefill 平均 | Decode 平均 | 单 worker 范围 | Planner coverage |
| --- | ---: | ---: | ---: | ---: |
| ShareGPT baseline | 67.3% | 88.7% | D: 80.7%--93.7% | - |
| ShareGPT HexAgenT | 76.9% | 92.0% | D: 89.2%--95.5% | 28.3% |
| BFCL baseline | 33.3% | 41.0% | P: 27.7%--38.5%; D: 38.1%--43.8% | - |
| BFCL HexAgenT | 31.4% | 44.3% | P: 29.6%--33.9%; D: 42.4%--46.1% | 7.6% |
| HotpotQA baseline | 76.0% | 57.9% | 两轮均值；全卡 P95=100% | - |
| HotpotQA HexAgenT | 72.6% | 54.6% | 两轮均值；全卡 P95=100% | 10.2%--11.3% |

ShareGPT 是 decode-heavy，1P+5D 能同时提高两阶段利用率并减小 decode 卡间极差。
BFCL 单 call 极度 prefill-heavy，但 workflow 内 calls 串行 reveal，四个 prefill worker
呈现短促的 100% 峰值而非持续满载；0.3 wf/s 是可持续对照点，继续加速到 0.4 wf/s
会形成 queueing amplification。HotpotQA 的 width-5 fan-out 和 barrier 同时造成 burst 与
空档，因此不能只看全程均值选择 P/D 比例。

Planner CPU 不是上述运行的主要瓶颈：ShareGPT/BFCL/HotpotQA 的 planner P95 分别为
5.85 ms、0.23 ms、1.18--1.29 ms。HotpotQA 较大的 router selection wall time主要是
global admission 等待 prefill credit，不是 solver CPU 时间。

## 4. KV payload 与传输耗时

Llama-3.1-8B BF16 的 MHA KV cache 为 128 KiB/token。独占 profile 对应的 payload：

| Trace | KV P50 | KV P95 | Native-copy P50/P95（推导） | API wall P50/P95（推导） |
| --- | ---: | ---: | ---: | ---: |
| ShareGPT | 92.75 MiB | 220.85 MiB | 0.48 / 1.14 ms | 1.59 / 3.79 ms |
| BFCL-v3 | 959.88 MiB | 1,236.48 MiB | 4.96 / 6.39 ms | 16.47 / 21.22 ms |
| HotpotQA | 207.25 MiB | 260.34 MiB | 1.07 / 1.35 ms | 3.56 / 4.47 ms |

表中每个 trace 的时间是按 payload 与下列**实测带宽**线性换算的派生值，不是正式 A/B
运行中的逐请求 CUDA-event 测量：

| 实测边界 | Payload | P50 | P95 | 有效带宽 |
| --- | ---: | ---: | ---: | ---: |
| Direct `cudaMemcpyPeerAsync` | 256 MiB | 0.959 ms | - | 260.7 GiB/s |
| Mooncake native CUDA-event copy sum | 250 MiB | 1.293 ms | 1.302 ms | 约 189 GiB/s |
| Python `batch_transfer_sync_write` wall-clock | 250 MiB | 4.292 ms | 4.355 ms | 约 56.9 GiB/s |

Native CUDA event 覆盖真实 GPU copy；Python API wall 还包含多 entry 构造、地址查找、
bookkeeping、completion polling 和同步返回。生产请求还可能包含 transfer queue、decode
allocation 和 producer/consumer 协调，因此不能用 0.959 ms 声称完整请求已 ready。

若定义“KV transfer 小于 `min(TTFT, decode service)` 的 10%”，逐请求计算得到所需有效
带宽 P95 为 ShareGPT 17.12、BFCL 60.19、HotpotQA 25.87 GiB/s。189 GiB/s native copy
均有充分余量；56.9 GiB/s 空载 API 边界对 ShareGPT/HotpotQA 有余量，对 BFCL 的严格
10% P95 标准略低。因此 BFCL 仍需监控 request-level producer-ready 尾延迟，但当前证据
不支持将 raw NVLink payload bandwidth 视为主瓶颈。

## 5. 结论与限制

- ShareGPT 的正式单轮 A/B 中，HexAgenT P95 SLO ratio 改善 9.23%，吞吐和 GPU
  utilization 同时提高。
- BFCL 的 0.3 wf/s clean A/B 中，P95 SLO ratio 改善 13.12%，但 P99 未改善且样本较小。
- HotpotQA 的两轮配对实验中，nearest-rank P95/P99 ratio 改善 1.11%/2.61%，mean ratio 恶化
  3.05%；目前只能声称小幅 tail 改善。
- A100 NVLink copy 本身足够快；正式 A/B 仍缺少每请求 producer-ready/CUDA-copy 的
  联合 telemetry，不能把派生 copy time 当成端到端 transfer latency。
- 下一轮正式结果应至少做三次独立重复并报告 bootstrap confidence interval；HotpotQA
  应增加不同 workflows，BFCL 应单独验证 P99。

## 6. 证据资产

- `real_trace_profiles/a100_hexgent_final_valid_assets_20260810.tar.gz`
- `real_trace_profiles/a100_hotpotqa_hexgent_valid_20260811.tar.gz`
- [A100/H100 profile 资产索引](../../real_trace_profiles/README.md)
- [Mooncake 测量边界详解](../plans/2026-07-30-mooncake-kv-transfer-producer-ready-contention.md)
