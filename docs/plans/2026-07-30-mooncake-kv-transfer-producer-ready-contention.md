# Mooncake KV 传输与 Producer-Ready 尾延迟

当前 transfer_duration 的定义是请求进入 decode transfer queue 到进入 decode wait queue，包含：
- decode transfer queue 排队
- Mooncake producer/consumer 协调与 metadata
- KV block 注册、查找和就绪检查
- 实际 NVLink 数据搬运
- 接收端状态切换与确认

## 目的

本文解释 P-D disaggregated serving 中一次 KV transfer 包含哪些时间，以及为什么多个 prefill request 在接近的时刻完成、并流向同一 decode worker 时，会显著拉长 `producer_ready_ms` 的尾延迟。

实验对象为单机双 H100 上的 Llama 3.1 8B 1P+1D 服务，SGLang 使用 Mooncake 的 intra-node NVLink transport。本文中的实测数值只描述该运行环境，不能直接视为 H100 NVLink 的峰值带宽。

## KV 数据量

Llama 3.1 8B 使用 32 层、8 个 KV heads、每个 head 维度 128，KV cache 为 BF16。每个 token 的 K/V 数据量为：

```text
2 (K 和 V) x 32 layers x 8 KV heads x 128 head_dim x 2 bytes
= 131,072 bytes/token
= 128 KiB/token
```

因此，一个 request 的理论 KV payload 为：

```text
payload_bytes = prompt_tokens x 128 KiB
```

| Prompt 长度 | 理论 KV payload |
| --- | --- |
| 1,700 tokens | 212.5 MiB |
| 2,000 tokens | 250.0 MiB |
| 2,100 tokens | 262.5 MiB |

这是模型 KV tensor 的逻辑数据量，不包括 Mooncake 元数据、地址列表、对齐、分块和控制消息开销。若使用不同模型、KV dtype、TP size 或 KV-head 配置，必须重新按公式计算。

## 时间定义与测量边界

一次 P-D transfer 不是单个 CUDA memcpy。当前日志和代码中至少有五类时间，含义不同。

| 指标 | 起止边界 | 包含的主要开销 | 当前获取方式 | 是否可直接用于计算原始链路带宽 |
| --- | --- | --- | --- | --- |
| `queue_ms` | KV chunk 入 Mooncake sender queue 到 transfer worker 开始处理 | sender queue 等待、线程池排队 | `KVTransferTiming.record_transfer` 的 `queued_ns -> started_ns` | 否 |
| `active_ms` | sender 的 `send_kvcache` 工作区间 | Python/C++ 提交、per-layer future、部分 native call 等待 | `started_ns -> completed_ns` | 否，不能保证只覆盖 GPU copy |
| `sync_critical_ms` | 第一个 `batch_transfer_sync` 进入到最后一个返回 | 每层/每块 native Mooncake sync 调用的临界路径 | `conn.py` 围绕 `engine.batch_transfer_sync(...)` 的 wall clock | 最接近 bulk path，但仍是 CPU wall-clock 边界 |
| `producer_ready_ms` | 第一个 KV chunk 排队到 prefill 端可通知 decode | 上述 queue/active、所有 layer/block 完成、AUX metadata、状态轮询和 producer 完成 | `first_queued_ns -> producer_ready_ns` | 否 |
| `decode_ready_ms` / `transfer_duration` | decode 收到完成状态到把 KV commit 到 decode waiting queue；或 decode request 记录的端到端 transfer duration | decode receiver、AUX metadata 校验、目标 KV allocation、状态消息、等待与 commit | `HEXGENT_KV_TRANSFER_DECODE_READY`、request time stats | 否 |

核心代码边界如下：

```python
# python/sglang/srt/disaggregation/mooncake/conn.py
started_ns = time.perf_counter_ns()
result = self.engine.batch_transfer_sync(
    mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
)
completed_ns = time.perf_counter_ns()
```

`batch_transfer_sync` 是 Mooncake native API 的同步返回边界，但仅在 Python 调用外使用 CPU wall-clock 时，无法单独证明 GPU copy 的纯 DMA 时长。2026-08-05 的 A100 校准已经按本文要求，在 Mooncake NVLink transport 内部真实 copy enqueue 前后插入同一 CUDA stream 的 `cudaEventRecord`，并在 completion 后调用 `cudaEventElapsedTime`。因此现在同时具备 native CUDA-copy 时间和 API wall-clock，两者必须分开解释。

```python
# KVTransferTiming 的请求级 sender 计时
def record_transfer(self, queued_ns, started_ns, completed_ns, payload_bytes):
    if not self._first_started:
        self.queue_wait_ns = max(0, started_ns - queued_ns)
    self.active_ns += max(0, completed_ns - started_ns)

def end_to_end_ns(self, completed_ns):
    return max(0, completed_ns - self.first_queued_ns)
```

producer 仅在所有 chunk 和目标 rank 成功后才变为 ready：

```python
if len(polls) == req.required_dst_info_num:
    self.update_status(req.room, KVPoll.Success)
    producer_ready_ns = time.perf_counter_ns()
    e2e_ms = timing.end_to_end_ns(producer_ready_ns) / 1e6
    self.sync_status_to_decode_endpoint(..., producer_ready_ns)
```

## 已有实测结果

以下数值来自此前的 H100 1P+1D Mooncake telemetry。它们是请求级或 sender-side wall-clock 时间，**不是**“250 MiB 在 NVLink 上实际 copy 了多少毫秒”的 CUDA-event 测量。

| 配置 | `queue_ms` P50 / P95 | `active_ms` P50 / P95 | `producer_ready_ms` P50 / P95 | 解释 |
| --- | --- | --- | --- | --- |
| transfer cap = 8 | 4.24 / 12.52 ms | 0.56 / 1.92 ms | 5.86 / 14.93 ms | 较保守的并发上限 |
| transfer cap = 16 | 未单独汇总 | 未单独汇总 | 9.97 / 84.95 ms | 更高并发引发明显 readiness 尾延迟 |

另一次 10-workflow replay 中，decode request time stats 的 `transfer_duration` 为 P50 `149.96ms`、P95 `224.39ms`。该指标覆盖 decode 端等待/commit 和请求生命周期相关开销，不能与 `active_ms` 混用，也不能倒推出 NVLink 带宽。

以上 H100 数据只能说明：bulk transport 被配置为 intra-node NVLink，且 producer-ready 的尾延迟主要受请求级排队和完成协议影响。它们自身不能倒推出纯 payload-copy 带宽。下一节的 A100 数据则来自新增的 native CUDA-event 测量，测量边界不同，不能与这里的 H100 request-level telemetry 混算。

## A100 Native CUDA-Event 校准（2026-08-05）

### 环境与方法

校准节点为单机 8 张 `NVIDIA A100-SXM4-80GB`；`nvidia-smi topo -m` 显示任意测试 GPU 对为 `NV12`。测量分为两层：

1. 直接调用 `cudaMemcpyPeerAsync`，在同一 CUDA stream 上记录 start/end event，并在目标 GPU 完整校验输出 buffer。这一层测量硬件 P2P copy 基线。
2. 使用带 CUDA-event telemetry 的 Mooncake v0.3.9 intra-node NVLink transport。每次传输 250 MiB，拆成 50 个 5 MiB segment；native 层对每个真实 copy 记录 event，并汇总为 `[HEXGENT_CUDA_COPY_BATCH]`。同时 Python 在 `batch_transfer_sync_write` 外记录同步 API wall-clock。

两层测试均在空载、单 initiator、单 target、单 transfer 下完成。Mooncake 正式数据包含 3 次 warmup 后的 20 次传输；target 对 250 MiB buffer 做分布式内容校验，initiator/target 均正常退出，日志明确选择 `Intra-Node NVLink transport`。

### 直接 CUDA P2P 基线

| GPU pair | Payload | CUDA-event time | 有效单向带宽 |
| --- | ---: | ---: | ---: |
| GPU 0 -> 1 | 64 MiB | 0.245 ms | 255.0 GiB/s |
| GPU 0 -> 1 | 256 MiB | 0.959 ms | 260.7 GiB/s |
| GPU 0 -> 1 | 1024 MiB | 3.815 ms | 262.1 GiB/s |
| GPU 0 -> 4 | 64 MiB | 0.244 ms | 256.2 GiB/s |
| GPU 0 -> 4 | 256 MiB | 0.958 ms | 261.1 GiB/s |
| GPU 0 -> 4 | 1024 MiB | 3.815 ms | 262.1 GiB/s |

这里的带宽由 `payload / cudaEventElapsedTime` 计算，是经过内容校验的实际单向 GPU P2P copy 吞吐，不是根据 producer-ready 或 request latency 反推的结果。它约为 261 GiB/s，而不是把 12 条 25 GB/s link 的标称速率直接相加得到的理论值。

### Mooncake 250 MiB 结果

| 指标 | P50 | P95 | 范围/有效带宽 | 包含内容 |
| --- | ---: | ---: | ---: | --- |
| Native CUDA-event batch copy sum | 1.293 ms | 1.302 ms | 1.284--1.303 ms，约 189 GiB/s | 50 个 5 MiB 真实 GPU copy 的 event elapsed time 总和 |
| Python `batch_transfer_sync_write` wall-clock | 4.292 ms | 4.355 ms | P50 约 56.9 GiB/s | native copy 加提交、查找、同步和完成处理 |
| Direct `cudaMemcpyPeerAsync` 256 MiB 基线 | 0.959 ms | - | 260.7 GiB/s | 单块直接 P2P copy，不经过 Mooncake |

因此，“250 MiB 在 Mooncake NVLink path 上实际 copy 了多少毫秒”的本次答案是：native CUDA events 汇总中位数约 `1.293ms`。若问 Python 调用从进入到同步返回花了多久，则答案是中位数约 `4.292ms`。二者描述的是不同边界。

### 为什么 API wall-clock 更长

受控 benchmark 将 250 MiB 人工构造成 50 个 5 MiB transfer entry。一次 `batch_transfer_sync_write` 在 API 层是一个 batch，但当前 intra-node transport 会逐项执行 50 次同步 `cudaMemcpy`，因此不是一次 250 MiB DMA。直接 256 MiB 单块 copy 约 0.959ms；按相同带宽换算 250 MiB 约需 0.936ms，而 Mooncake 多 entry GPU-copy 总和约 1.293ms。约 0.357ms 的差值相当于每个 entry 约 7.1us，来自小 copy 固定成本、无法完全进入大块 copy 的稳定带宽区间及逐 entry telemetry 扰动，不能解释成 NVLink 链路本身变慢。API wall-clock 还比 native copy 多约 3ms，主要来自：

- 50 个 segment 的任务构造、提交和完成状态维护；
- IPC 地址、CUDA handle 和目标地址的查找与 relocation；
- transfer task bookkeeping 与 completion polling/synchronization；
- Python/C++ API 边界和同步返回处理。

该 50-entry 形状是校准配置，不是生产请求的固定分段数。Llama 3.1 8B 的 MHA KV cache 按 32 层分别存放 K/V，单个连续 token run 通常至少形成 64 个 K/V-layer 地址区间；token slots 不连续时还会更多。以 2,037 tokens 为例，每个 K 或 V layer region 约 3.98 MiB，64 个 region 合计约 254.625 MiB。因此生产中的“一个逻辑 KV 请求”本来就可能是一批物理 copies。将其强制合成单 copy 需要额外 pack/unpack staging buffer，未必更快。

该约 3ms 差值是在单 transfer 空载条件下观测到的 API 固定/软件开销，不包含 serving 中 transfer queue 排队、多个请求争抢 engine、decode 端 KV allocation、producer/consumer 协调或 request-level ready/commit。因此 production 中的 `producer_ready_ms` 和 `transfer_duration` 仍可能远大于 4.3ms。

### 可复现资产

Pod 上原始 direct P2P 结果位于：

```text
/workspace/hexgent_e2e/measurements/a100_nvlink_20260805/raw_cuda_peer_results.txt
```

Mooncake native 测量位于：

```text
/workspace/hexgent_e2e/measurements/a100_mooncake_native_nvlink_20260805
```

patched source/build 与 overlay 分别位于：

```text
/workspace/third_party/mooncake-0.3.9-a100
/workspace/third_party/mooncake-0.3.9-a100/build-a100-nvlink
/workspace/hexgent_e2e/mooncake_overlays/a100_cuda_events_nvlink
```

## 为什么并发会拉长 Producer-Ready

Mooncake 对一个 request 会按层、K/V block、目标 rank 拆分任务；多个 request 同时完成时，它们竞争的不只是 NVLink copy engine，还包括：

1. prefill 侧 transfer queue 与 thread-pool slot。
2. Mooncake engine 的任务提交、内存注册和 block 查找。
3. decode 侧目标 KV allocation、metadata buffer、状态接收和 transfer commit。
4. AUX metadata 与完成通知路径。大 payload 使用 intra-node NVLink；小型 CPU 可见元数据仍走辅助 TCP/ZMQ 路径。

```python
self.transfer_queues = [FastQueue() for _ in range(transfer_queue_size)]
self.executors = [
    concurrent.futures.ThreadPoolExecutor(
        transfer_thread_pool_size // transfer_queue_size
    )
    for _ in range(transfer_queue_size)
]
```

```python
futures = [
    executor.submit(process_layer, src_ptr, dst_ptr, item_len)
    for (src_ptr, dst_ptr, item_len) in layers_params
]
for future in concurrent.futures.as_completed(futures):
    status = future.result()
```

因此 cap 从 8 增至 16 时，copy 本身未必慢很多，但 sender/receiver 的排队和完成确认会被放大，最终表现为 `producer_ready_ms` 的 P95 从 `14.93ms` 升至 `84.95ms`。

## 为什么不同请求会同时结束 Prefill

请求不需要有相同 input/output 长度，也可能在很短的时间窗口内完成 prefill：

1. Prefill batching：一个 forward batch 内的请求共同完成。
2. Chunked prefill：不同长度的 prompt 会落在相同或相邻的最后一个 chunk 边界。
3. Workflow fan-out：一个 parent 完成后，同时 reveal 多个独立 child calls。
4. 多个 prefill worker：相似 batch 的完成时间可能重合，并向同一个 decode worker 汇聚。
5. DAG barrier：多个 child 在同一 dependency barrier 后一起变为 eligible。

HexAgenT 因而需要预测“某个 decode worker 何时真正 transfer-ready”，而不是只计算 `payload_bytes / nominal_bandwidth`。

## 对调度与后续测量的要求

1. 保持保守 transfer concurrency cap；不要仅为提高 in-flight bytes 而盲目提高 cap。
2. planner 应为每个 decode worker 维护 transfer credit，直到预计 producer-ready 时刻才释放。
3. latency model 应分开建模 raw-copy、sender queue、producer-ready 与 decode-ready。
4. 相近 decode worker 之间，应选择预计 transfer-ready 更早的目标，而非只比较孤立 decode latency。
5. 下一次 1P+1D 校准需要记录每 request 的 `payload_bytes`、`queue_ms`、`active_ms`、`sync_critical_ms`、`producer_ready_ms`、`decode_ready_ms`，并在 Mooncake NVLink transport 内部加入 CUDA event；只有后者能回答纯数据搬运实际用了多久。
