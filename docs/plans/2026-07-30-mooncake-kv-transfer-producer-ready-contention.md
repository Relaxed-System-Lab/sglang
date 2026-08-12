# Mooncake KV 传输与 Producer-Ready 尾延迟

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

`batch_transfer_sync` 是 Mooncake native API 的同步返回边界，但现有 telemetry 仍由 CPU wall-clock 记录。它不是 CUDA event，无法单独证明 GPU copy 的纯 DMA 时长。要测实际 CUDA copy，必须在 Mooncake NVLink transport 内部、真实 copy enqueue 前后插入同一 CUDA stream 的 `cudaEventRecord`，并在 completion 后调用 `cudaEventElapsedTime`。

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

因此当前证据只能说明：bulk transport 被配置为 intra-node NVLink，且 producer-ready 的尾延迟主要受请求级排队和完成协议影响。当前没有可信的 CUDA-event 纯 payload-copy 带宽数值；不能据此宣称已测得 500 GB/s 或任何固定 NVLink 带宽。

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
