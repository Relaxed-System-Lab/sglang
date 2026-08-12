# 2026-08-04 Token-Aware Prefill Admission

## 1. 问题与根因

此前 Rust router 已将实时 worker snapshot、router inflight request identity、worker request
capacity 和 snapshot TTL 合并，用于避免重复计算 admission occupancy。但 admission 的主要限制仍是
固定 request 数：4P 配置下每个 prefill worker 最多提前接收 32 个请求。

该口径无法适配不同 trace。当前 profile 中 ShareGPT、BFCL 和 HotpotQA 的平均 input length
分别约为 771、6658 和 1772 tokens。在 `max_prefill_tokens=16384` 时，填满一批大约需要
21、3 和 9 个请求。固定 32 条会使 BFCL 请求过早离开 global queue，同时并不代表 ShareGPT
一定形成了相同的 token-level batching pressure。

## 2. 新 admission 模型

实现采用两层限制：

1. `max_running_requests` / `prefill_max_requests` 与
   `hexgent_max_prefill_queue_per_worker` 继续作为 request-count 硬安全上限。
2. fresh snapshot 提供 `max_prefill_tokens` 时，正常 admission 改由 token window 控制。

Token window 只统计尚未开始 prefill 的 bootstrap、waiting、inflight 和 router inflight 请求。
已经出现在 snapshot `running` 中的请求不占下一批 waiting token window。默认 window 为：

```text
max_prefill_tokens * (1 + hexgent_prefill_token_headroom_percent / 100)
```

默认 headroom 为 25%。在 16K batch 下，约 7000-token 的 BFCL 请求会预放 3 条；这 3 条进入
running 后，可以再预放下一批 3 条。约 770-token 的 ShareGPT 请求可预放约 27 条。窗口允许最后
一个请求略微越过 token target，避免单个长请求因长度大于剩余 token credit 而永久无法 admission。

Router 通过 `(workflow_id, call_id)` 对 snapshot 与 router inflight 去重；同一请求在两侧出现时
采用较大的 input-token 记录，不重复相加。snapshot 过期或未报告 `max_prefill_tokens` 时，退化为原有
request-count credit，避免因 telemetry 缺失阻塞 serving。

Request-count admission cap 同样只约束未开始 prefill 的 backlog，不再把 snapshot `running`
计入下一批 cap。Running batch 的数量仍由 worker 自己的 `max_running_requests` 和
`prefill_max_requests` 控制。否则短 prompt 当前批次已有 27 条时，32 条 cap 只允许准备 5 条下一批，
仍会人为制造 batch 间气泡。

## 3. 配置与实验入口

新增 CLI 参数：

```text
--hexgent-prefill-token-headroom-percent 25
```

1P+1D、2P+2D 和 4P+4D launcher 均已传递该参数。固定 request cap 仍可保留为 32 或更高，
但正常 credit 会优先由 token window 限制，因此 BFCL 不会再仅因为硬上限为 32 就提前 dispatch
32 条请求。

新增 `run_h100_sharegpt_rate_staircase.sh`，默认要求至少 128 个不同 ShareGPT workflows，并按
3/6/9/12 workflows/s 依次运行 HexAgenT。只提高 50-workflow trace 的 arrival rate 不一定能够
产生 global backlog：并发 call 总 token 数可能仍低于 4 个 prefill worker 的 token window 总和。
因此校准必须同时增加不同 workflow 数量，不能复制少量 workflow。按当前约 771-token 均值，
4 个 20.48K-token windows 至少需要约 107 个同时 active calls 才可能产生 global backlog；128
只是带余量的起点，不是固定最优值。

合适 arrival rate 的验收条件不是单看 GPU utilization，而是同时满足：

- prefill batch tokens 接近 `max_prefill_tokens`，没有持续 underfill；
- decode running batch 持续存在且无明显 starvation；
- global admission 出现非零等待，且成功请求中 `schedule_revision > 0`；
- planner revision coverage、错误率、KV transfer tail 和 P/D GPU telemetry 均正常。

## 4. 验证状态

- Python router CLI 解析测试通过。
- launcher 参数与 ShareGPT staircase 测试通过。
- 本轮涉及的 shell scripts 已转换为 LF，并通过 `bash -n`。
- Rust token-credit 测试已编写，覆盖 BFCL、ShareGPT 和 running/next-batch 分离语义。
- Windows 本机缺少 MSVC `link.exe`；本地 WSL 也缺少 `cc`，且安装系统 build dependencies
  时 apt 下载停滞。Rust 测试均在依赖编译阶段被环境阻断；需要在下一次 Linux/RunPod 构建
  extension 前执行目标 Rust tests。

当前没有新的 GPU 实验结果，因此 3/6/9/12 只是待校准 staircase，不是已经确认的最佳 arrival rate。
