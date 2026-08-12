# Token-aware Admission Recovery and Full-profile Replay

## 1. 本轮目标

本轮工作修正两个会影响正式实验可信度的问题：

1. token-aware prefill admission 和 revisioned execution-plan 代码此前只存在于
   `D:\wt\hexgent-admission` 的未提交改动中，没有进入主工作区；
2. A100/Llama-8B profiling 之前只使用 50 ShareGPT、50 BFCL 和 20 HotpotQA
   workflow，而正式 workload 应复用已有 H100/Llama-70B profiles 中固定的
   workflow、DAG 和 input/output token length。

## 2. 代码恢复与验证

已将 `hexgent-admission` worktree 中仍然有效的改动合并到主工作区，并保留主工作区更新的
pairwise local search、SSE parser、Mooncake overlay 和 A100 启动逻辑。生产路径仍为 Rust
PD router 加 `python/sglang/hexgent` planner sidecar；旧的 Python `hexgent_lb` 和
`hexgent_vendor` 不属于当前 serving data plane。

已恢复的关键语义包括：

- fresh worker snapshot 提供 `max_prefill_tokens`；
- 下一批 admission credit 按未开始 prefill 的 input tokens 计算，并加可配置 headroom；
- running prefill 不占用下一批 token window；
- router inflight 与 snapshot 通过 workflow/call identity 去重；
- snapshot 缺失或过期时才退化为 request-count safety cap；
- revision watcher 发布完整 global plan，并支持安全的 waiting-request priority update。

本地 Python regression 为 149 passed。A100 Pod 上 `hexgent_token_credit` 两项 Rust 测试和
全部 35 项 `hexgent_` Rust 测试通过。Pod 编译时另修复了 token-credit map 缺少显式
`HashMap<HexgentAdmissionKey, usize>` 类型导致的推断错误。

## 3. 固定 workload 审计

权威输入来自：

```text
real_trace_profiles/H100_REAL_TRACE_PROFILES_FINAL/profiles/llama/
```

原始 profile 统计为：

| Trace | workflows | calls | replay dependency |
|---|---:|---:|---|
| ShareGPT | 100 | 480 | conversation turns serial |
| BFCL-v3 | 100 | 1213 | calls serial in recorded order |
| HotpotQA/LATS | 20 | 636 | `fanout_group_id` forms width-5 parallel stages |

BFCL 和 HotpotQA 均可原样转换。ShareGPT 中 2 个 workflow 含空 user turn，当前 SGLang
chat validator 无法原样接收。正式 replay 仍保留原始 100 个 workflow，但对这两个异常会话
从首个不可重放 turn 开始截断后缀：`J410gdS_7` 保留 5/6 calls，`J410gdS_38` 保留
1/3 calls。最终为 100 workflows/477 calls；所有保留 calls 的 request、dependency 和
input/output length 均来自原 profile，没有复制 workflow 或补入新会话。

Pod canonical traces 位于：

```text
/workspace/hexgent_e2e/formal_traces/h100_llama70b_profiled_lengths/
  sharegpt_100.jsonl
  bfcl.jsonl
  hotpotqa.jsonl
```

这些文件只复用 workload shape 和固定 token length。H100/Llama-70B 的 latency 不作为
A100/Llama-8B SLO；后者必须通过同模型、同硬件、无排队的 exclusive profiling 重新测量。

## 4. 当前运行状态

A100/Llama-8B BF16 全量 exclusive profiling 已在 GPU 0 串行启动，输出目录为：

```text
/workspace/hexgent_e2e/a100_llama8b_bf16_profile_100_100_20_20260805
```

`run_a100_llama8b_exclusive_profile.sh` 已支持通过 `SHAREGPT_TRACE_PATH`、
`BFCL_TRACE_PATH` 和 `HOTPOTQA_TRACE_PATH` 显式传入 workload；旧 50/50/20 默认值保持兼容。

第一次重启 profile 时发生过一次 OOM。证据表明 OOM 位于新 server 的 weight loading，根因
不是长请求或 `mem_fraction_static`，而是旧 `sglang::scheduler` 子进程在 launch parent
退出后仍占用约 69 GiB。Profiler 现通过 `setsid` 启动独立 server process group，并在
cleanup 中先终止当前 profiler child，再向整个 server group 发送 TERM；有界等待 10 秒后
仍未退出则升级到 SIGKILL。对应回归测试先复现失败，再在修复后通过。

配置审计还发现，正式 4P+4D launcher 使用 `--disable-radix-cache`，而最初的 exclusive
profiler 没有禁用缓存。ShareGPT 后续 turns 因此出现大量 cached tokens，会系统性低估独占
TTFT。Profiler 现默认 `DISABLE_RADIX_CACHE=1`，并保留显式 override；所有缓存版 partial
均已删除。权威 A100/Llama-8B SLO 只允许使用无 radix cache 的最终 run。

## 5. 下一步

1. 等待全量 no-cache exclusive profile 完成并校验所有 rows 无 error、output length 精确匹配。
2. 构建 A100/Llama-8B service profile table 和 per-workflow exclusive SLO。
3. 用至少 128 个不同 ShareGPT workflow 做 3/6/9/12 workflow/s token-aware staircase；
   其中正式 100-workflow profile 集保持独立，额外会话只用于压力校准。
4. 选择能够同时形成 prefill/decode batching、非零 global wait 和有效 planner revision，
   但尚未过载的最低 arrival rate，再进行 baseline/HexAgenT 对比。
