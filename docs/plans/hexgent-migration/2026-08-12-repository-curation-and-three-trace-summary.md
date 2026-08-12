# 2026-08-12 Repository Curation and Three-Trace Summary

## Scope

本轮将可复现入口、A100 三 trace 正式结果和保留资产统一到当前仓库，并删除已经不属于
生产 serving 路径的原型代码、错误实验文档和中间结果。仓库基于 SGLang `v0.5.9`，当前
checkout 为 `v0.5.9-4-g6c6ec06fb3` 加 HexAgenT 改动。

## Authoritative Documents

- `docs/hexgent/README.md`：当前架构、环境准备和三 trace 复现入口。
- `docs/hexgent/2026-08-12-a100-three-trace-results.md`：A100 三 trace 的 GPU utilization、
  KV transfer 和 baseline/HexAgenT 对比。
- `docs/plans/2026-07-30-mooncake-kv-transfer-producer-ready-contention.md`：传输测量边界。
- `real_trace_profiles/README.md`：保留 profile/result archive 及 SHA-256。

## Retained Evidence

- H100 TP4 Llama-70B/Qwen-235B 三 trace profile，包含固定 input/output length、TTFT、
  total latency、response records 和 service tables。
- H100 TP1 Llama-8B model-matched exclusive profile。
- A100 TP1 Llama-8B BF16 三 trace exclusive profile、正式 ShareGPT/BFCL A/B、NVLink
  calibration 和最终 HotpotQA 两轮 paired A/B。
- `python/sglang/hexgent/hardware.py` 及 planner/service-profiler 中的 model/hardware
  参数与 roofline 路径。

## Removed Material

- Python `HexgentLoadBalancer` 和 `hexgent_vendor` simulator prototype。生产路径已经是 Rust
  PD router 加 Python planner sidecar，旧实现无运行时引用。
- 使用错误 SLO、revision-0 fallback、错误 serving config、未完成模式或历史 imputation
  生成的结果和报告。
- 缺失真实 decode timing 时进行插补/修复的 Qwen profiling scripts。正式 service profile
  只接受同一成功请求实测的 input/output length、TTFT 和 total latency。
- 被当前 `docs/hexgent/README.md` 取代的旧 setup/implementation 文档。

## HotpotQA Metric Clarification

早期 `2.1740x` 是 `0.17 workflow/s` 的一次 HexAgenT-only 消融，baseline `2.2130x` 和
`2.3290x` 来自其他 invocation。它缺少 paired mode、逐模式 warmup 和 runtime namespace，
不能计算正式提升。最终有效实验使用 `0.19 workflow/s`、两种 mode 顺序各一轮：逐轮 P95
均值改善约 `0.58%`；合并 40 个 workflow-runs 后，nearest-rank P95 改善 `1.11%`，归档
aggregate 使用线性插值时为 `1.27%`。这解释了调试记录与最终报告幅度不同的原因。

## Validation Requirements

在发布或迁移前应验证：归档 SHA-256、零 replay errors、非零 planner revision coverage、
Python regression tests、Rust `cargo check`，并确认 README 中没有指向已删除文档或旧 Python
router 的活动链接。
