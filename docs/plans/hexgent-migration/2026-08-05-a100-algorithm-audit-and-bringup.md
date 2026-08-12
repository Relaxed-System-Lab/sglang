# 2026-08-05 A100 算法审计与环境部署

## 1. 本轮目标

- 核对当前 HexAgenT planner 是否实现分阶段的 projected-SLO-ratio 调度。
- 在新的 8x A100-SXM4-80GB 节点恢复当前权威代码和实验资产。
- 验证 A100 节点的 GPU 互联，并为 A100 独占 profile 与 4P+4D 实验准备可复现环境。

## 2. Planner 语义审计

当前实现与“分别维护 `H_w^P/C_w^P` 和 `H_w^D/C_w^D`”的概括并不完全一致。

代码确认的当前语义如下：

- `scheduler.py` 的 call records 只有一组 `workflow_horizon_ms` 和
  `workflow_completion_ms`。Planner 的 workflow runtime state 同样只有一组 horizon 和
  completion，没有四个 stage-specific 状态。
- Prefill 与 decode 分别通过 `schedule_prefill` 和 `schedule_decode` 求解，并分别计算
  projected score；但是两个 score 的分母和已完成时间均来自同一组 workflow-level
  horizon/completion。
- Prefill projected finish 使用候选 prefill、KV transfer 和 decode 的联合预计完成时间，
  并在请求到达时同时确定 prefill/decode placement。
- Decode projected finish 使用目标 decode instance 的容量区间和预计 decode service time。
  已锁定 decode instance 的 call 只能在该 instance 内重排；尚未锁定的 call 可以跨实例选择。
- 普通分支和 fan-out 分支现在都按 projected ratio 从高到低处理。原先 fan-out 独有的
  `critical_path_ms / workflow_horizon_ms` 首要排序已移除，避免在 horizon/completion 已由 revealed
  DAG critical path 计算的基础上重复加权 critical path。短串行与 decode-dominated 特殊分支仍使用
  laxity/reveal order。
- 已加入低求解压力下的 bounded pairwise-swap local search。它对普通串行和 fan-out 都生效，
  只要候选 call 不超过 8 个；最多评估 28 个 pair，只应用一次 best-improving swap。目标按
  `(max workflow projected ratio, mean workflow projected ratio, total projected elapsed)` 字典序比较。
  超过阈值的高压力 batch 不执行 local search。

因此，当前实现是“prefill/decode 分阶段求解、共享 workflow horizon/completion、prefill
联合预测 P+transfer+D、decode 使用容量约束、低压力候选集使用 bounded swap”的算法，
而不是严格的 stage-specific `R_P`/`R_D` 双 horizon 版本。后续性能报告必须使用这一真实语义。

验证命令：

```text
python -m pytest -q \
  test/unit/utils/test_hexgent_scheduler.py \
  test/unit/utils/test_hexgent_planner.py
```

最新结果：`67 passed`，其中 scheduler 单测 `14 passed`。

## 3. A100 节点与直接 P2P 测量

直接观测：

- 8x NVIDIA A100-SXM4-80GB。
- 任意 GPU 对在 `nvidia-smi topo -m` 中均显示 `NV12`。
- 每张 GPU 的 `nvidia-smi nvlink --status` 显示 12 条 25 GB/s link。

使用 CUDA event 包围 `cudaMemcpyPeerAsync`，并在目标 GPU 对输出 buffer 做完整校验后：

| GPU pair | Size | Median | Effective bandwidth |
|---|---:|---:|---:|
| 0 -> 1 | 64 MiB | 0.245 ms | 255.0 GiB/s |
| 0 -> 1 | 256 MiB | 0.959 ms | 260.7 GiB/s |
| 0 -> 1 | 1024 MiB | 3.815 ms | 262.1 GiB/s |
| 0 -> 4 | 64 MiB | 0.244 ms | 256.2 GiB/s |
| 0 -> 4 | 256 MiB | 0.958 ms | 261.1 GiB/s |
| 0 -> 4 | 1024 MiB | 3.815 ms | 262.1 GiB/s |

以上是已校验的纯 CUDA P2P copy，不包含 Mooncake metadata、buffer registration、队列等待和
producer/consumer 协调，不能直接作为 production KV pipeline latency。

## 4. Mooncake native 测量状态

Mooncake v0.3.9 已应用以下本地补丁并成功编译：

- intra-node NVLink transport priority；
- CUDA peer access 与 caller device 恢复；
- native CUDA-event copy telemetry。

第一次 native driver 运行无有效结果。根因是原 build cache 中
`USE_INTRA_NVLINK:BOOL=OFF`；日志明确报告 `nvlink_intra` 需要
`-DUSE_INTRA_NVLINK=ON`，target 随后退出。该失败运行不得用于带宽或延迟结论。

已将 clean build 配置为 `USE_CUDA:BOOL=ON`、`USE_INTRA_NVLINK:BOOL=ON`，并显式使用
Python 3.11 headers/runtime 重新构建。第一次只启用 `USE_INTRA_NVLINK` 的 clean build 因
CUDA headers 未加入 target 而失败；随后同时启用 `USE_CUDA` 后构建成功。

最终 250 MiB native 测量同时满足：

- target 对 250 MiB buffer 的分布式 samples 校验成功；
- Python 记录同步 API wall-clock；
- native 日志出现 `[HEXGENT_CUDA_COPY]` 和 `[HEXGENT_CUDA_COPY_BATCH]` CUDA-event
  telemetry；
- 初始化日志明确选择 `Intra-Node NVLink transport`；
- initiator 与 target 均返回 0。

测量将 250 MiB 分成 50 个 5 MiB copy。20 次正式结果中，native CUDA-event batch copy
sum 中位数为 1.293ms、P95 为 1.302ms（范围 1.284--1.303ms），对应约
187--190GiB/s；Python `batch_transfer_sync_write` API
wall-clock 中位数为 4.292ms、P95 为 4.355ms，对应 56.88GiB/s。直接单块
`cudaMemcpyPeerAsync` 的 256 MiB 测量约 0.959ms，因此分块 Mooncake copy 本身增加了
约 0.3ms，而 API 边界还包含约 3ms 的提交、查找、同步和完成确认开销。

这些数字是在空载、单 initiator/target、单 transfer 下测得的，不包含 serving 中的 transfer
queue contention，也不能用于替代 producer-ready/request-level pipeline telemetry。

## 5. 环境问题与修复

本轮发现并修复了两个 RunPod network-volume 特有问题：

- 在 `/workspace` 的 NFS volume 内安装 venv 会因大量小文件写入而长期阻塞。runtime venv
  已改为 container disk 上的 `/opt/hexgent_sglang_venv`，pip cache 和实验输出仍保存在
  `/workspace`。
- Rust release target 放在 NFS volume 会产生 LLVM output-stream I/O failure。
  `CARGO_TARGET_DIR` 已改为 `/opt/hexgent-router-target`。

同时补充了 `libgtest-dev`，并让生成的 `env.sh` 显式将 venv 和 Cargo bin 加入 `PATH`。
对应 setup tests 已先失败、再随修复通过。当前 Python、SGLang、SGL kernel、Mooncake、
HexAgenT sidecar 和 Rust router extension 均通过 import preflight。

Pod 上 scheduler/planner/collector focused tests 为 `92 passed`（本地同一集合为
`93 passed`，平台相关 collection 数量不同）。Rust release extension 已成功构建并导入；随后
单独运行 target `cargo test` 时，Cargo 进程在读取 network-volume 上的 source/registry 时进入
kernel `request_wait_answer`。这不是 assertion 或 compilation failure。只把
`CARGO_TARGET_DIR` 移到 `/opt` 不足以保证 test build 不受 NFS 影响；后续 Rust test 应将测试用
source 和 Cargo home 也复制到 container disk，再从 `/opt` 执行。

## 6. 持久化资产与待办

已保存在 `/workspace` 的可复用资产包括当前权威源码、50/50/20 formal traces、Llama-8B
H100 profiles、A100 CUDA P2P 测量、Mooncake patched source/build 和环境 manifest。

Hugging Face 已认证为授权用户。A100 BF16 的 `meta-llama/Llama-3.1-8B-Instruct` 权重已完整
下载到持久化 `/workspace/hub`，snapshot revision 为
`0e9e39f249a16976918f6564b8830bc894c89659`。

新增可复现入口：

```text
scripts/hexgent_real_trace_collectors/run_a100_llama8b_exclusive_profile.sh
```

该脚本在单张 A100 上以 TP1/BF16、CUDA graph enabled、`max_running_requests=256` 启动
Llama-3.1-8B-Instruct，并严格串行 profile：

- 50 ShareGPT workflows，227 calls；
- 50 BFCL workflows，487 calls；
- 20 HotpotQA workflows，636 calls。

总计 1,350 calls。所有请求固定 replay trace 中的 input/output length；每个 trace 先执行 3 个
warmup calls。结果持久化至：

```text
/workspace/hexgent_e2e/a100_llama8b_bf16_profile_50_50_20_20260805
```

Profiling driver 已正常退出（status 0），三个 trace 的 raw/prefill/decode 表均通过 identity、顺序、
行数与 request-level timing 校验：ShareGPT 227 行、BFCL 487 行、HotpotQA 636 行，共 1,350 行，
error、非正 TTFT、`total_latency_ms < ttft_ms`、非正 output 和重复 call id 均为 0。

请求级 decode service time 的权威定义为 `total_latency_ms - ttft_ms`。可见 content delta 的
inter-token 列表只用于诊断：完整率分别为 ShareGPT 86.3%、BFCL 45.4%、HotpotQA 96.7%。
BFCL 的 tool-call/结构化 token 不一定产生可见 content delta，此外旧 profiler 对跨网络 chunk 的
SSE event 未保留 parser buffer。后者已用失败测试复现并改为 stateful parser；该问题不影响已经
记录的请求起止 wall-clock、TTFT 或上述 decode service time，因此无需重跑 1,350 个请求。

权威摘要保存在结果目录的 `validation_summary.json`。`exclusive_profile.jsonl` 的 SHA-256 为
`97d0c6e8325dc711d77a210b43aa33ce9960bd501a1c438c098ab6370cc4e38b`。归档为：

```text
/workspace/hexgent_e2e/a100_llama8b_bf16_profile_50_50_20_20260805.tar.gz
```

归档 SHA-256 为
`6f4c22d26cd9a833c320d0b33b756d31c2352e577fda01043c3385835835b5cc`。归档还包含合并后的
`prefill_all.jsonl`、`decode_all.jsonl` 和 `latency_db.json`。后者的请求级拟合仅作为 service
profile table；正式 planner 仍应以 model/hardware roofline 和实时 snapshot 为主，不能把拟合中的
trace correlation 当成物理因果关系。

## 7. A100 1P+1D 在线 smoke

使用 GPU0 prefill、GPU1 decode、BF16 Llama-8B 和 A100 Mooncake overlay 完成真实 P-D
disaggregated serving。两侧日志均明确选择 `Intra-Node NVLink transport`，snapshot endpoint 返回
正确的 prefill/decode limits 与空闲队列。

单个 ShareGPT workflow 的 6 个串行 call 全部成功，router mode 均为 `hexgent`。第一笔 KV
transfer 包含约 223ms 的 transport cold-start；后续 47--218MiB 请求的 native sync critical
copy 约 0.35--0.49ms，producer-ready 约 2.3--2.8ms。Decode worker 的 request-level
`transfer_duration` 仍包含等待与 pipeline 协调，不能替代 sender-side CUDA-event copy。

该 smoke 同时发现当前 Rust global admission queue 仍按固定 request count credit 工作，而非本项目
文档描述的 token-aware credit。默认 credit=64 时所有低负载请求立即使用 revision 0 native fallback，
planner admission count 为 0。把 smoke-only credit 临时设为 1，并同时提交 5 个不同 workflow 后，
21 个 call 全部成功，其中 3 个 call 获得非零 revision；planner 最大 batch 为 3 个 admissions，
最大 wall time 4.65ms、最大 CPU time 3.45ms。该结果证明 planner/router/worker 全链路可用，但
credit=1 不是正式性能配置。进入 4P+4D 比较前必须恢复并实测 token-aware admission，避免默认
request credit 让 planner 没有 placement 覆盖机会。

两轮 1P+1D smoke 的 calls、planner、snapshot、GPU 和 worker/router 日志已归档为
`/workspace/hexgent_e2e/a100_1p1d_smoke_20260805.tar.gz`，SHA-256 为
`00ef9cd33c7928aec325c7a92e19743b94f737cf849d6d49cf861c386e95b214`。归档后所有
router/planner/prefill/decode 进程已停止，8 张 GPU 显存占用均回到 0MiB。

后续顺序：

1. 使用 A100 独占 profile 构建 model/GPU-matched SLO；HotpotQA 使用 DAG critical-path 聚合。
2. 完成 1P+1D smoke，再运行 4P+4D baseline/HexAgenT。同一轮必须保持 trace、arrival
   seed、input/output lengths 和 cache 初始状态一致。
3. 初始 arrival rate 使用 3/0.6/0.2 workflows/s；仅在没有 queue pressure、GPU 未充分利用
   且 planner revision 已实际生效时提高负载。
