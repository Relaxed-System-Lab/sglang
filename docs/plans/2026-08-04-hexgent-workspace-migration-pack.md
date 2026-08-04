# HexAgenT Workspace Migration Pack

## 1. 文档用途

本文档用于将 HexAgenT 项目迁移到新的 workspace 或交接给不了解历史会话的开发者。它记录当前可信的代码基线、系统架构、调度语义、profile 数据、实验经验、已知问题和下一步工作。

不建议向新 workspace 直接导入完整 conversation history。历史会话包含大量已经被后续实验否定的假设、临时 SSH 地址和过期命令，既占用上下文，也容易让接手者采用错误结论。本文档与列出的当前代码、测试和原始实验归档应作为权威交接材料。

## 2. Repository 与版本基线

- 上游基线：SGLang `v0.5.9`，基线 commit `bbe9c7eeb5`。
- 独立 repository：`Relaxed-System-Lab/sglang`。
- 当前开发分支：`hexgent-v0.5.9-main`。
- 已推送的 HexAgenT commits：
  - `ef024808e1 Add HexAgenT PD scheduling integration`
  - `ed667989a0 Exclude legacy profile-only tests from core branch`
  - `e268d468a5 Add curated H100 Llama and Qwen trace profiles`
- 不要向官方 `sgl-project/sglang` 推送。`upstream` 只用于对照原版代码。

迁移到新 workspace 后首先执行：

```bash
git clone https://github.com/Relaxed-System-Lab/sglang.git
cd sglang
git switch hexgent-v0.5.9-main
git rev-parse HEAD
```

预期至少包含上述三个 commits。当前 workspace 在它们之上还有未提交的 non-blocking admission 修复和实验默认 arrival-rate 修改，迁移前应先提交或生成 patch。

## 3. 系统目标与模块关系

HexAgenT 在 SGLang P-D disaggregated serving 上增加 workflow-aware scheduling。Python planner 负责求解，Rust router 负责执行，worker 保留 SGLang 原有的本地调度与 continuous batching。

```text
DAG-aware replay/client
  -> Rust PD router global admission queue
  -> Python planner sidecar
  -> revisioned P/D worker decision and priorities
  -> selected prefill worker local scheduler
  -> Mooncake KV transfer/bootstrap
  -> selected decode worker local scheduler and continuous batch
  -> stage/completion event
  -> planner workflow state and capacity update
```

### 3.1 DAG-aware client/replay

主要代码：

- `python/sglang/hexgent/workflow_trace.py`
- `python/sglang/hexgent/schemas.py`
- `scripts/hexgent_real_trace_collectors/`

每个 LLM call 至少携带：

- `workflow_id`
- `call_id`
- `workflow_arrival_ms`
- `parent_call_ids`，只包含当前已经 reveal 的依赖边
- `input_tokens`
- `predicted_output_tokens`

线上算法不允许预知未来 call 或完整 DAG。新 call 只有在其依赖满足并实际 reveal 时才进入 planner。固定长度 replay 使用 profile 中的 input/output token length，并按照 `parent_call_ids` 释放并发 sibling calls。

### 3.2 Python planner sidecar

主要代码：

- `python/sglang/hexgent/planner.py`
- `python/sglang/hexgent/scheduler.py`
- `python/sglang/hexgent/sidecar.py`
- `python/sglang/hexgent/snapshots.py`
- `python/sglang/hexgent/latency_estimator.py`
- `python/sglang/hexgent/hardware.py`

Planner 输入包括 call metadata、P/D worker 集合、模型与硬件参数、worker snapshot、当前 reservations 和已 reveal 的 workflow state。输出包括 P/D worker、stage priority、decode reservation、revision 和预计输出长度。

Sidecar 目前提供 `/health`、`/schedule_request`、`/plan`、`/enqueue_admission`、`/admission_status`、`/wait_admission` 和 `/event`。

`AdmissionCoordinator` 将短时间内到达的 admission 合并成一个 batch，并在后台线程中求解，避免每个 HTTP 请求同步运行 solver。新的 arrival、completion、cancellation 和 stage event 唤醒 coordinator。

### 3.3 Rust PD router

主要代码：

- `sgl-model-gateway/src/routers/http/pd_router.rs`
- `sgl-model-gateway/src/routers/http/pd_types.rs`
- `sgl-model-gateway/src/config/types.rs`
- `sgl-model-gateway/bindings/python/src/sglang_router/router_args.py`

Router 负责注册 worker、选择 fallback P/D pair、维护 global admission、调用 planner、验证 worker、在 dispatch 时注入 priority/revision/bootstrap metadata，并在 prefill 和完整 call 完成时上报事件。

启动开关为 `--hexgent-router`、`--hexgent-planner-url` 和 `--hexgent-admission-queue`。当前生产路径不是旧的 Python `HexgentLoadBalancer`。

### 3.4 Worker 与 local scheduler

主要代码：

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/disaggregation/mooncake/`

Worker 通过 `/hexgent/pd_snapshot` 暴露 waiting/running、token capacity、transfer 和 stage 状态。请求 dispatch 后仍由 SGLang local scheduler 形成 prefill batch 和 decode continuous batch。

Global queue 不应取代 worker scheduler：global queue 决定尚未 bootstrap 的请求何时、去哪个 worker；local scheduler 管理已经 dispatch 的请求、实际 GPU batch 和 decode 生命周期。

## 4. Workflow horizon、SLO 与 priority

### 4.1 Horizon 与实验 SLO

Planner 的 `workflow_horizon_ms` 由当前已经 reveal 的 DAG 构建，不使用 client 提供的固定 60 秒默认值。每个 call 的理想 service 由同模型、同 GPU、同 TP/精度的 profile 或 roofline estimator 得到。

新 call reveal 时，planner 将它加入 DAG，并把 horizon 更新为当前 revealed graph 中最长 root-to-node 理想 service path。call 完成时，实际 measured service 更新对应 node；`workflow_attained_service_ms` 使用已完成 DAG 的 critical path，而不是并行分支求和。

实验中使用的 workflow SLO 也必须是同模型/同硬件独占 profile 的 DAG critical path：

```text
workflow_slo_ms = max(root-to-leaf sum of exclusive per-call service)
workflow_slo_ratio = observed workflow wall latency / workflow_slo_ms
```

不能用 Llama-70B profile 作为 Llama-8B 的 SLO，也不能把 fan-out 所有 branch 串行求和。

### 4.2 通用 projected-SLO-ratio

通用分支计算候选 assignment 的预计完成时间，并按较大的 projected SLO ratio 优先：

```text
projected_ratio = projected_workflow_completion_ms / workflow_horizon_ms
```

同一个 call 的 placement 会比较各个 P/D worker 的预计可用时间、prefill、transfer、decode 和 decode capacity。高 projected ratio 代表该 workflow 更接近或已经超过目标 horizon。

### 4.3 “短串行”特殊分支

当前代码对一个同时满足以下条件的 batch 使用 `call_laxity`，再用 `reveal_sequence` 和 `call_id` tie-break：

- 最大 input tokens 不超过 256；
- 最大 output tokens 不超过 96；
- `child_count <= 1`；
- 最大 ideal service 不超过 6500 ms。

该分支不是 projected-SLO-ratio 的同义表达。它来自早期针对短、近似均匀、串行 workload 的经验优化，目的是优先最紧迫 deadline 并维持 reveal order。大部分真实 ShareGPT 请求并不满足 256/96 阈值，因此通常仍进入 projected-ratio 分支。

后续不要默认该分支正确。应在真实 50-workflow workload 上做一次 ablation：保留当前规则、全部统一为 projected ratio、以及 `projected ratio + reveal order tie-break`。在没有数据证明前，不扩大该分支适用范围。

### 4.4 Fan-out 分支

已经 reveal 的 sibling calls 通过相同 `parent_call_ids` 识别，不依赖预知未来 `child_count`。Fan-out batch 首先比较：

```text
critical_path_pressure = critical_path_ms / workflow_horizon_ms
```

随后比较 projected ratio、critical-path service、child count 和 service tie-break。目标是优先可能延长当前 workflow critical path 的 branch，同时尽量保持串行 workflow 原有行为。

## 5. Global admission 与 local queue

### 5.1 当前实现

Router 中每个未 bootstrap 请求保存：

- native PD fallback decision；
- 可选 planner decision；
- arrival order；
- fallback eligibility；
- 当前 P worker admission credit。

`max_inflight_per_prefill` 限制 router 已经 dispatch、但 prefill 尚未完成的请求数量。Prefill worker 返回成功后释放 credit；decode 可继续在 local queue/running batch 中执行。

当前 planner 只对 admission window 内选中的请求发布完整 schedule。未选中的请求保持 pending。已经发布 `dispatch_ready`、但尚未得到 router credit 的请求不会自动参与后续完整重求解，因此当前实现还不是“每个 pending 请求都拥有可持续覆盖的 plan table”。

### 5.2 2026-08-04 non-blocking 修复

历史代码仅在 fallback P worker 完全 idle 时允许立即使用 fallback。worker 已有 inflight prefill 时，即使仍有 admission credit，请求也会等待 planner decision。这违背“planner 不得阻塞 serving”。

修复后的原则是：

- 每个请求从到达起始终拥有 native fallback plan；
- 只要 fallback 目标 worker 有 admission credit，请求就可以 dispatch；
- 请求仍在 global queue 时，更新的 planner decision 可以覆盖 fallback；
- admission credit 已满时，请求自然留在 global queue，planner 有时间更新 placement 和 priority；
- dispatch/bootstrap 后 worker pair 不再跨 worker 修改。

对应 Rust 单元测试：`test_hexgent_global_admission_does_not_wait_for_planner_when_credit_exists`。

### 5.3 推荐的最终两级队列设计

推荐保留大部分未 dispatch 请求在 router global queue，而不是把它们提前推入 worker local queue：

1. Arrival 时 router 计算 fallback pair，并将请求放入 global queue。
2. Planner 在 arrival、prefill completion、request completion、cancellation、worker change 等事件上，以旧 plan 作为 warm start 求解所有当前 pending calls。
3. Sidecar 原子发布一整张 revisioned execution plan table，而不是只返回单个 request 的一次性 schedule。
4. Router 对每个 pending key 保存最新有效 revision、P/D pair 和 global priority。
5. 当 P worker 出现 admission credit 时，router 从当前 plan 中选择最优 eligible request；没有 planner entry 时使用 fallback FCFS/priority。
6. Router 在 dispatch 前最后一次验证 worker health、revision 和 credit，然后创建 bootstrap，并原子地把 request 状态改为 `dispatching`。
7. Dispatch 后不再跨 worker migration。Worker local priority仍用于小量已 dispatch waiting 和 decode continuous batch。

这种方案优于“请求先进入 local queue、再跨 worker reassign”：后者需要撤销或重建 bootstrap room，处理 Mooncake metadata、重复 dispatch、worker race、stream cancellation 和局部 KV 状态。对尚未 prefill 的请求虽然理论上可迁移，但实现与验证成本较高，而且收益可以通过浅 local prefill queue 和 dispatch-time selection 获得。

需要处理的 edge cases：

- planner 无响应或 revision 落后：使用 fallback，不等待；
- planner 选择的 worker 下线：重新使用当前 registry 计算 fallback；
- plan publish 与 dispatch 竞争：在 router 内使用单次原子状态转移；
- client cancellation：从 global queue 和 planner reservations 同时删除；
- prefill HTTP 失败：释放 P credit，并上报 cancellation；
- streaming 未收到正常结束：释放未完成 reservations；
- planner 长时间偏爱某些 calls：保留 arrival-order aging/fairness guard；
- local queue 已接收请求：不再允许 global plan 修改 P/D pair。

## 6. Profile 与 workload inventory

### 6.1 已推送的 H100 profiles

`real_trace_profiles/h100_real_trace_profiles_final.tar.gz` 已提交到 repository，包含：

| Model/profile | ShareGPT | BFCL | HotpotQA |
|---|---:|---:|---:|
| Llama-70B H100 TP4 | 100 workflows / 480 calls | 100 / 1213 | 20 / 636 |
| Qwen-235B H100 TP4 FP8 | 100 / 480 | 100 / 1104 | 20 / 550 |

这些 profiles 用于保留真实 trace、长度和相应模型的 H100 timing。不要把其中 Llama-70B timing 作为 Llama-8B SLO。

### 6.2 Llama-8B profiles

当前本地保留：

- `real_trace_profiles/llama8b_h100_2p2d_exclusive_profile_20260802.jsonl`
- `real_trace_profiles/e2e/llama8b_4p4d_nvlink_latencycache_ar8_2_0p6_20260726.tar.gz`
- `real_trace_profiles/e2e/hexgent_h100_4p4d_fanout_parentpath_4p4d_20260803_r2.tar.gz`
- `real_trace_profiles/llama8b_h100_4p4d_formal/model_matched_slo_20260722/`

`llama8b_h100_2p2d_exclusive_profile_20260802.jsonl` 有 456 rows、30 workflow IDs，即每个 trace 10 个不同 workflow。Fan-out archive 内的 `exclusive_profile.jsonl` 使用 `hexgent-llama8b-fp8` 和 `H100_80G_TP1_LLAMA8B_FP8`。

这些 Llama-8B artifacts 当前大部分被 `.gitignore` 排除，没有随 repository push。迁移 workspace 前必须单独复制，或选择一个可信 archive 明确加入 Git/LFS/object storage。

### 6.3 “50 workflow”压力归档的真实性

`llama8b_4p4d_nvlink_latencycache_ar8_2_0p6_20260726.tar.gz` 中每个 trace 显示 50 workflow IDs，但它们来自 10 个原始 workflow 各复制 5 次，例如 `hotpotqa_0__replica_0` 到 `__replica_4`。它们不是 50 个不同内容的 workflow，可能放大 prefix similarity，并降低 workload heterogeneity。

结论：目前没有三个 trace 各 50 个不同且完成 Llama-8B 独占 profiling 的 workflow。

下一轮应采用：

- ShareGPT：从已有 100 个真实 workflow 中选 50 个，并补做 Llama-8B exclusive timing；
- BFCL：从已有 100 个真实 workflow 中选相同 50 个，并补做 Llama-8B exclusive timing；
- HotpotQA：当前先用全部 20 个不同 workflow；如果实验必须使用 50 个，先额外采集 30 个，不使用 replica 伪装多样性。

### 6.4 Arrival rate

正式实验从以下上限开始：

- ShareGPT：3 workflows/s；
- BFCL：0.6 workflows/s；
- HotpotQA：0.2 workflows/s。

只有当 GPU utilization、worker snapshot 和 queue telemetry 都显示没有 queuing pressure 时才逐步增加。每次增加 arrival rate 都必须保存实际 queue depth、running batch、GPU utilization、planner overhead 和 transfer telemetry，不能只根据请求是否完成判断负载是否足够。

`run_h100_formal_comparison.sh` 的默认值已调整为上述三组 rate。

## 7. Serving configuration 与测量经验

### 7.1 H100 4P+4D Llama-8B

当前 launcher 为 `scripts/hexgent_real_trace_collectors/start_hexgent_pd_smoke_4p4d.sh`：

- GPU 0--3：四个 TP1 prefill workers；
- GPU 4--7：四个 TP1 decode workers；
- prefill `max_running_requests=32`；
- prefill batch request cap 32；
- decode `max_running_requests=128`；
- transfer inflight cap 默认每 decode worker 1；
- global admission credit 默认与 prefill max-running 对齐。

不要再把 P/D `max_running_requests` 都设为 4。Decode 必须保留足够大的 running pool 形成 continuous batch。Admission credit 也不能小到让 32-request prefill batch 永远无法形成。

### 7.2 Worker snapshots

Snapshot 是 planner 的实时 capacity 输入，但它是近似模型而不是 GPU simulator。需要观察：

- P waiting/inflight 和 token/request batch limits；
- D waiting/running request 数和剩余 output tokens；
- decode capacity token intervals；
- transfer queue/inflight count；
- snapshot timestamp 和 staleness。

Snapshot RPC 不应处于请求 dispatch 的同步关键路径。Router 已有缓存与后台 refresh 逻辑；新实现应使用最新可用 snapshot，过期时异步刷新，而不是让 worker idle 等 HTTP snapshot。

### 7.3 KV transfer

必须区分：

1. CUDA/NVLink payload copy：CUDA event 覆盖实际 device-to-device copy；
2. Mooncake transfer-engine queueing；
3. KV block 注册/查找和 producer/consumer metadata；
4. producer-ready/decode-ready 通知；
5. request-level transfer pipeline wall time。

历史的 `7 GB/s + 10 ms` 是 request-level pipeline 的近似，不是 NVLink raw bandwidth。Launcher 中仍保留这个兼容默认值，属于待替换模型。不能用它声称 NVLink 只有 7 GB/s，也不能只用 CUDA copy 的亚毫秒结果忽略 producer-ready 尾延迟。

新模型应分别保存 raw payload bandwidth、固定 coordination overhead 和由 snapshot/telemetry 推断的 contention delay。

### 7.4 MFU 与 GPU utilization

`nvidia-smi` GPU utilization 表示采样窗口内 GPU 是否有 kernel 工作，不等于模型浮点峰值利用率。Decode 可以出现 90% 以上 GPU utilization，同时由于 memory-bound 而只有很低的 roofline compute-MFU proxy。

严格 MFU 需要实际 FLOP/tensor-core counter，例如 Nsight Compute 或 CUPTI。Roofline MFU proxy 只能用于判断 workload 更接近 compute-bound 还是 memory-bound，不能当作硬件 counter 报告。

## 8. 已知实验结论与使用限制

- 早期低负载或三次重复报告不能直接与后续 stress run 混合，因为模型匹配 SLO、arrival rate、global admission、transfer backend 和 queue cap 曾发生变化。
- 2026-08-03 4P+4D 单次、10-workflow run 中，BFCL 的 HexAgenT P95 ratio 优于 baseline；ShareGPT 和 HotpotQA 更差。该运行顺序固定为 baseline 后 HexAgenT，不能作为最终统计结论。
- 该 run 已证明 HexAgenT 与 baseline 选择了不同 worker pair，因此问题不只是“两个 scheduler 实际走了相同路由”。
- 同构 cluster 上，placement 差异本身收益有限；真正的 tail 改善主要依赖 global ordering、critical-path priority 和对实时 queue/capacity 的准确估计。
- 复制的 10x5 stress trace 多样性不足。下一次性能结论必须使用不同 workflows。

## 9. Legacy 与不应继续使用的路径

以下本地文件属于旧 Python router/simulator 路径，当前生产 serving 不引用：

- `sgl-model-gateway/bindings/python/src/sglang_router/hexgent_lb.py`
- `sgl-model-gateway/bindings/python/src/sglang_router/hexgent_vendor/`
- 对应 `test_hexgent_lb.py` 和 `test_po2c_*` tests
- `scripts/hexgent_pod_profile/`

当前生产路径是 Rust PD router 加 `python/sglang/hexgent` sidecar。旧文件可以单独归档后删除，但不要误删真实 trace collectors。`docs/get_started/hexgent_v0_5_9_dev_docker.md` 中关于 `hexgent_vendor` 的描述已经过期，需要在后续 cleanup commit 修正。

## 10. 当前验证状态

本地可执行 Python 回归：

```text
81 passed
```

覆盖 planner、scheduler、trace tools 和 scheduling summary。

Rust 已增加 non-blocking admission 单元测试，但当前 Windows 环境缺少 MSVC `link.exe`，`cargo test` 在依赖链接阶段停止，尚未执行到测试本身。必须在 Linux/RunPod 或安装 Visual C++ Build Tools 后运行：

```bash
cargo test --manifest-path sgl-model-gateway/Cargo.toml \
  test_hexgent_global_admission_does_not_wait_for_planner_when_credit_exists \
  --lib --no-default-features
```

随后运行完整 router tests 和真实 1P+1D smoke test。

## 11. 下一阶段实现计划

### Task 1：提交并远程验证 non-blocking fallback

1. 在 Linux/RunPod 运行新增 Rust 测试，先确认旧逻辑下失败、修复后通过。
2. 运行 `cargo fmt --all -- --check` 和相关 router tests。
3. 启动 1P+1D，暂时停止 planner sidecar，确认有 admission credit 时请求仍通过 native fallback 完成。
4. 在 prefill credit 已满时恢复 planner，确认 pending request 可以在 dispatch 前被新 plan 覆盖。

### Task 2：实现完整 revisioned execution plan table

1. 在 sidecar 测试中定义一次 batch solve 为所有 pending keys 发布同一 revision。
2. 扩展 API，使 router 一次获取或订阅完整 plan revision，而不是每个 key 结束一次 long-poll 后退出。
3. Router 为 pending request 保存 fallback、latest planner decision、revision 和 global rank。
4. Plan publication 原子更新所有仍未 dispatch 的 entries。
5. Prefill credit event 触发 dispatch，不等待 planner HTTP；dispatch 前读取最新 revision。
6. 添加 stale worker、cancel、publish/dispatch race 和 starvation tests。

### Task 3：50-workflow profile 与 trace 重建

1. ShareGPT/BFCL 选择 50 个不同 workflows，禁止 replica。
2. HotpotQA 先使用 20 个不同 workflows，或补采到 50。
3. 使用 Llama-8B、相同 H100 拓扑建立 exclusive per-call profile。
4. 检查同一个 workflow 在 baseline/HexAgenT 中的 DAG、input/output lengths 和 arrival seed 完全一致。

### Task 4：压力与 ablation

1. 从 3/0.6/0.2 workflows/s 运行 baseline 与 HexAgenT。
2. 若没有 queue pressure，再按固定倍数增加 arrival rate。
3. 比较 P95/P99 workflow latency 和 SLO ratio。
4. 对短串行 priority 分支做三组 ablation。
5. 保存 worker selection、global rank、planner revision、dispatch reason、fallback/override、snapshot、GPU 和 transfer telemetry。

## 12. 新 workspace 接手检查表

- 确认 branch 和 commit 基线正确。
- 复制或上传被 Git ignore 的 Llama-8B exclusive profile 和 E2E archives。
- 不复制 SSH command、HF token 或其他 credentials 到 repository。
- 运行 `setup_hexgent_e2e_pod_env.sh` 和环境 preflight，再启动模型。
- 确认模型、GPU、TP、precision 与 exclusive profile 完全匹配。
- 确认 Mooncake 日志中的真实 transport，不根据 GPU topology 猜测。
- 确认 P/D max-running、prefill batch cap、decode continuous batch 和 transfer inflight cap。
- 先运行 Rust/Python tests，再运行 1P+1D smoke，最后运行 4P+4D comparison。
- 所有性能结论必须附带原始 calls、summary、planner telemetry、worker logs、GPU telemetry 和精确配置。
