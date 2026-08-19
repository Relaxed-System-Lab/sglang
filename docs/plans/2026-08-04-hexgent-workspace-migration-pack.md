# HexAgenT Workspace Migration Pack

> 最后更新：2026-08-04。本文已经纳入 revisioned global admission 实现、
> Llama-8B H100 50/50/20 profiling，以及最新 4P+4D 真机对比结果。

详细的逐轮调查证据保存在 `docs/plans/hexgent-migration/`。主文档只保留当前可信结论；
后续每轮工作都必须新增或更新对应 session note，并将仍然有效的结论提炼回本文。

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

预期至少包含上述三个 commits。当前 worktree `hexgent/revisioned-admission-priority`
在它们之上还有约 2200 行未提交改动，包括 revisioned execution plan、
non-blocking fallback、snapshot-aware admission credit、worker priority action、
真实 trace 修复和实验脚本更新。迁移前必须先提交该分支或生成完整 patch；不要只复制
Python planner 而遗漏 Rust router、Python binding 和测试。

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

Sidecar 目前提供 `/health`、`/schedule_request`、`/plan`、`/enqueue_admission`、
`/admission_status`、`/wait_admission`、`/wait_plan_revision` 和 `/event`。

`AdmissionCoordinator` 将短时间内到达的 admission 合并成一个 batch，并在后台线程中求解，避免每个 HTTP 请求同步运行 solver。新的 arrival、completion、cancellation 和 stage event 唤醒 coordinator。每次求解以旧 plan 作为 warm start，并为当前 pending calls 原子发布同一 revision 的完整 plan；新请求在该 revision 发布前仍拥有 native PD fallback，不会阻塞 serving。

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

通用分支在 pending calls 不超过 32 时不是先做一次静态排序再 placement，而是执行动态
greedy recomputation：每轮基于当前虚拟 P/D availability 计算各 call 的最佳 P/D pair 和
projected ratio，选择最紧迫 call，写入该虚拟 reservation，再为剩余 calls 重算。超过该
规模时才先按 priority key 排序，然后依次选择预计最早完成的 P/D pair。每次 assignment
都会推进对应 P/D worker 的虚拟可用时间，因此后一个 call 会看到前面 assignments 的影响。

Planner 输出的 `priority_score` 是该顺序中的 rank，数值越小优先级越高。Router global
admission 按该值选择 eligible request；worker 以
`--enable-priority-scheduling --schedule-low-priority-values-first` 启动，因此仍处于 local
waiting queue 的请求也按 rank 后再按 FCFS 排序。已经进入 decode running batch 的请求
不会被解释成严格串行队列位置；continuous batching 仍同时推进 active requests。

### 4.3 “短串行”特殊分支

当前代码对一个同时满足以下条件的 batch 使用 `call_laxity`，再用 `reveal_sequence` 和 `call_id` tie-break：

- 最大 input tokens 不超过 256；
- 最大 output tokens 不超过 96；
- `child_count <= 1`；
- 最大 ideal service 不超过 6500 ms。

该分支不是 projected-SLO-ratio 的同义表达。它来自早期针对短、近似均匀、串行 workload 的经验优化，目的是优先最紧迫 deadline 并维持 reveal order。大部分真实 ShareGPT 请求并不满足 256/96 阈值，因此通常仍进入 projected-ratio 分支。

后续不要默认该分支正确。应在真实 50-workflow workload 上做一次 ablation：保留当前规则、全部统一为 projected ratio、以及 `projected ratio + reveal order tie-break`。在没有数据证明前，不扩大该分支适用范围。

### 4.4 Fan-out 分支

已经 reveal 的 sibling calls 通过相同 `parent_call_ids` 识别，不依赖预知未来 `child_count`。
Fan-out 与普通路径统一按 projected ratio 从高到低排序。原先独立的
`critical_path_ms / workflow_horizon_ms` 首要排序已禁用，因为 workflow horizon/completion
本身已经由 revealed DAG critical path 计算，再使用该 pressure 会重复加权 critical path。

当任意候选集不超过 8 个时，scheduler 额外评估最多 28 个 pairwise swaps，只应用一次
best-improving swap。目标依次最小化最大 workflow projected ratio、平均 workflow projected
ratio 和总 projected elapsed。该优化同时覆盖串行与 fan-out；超过阈值的高压力 batch 直接跳过。

## 5. Global admission 与 local queue

### 5.1 当前实现

Router 中每个未 bootstrap 请求保存：

- native PD fallback decision；
- 可选 planner decision；
- arrival order；
- fallback eligibility；
- 当前 P worker admission credit。

`max_inflight_per_prefill` 限制 router 已经 dispatch、但 prefill 尚未完成的请求数量。Prefill worker 返回成功后释放 credit；decode 可继续在 local queue/running batch 中执行。

当前 credit 是 request-count credit，不是 token-aware credit。Snapshot 会用
`prefill_max_requests` 或 `max_running_requests` 限制每 worker capacity，并统计
bootstrap/waiting/inflight/running requests；它尚未根据 waiting input tokens 与
`max_prefill_tokens` 计算“填满下一批还缺多少 token”。降低 credit 会让更多请求保持全局
可重排，但过低会减少 prefill 可组成的 token batch，并间接让 decode continuous batch
缺少输入。合理目标不是固定留满 32 个 local requests，而是至少准备足以填满下一次
prefill token budget 的 eligible work，并保留少量 headroom。

2026-08-04 workload 的平均 input tokens 分别约为 ShareGPT 771、BFCL 6658、HotpotQA
1772；在 `max_prefill_tokens=16384` 下，平均约需 21、3、9 个 requests 才能填满一批。
因此相同 credit 对不同 trace 的影响很大：每 worker credit 8 容易使 ShareGPT prefill
underfill，但对 BFCL 已明显充足；credit 16 对 ShareGPT 仍可能不足一整批，对另外两类
通常充足。后续应优先实现或验证 token-aware admission，而不是统一把 credit 降到很小。

当前 sidecar 会为所有仍 pending 的 admission request 发布 revisioned plan table。Router
维护 revision high-water mark，只接受更新的 plan；已经 dispatch/bootstrap 的请求不会被
新 revision 跨 worker 修改。对于暂时没有 planner entry 的新请求，router 保留 native PD
fallback decision 和 arrival order，并在获得 prefill credit 时允许其继续执行。

Router 还支持安全的 `priority_update` action。该 action 只允许修改尚未执行的 worker-local
waiting request；prefill 已开始、KV transfer 已开始或 decode running 后不会跨 worker
迁移，也不会重建 bootstrap room。旧 revision 的 priority action 会被新 revision 覆盖，
失败 action 会在有限次数内重试。

### 5.2 2026-08-04 non-blocking 修复

历史代码仅在 fallback P worker 完全 idle 时允许立即使用 fallback。worker 已有 inflight prefill 时，即使仍有 admission credit，请求也会等待 planner decision。这违背“planner 不得阻塞 serving”。

修复后的原则是：

- 每个请求从到达起始终拥有 native fallback plan；
- 只要 fallback 目标 worker 有 admission credit，请求就可以 dispatch；
- 请求仍在 global queue 时，更新的 planner decision 可以覆盖 fallback；
- admission credit 已满时，请求自然留在 global queue，planner 有时间更新 placement 和 priority；
- dispatch/bootstrap 后 worker pair 不再跨 worker 修改。

对应 Rust 单元测试：`test_hexgent_global_admission_does_not_wait_for_planner_when_credit_exists`。

### 5.3 已实现的两级队列设计

当前实现保留大部分未 dispatch 请求在 router global queue，而不是把它们提前推入 worker local queue：

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

2026-08-04 在 8x H100 RunPod 上新增了一轮模型匹配的独占 profile：

- trace 选择：ShareGPT 50 个、BFCL 50 个、HotpotQA 20 个不同 workflow；
- 总计 1347 个 profile rows：ShareGPT 224、BFCL 487、HotpotQA 636；
- 模型：`neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`；
- worker label：`H100_80G_TP1_LLAMA8B_FP8`；
- Pod 路径：`/workspace/hexgent_e2e/llama8b_profile_50_50_20_20260804/profile/`；
- 实验中使用的合并 SLO profile：`exclusive_profile.jsonl`。

这轮 profile driver 正常结束，但最初选取的 ShareGPT 50 workflows 中有两个 workflow
包含空 user message，导致后续 baseline 和 HexAgenT 各出现相同的 3 个 HTTP 400。
BFCL 和 HotpotQA 不受该数据问题影响。`workflow_trace.py` 已修改为整体排除任何包含空
message content 的 workflow，并已重新生成
`sharegpt_workflows50_valid.jsonl`；新的 50-workflow ShareGPT 独占 profile 尚未完成，
原因是补测时 standalone SGLang server 在 CUDA 初始化前停滞。不要把当前 ShareGPT
结果作为最终无错误结论。

### 6.3 “50 workflow”压力归档的真实性

`llama8b_4p4d_nvlink_latencycache_ar8_2_0p6_20260726.tar.gz` 中每个 trace 显示 50 workflow IDs，但它们来自 10 个原始 workflow 各复制 5 次，例如 `hotpotqa_0__replica_0` 到 `__replica_4`。它们不是 50 个不同内容的 workflow，可能放大 prefix similarity，并降低 workload heterogeneity。

结论更新：BFCL 已有 50 个不同 workflow，HotpotQA 已有全部 20 个不同 workflow；
ShareGPT 已能生成 50 个内容有效且不同的 workflow，但仍需完成一次干净的 Llama-8B
独占 timing。旧的 10x5 replica 压力归档仍不得用于证明 workload 多样性。

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

2026-08-04 真机 snapshot 确认实际生效值为：prefill
`prefill_max_requests=32`、`max_running_requests=32`，decode
`max_running_requests=128`，每个 worker 的 `max_total_num_tokens=380134`。
因此本轮没有再次落入历史上的 `max_running_requests=4` 瓶颈。

Python router CLI 曾缺少已经存在于 Rust config/binding 中的
`--hexgent-prefill-batching-headroom` 参数，导致 worker 全部 ready 后 router 启动失败。
该参数已补入 `sglang_router/router_args.py`，测试通过，并在 Pod 上重新执行
`maturin develop` 构建。随后 4 个 prefill、4 个 decode、HexAgenT router、baseline
router 和 planner 全部通过 health check。

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

本次新 Pod 的 PyPI `mooncake-transfer-engine==0.3.9` wheel 即使显式选择 `tcp`，
初始化时仍尝试安装未编译的 NVLink transport，并报
`Unsupported transport nvlink, please rebuild Mooncake`。持久卷中已有此前 source-built
overlay：`/workspace/third_party/mooncake_engine_overlays/p2p_probe`。使用
`MOONCAKE_INTRANODE_NVLINK=1` 和该 overlay 后，8 个 P/D workers 均成功 ready。
实验记录必须继续保留该 overlay 路径和 Mooncake 日志，不能假设普通 PyPI wheel 已包含
同样能力。

### 7.4 MFU 与 GPU utilization

`nvidia-smi` GPU utilization 表示采样窗口内 GPU 是否有 kernel 工作，不等于模型浮点峰值利用率。Decode 可以出现 90% 以上 GPU utilization，同时由于 memory-bound 而只有很低的 roofline compute-MFU proxy。

严格 MFU 需要实际 FLOP/tensor-core counter，例如 Nsight Compute 或 CUPTI。Roofline MFU proxy 只能用于判断 workload 更接近 compute-bound 还是 memory-bound，不能当作硬件 counter 报告。

### 7.5 RunPod 环境与可复用依赖

本次新 Pod 需要补装以下系统依赖后，SGLang kernel、Mooncake 和 Rust router extension
才能正常加载或编译：

```text
libnuma1
libgoogle-glog0v5
libibverbs1
libjsoncpp25
librdmacm1
rdma-core
pkg-config
libssl-dev
protobuf-compiler
libprotobuf-dev
patchelf
```

Rust toolchain 保存在持久卷 `/workspace/.cargo` 和 `/workspace/.rustup`；环境文件为
`/workspace/hexgent_e2e/rust.env`。Router extension 使用
`scripts/hexgent_real_trace_collectors/rebuild_router_extension.sh` 构建，target 应放在 Pod
本地 `/tmp`，不要放到 network volume，以避免历史上出现的 linker/bus error。

从 Windows 通过 `scp` 上传的 shell/Python 文件可能带 CRLF。Pod 上运行 shell script 前先用
`bash -n` 检查；若出现 `set: pipefail\r: invalid option name`，应只对明确上传的文件执行
CRLF-to-LF 转换。不要递归改写整个 repository。

本次持久卷中的关键模型和 transport 资产：

- Llama-8B FP8 snapshot：`/workspace/hub/models--neuralmagic--Meta-Llama-3.1-8B-Instruct-FP8/snapshots/12fd6884d2585dd4d020373e7f39f74507b31866`；
- Mooncake source-built overlay：`/workspace/third_party/mooncake_engine_overlays/p2p_probe`；
- SGLang source：`/workspace/sglang`；
- Python venv：`/workspace/hexgent_sglang_venv`。

## 8. 已知实验结论与使用限制

- 早期低负载或三次重复报告不能直接与后续 stress run 混合，因为模型匹配 SLO、arrival rate、global admission、transfer backend 和 queue cap 曾发生变化。
- 2026-08-03 4P+4D 单次、10-workflow run 中，BFCL 的 HexAgenT P95 ratio 优于 baseline；ShareGPT 和 HotpotQA 更差。该运行顺序固定为 baseline 后 HexAgenT，不能作为最终统计结论。
- 该 run 已证明 HexAgenT 与 baseline 选择了不同 worker pair，因此问题不只是“两个 scheduler 实际走了相同路由”。
- 同构 cluster 上，placement 差异本身收益有限；真正的 tail 改善主要依赖 global ordering、critical-path priority 和对实时 queue/capacity 的准确估计。
- 复制的 10x5 stress trace 多样性不足。下一次性能结论必须使用不同 workflows。

### 8.1 2026-08-04 4P+4D、50/50/20 真机运行

配置：Llama-8B FP8，8x H100，4 个 TP1 prefill workers、4 个 TP1 decode workers，
Mooncake source-built intra-node overlay；arrival rate 分别为 ShareGPT 3、BFCL 0.6、
HotpotQA 0.2 workflows/s。Baseline 和 HexAgenT 使用相同 DAG、input/output length、
arrival seed。SLO 分母来自同模型、同 H100 的 Llama-8B 独占 profile，并按 DAG critical
path 聚合，绝不能替换为 Llama-70B profile。

| Trace | Calls / workflows | Errors | Baseline P95 SLO ratio | HexAgenT P95 SLO ratio | 当前判断 |
|---|---:|---:|---:|---:|---|
| ShareGPT | 224 / 50 | 两侧各 3 | 1.324x | 1.265x | 原始差值；没有 planner plan 被执行 |
| BFCL | 487 / 50 | 0 | 1.501x | 1.690x | 原始差值；没有 planner plan 被执行 |
| HotpotQA | 636 / 20 | 0 | 1.185x | 1.195x | 原始差值；没有 planner plan 被执行 |

后续审计发现，这一轮不能用于判断 HexAgenT 策略优劣：所有成功的 HexAgenT calls
都携带 `schedule_revision=0`，分别为 ShareGPT 221、BFCL 487、HotpotQA 636 个。
Rust router 中 revision 0 明确定义为 native fallback。对应 planner telemetry 的
`admission_count` 和 `result_count` 在三条 trace 中均始终为 0，说明 admission entry
在 planner coordinator 取走前已经因 prefill credit 可用而 dispatch。

负载也没有形成 global scheduling pressure：请求区间的最大并发分别为 23、8、20；
prefill snapshot 的 waiting 峰值均为 0；decode running 峰值分别为 9、2、4，远低于
4 个 prefill workers 共 128 个 admission credits。因此当前表格仅记录两次 native
fallback runs 的观测差值，不能解释为 projected-SLO-ratio 相对 baseline 的收益或退化。

Baseline router 没有显式指定 P/D policy，实际继承 SGLang router 默认
`cache_aware` policy，而不是纯 FCFS。两侧 worker-local scheduler 相同，本轮 snapshot
又没有观测到 local waiting queue，所以也不能将该实验描述成“HexAgenT 对 FCFS”。
此外实验顺序固定为 baseline 后 HexAgenT，并复用同一组 workers/cache，仍存在 run-order
和 prefix-cache 状态混杂。完整证据见
`docs/plans/hexgent-migration/2026-08-04-4p4d-scheduling-audit.md`。

为防止同类误判，`summarize_scheduling_replay.py` 已新增
`planner_plan_execution`：统计成功 HexAgenT calls 中 revision 0、revision>0 和缺失 revision
的数量、planner revision 覆盖率，以及是否实际观测到 planner effect。对应 summary 单测
为 6 passed。

对应 P95 workflow latency：ShareGPT baseline 10282ms、HexAgenT 9902ms；BFCL
baseline 6495ms、HexAgenT 6283ms；HotpotQA baseline 8924ms、HexAgenT 9131ms。
BFCL 的绝对 P95 latency 较低但 SLO ratio 更差，说明必须以 per-workflow 模型匹配 SLO
比较，不能只看聚合绝对 latency。

Planner 本轮单次 invocation 很轻：ShareGPT P95 0.213ms、BFCL 0.156ms、
HotpotQA 0.138ms；没有再次出现历史上 100ms 级 solver CPU tail。该数据只说明本轮
sidecar solve 本身不重，不代表 snapshot、router HTTP 或 transfer coordination 没有其他
控制面开销。

原始结果：

- Pod：`/workspace/hexgent_e2e/formal_results_20260804/hexgent_h100_4p4d_llama8b_50_50_20_20260804_r1.tar.gz`；
- 当前 worktree：`artifacts/20260804/hexgent_h100_4p4d_llama8b_50_50_20_20260804_r1.tar.gz`。

该 archive 包含 calls、summary、planner invocation、worker snapshot 和 GPU telemetry。
ShareGPT 的三条数据错误必须保留在原始 archive 中用于审计，不要就地修改归档。

## 9. Legacy 与不应继续使用的路径

以下本地文件属于旧 Python router/simulator 路径，当前生产 serving 不引用：

- `sgl-model-gateway/bindings/python/src/sglang_router/hexgent_lb.py`
- `sgl-model-gateway/bindings/python/src/sglang_router/hexgent_vendor/`
- 对应 `test_hexgent_lb.py` 和 `test_po2c_*` tests
- `scripts/hexgent_pod_profile/`

当前生产路径是 Rust PD router 加 `python/sglang/hexgent` sidecar。旧 Python router、
`hexgent_vendor` 和对应过期文档已在 2026-08-12 清理；真实 trace collectors、profile
assets 和 Mooncake measurement tools 均已保留。

## 10. 当前验证状态

在 revisioned admission 大改完成后，本地完整 Python 回归曾达到：

```text
100 passed
```

随后针对本轮新增的 trace validation、Mooncake protocol override 和 Python router CLI
参数又执行了 4 个 focused tests，全部通过。由于最后的 trace validation 修改后尚未再次
运行完整测试集，接手者应重新执行全量 Python tests。

Rust router 在 RunPod 上通过 `cargo check -p sgl-model-gateway --lib`，并成功通过
`maturin develop` 构建安装，真实 4P+4D router 已启动运行。`cargo check --tests` 仍被仓库
中与 HexAgenT 无关的 `tests/wasm_test.rs` API 不兼容阻塞；Windows 本地则缺少 MSVC
`link.exe`。仍应在 Linux 上运行目标单测：

```bash
cargo test --manifest-path sgl-model-gateway/Cargo.toml \
  test_hexgent_global_admission_does_not_wait_for_planner_when_credit_exists \
  --lib --no-default-features
```

随后运行完整 router tests 和真实 1P+1D smoke test。

## 11. 下一阶段实现计划

### Task 1：提交 revisioned admission 实现

1. 整理当前 worktree 的 Rust、Python、binding、tests 和文档为一个独立 commit/PR。
2. 在 Linux/RunPod 运行目标 Rust 单测和 `cargo fmt --all -- --check`。
3. 修复或隔离无关的 `wasm_test.rs` API 不兼容后运行完整 router tests。
4. 保留本次成功构建 extension 与真实 4P+4D health check 的日志。

### Task 2：补强 revisioned execution plan 验证

1. 已实现同 revision 完整 plan、`/wait_plan_revision`、router high-water mark 和 fallback。
2. 增加长时间压力下的 stale worker、cancel、publish/dispatch race 和 starvation 真机测试。
3. 验证 `priority_update` 只修改安全 waiting state，并记录成功、重试和 superseded 计数。
4. 从 calls/telemetry 中直接核对 fallback plan 被新 revision 覆盖的比例和 dispatch reason。

### Task 3：50-workflow profile 与 trace 重建

1. BFCL 50 和 HotpotQA 20 已完成；保留现有独占 profile。
2. 使用 `sharegpt_workflows50_valid.jsonl` 完成无空消息的 Llama-8B 独占 profile。
3. 只补跑 ShareGPT baseline/HexAgenT，并替换暂定 summary，不修改原始 archive。
4. 检查同一个 workflow 在 baseline/HexAgenT 中的 DAG、input/output lengths 和 arrival seed 完全一致。

### Task 4：压力与 ablation

1. 从 3/0.6/0.2 workflows/s 运行 baseline 与 HexAgenT。
2. 若 `schedule_revision>0` 覆盖率为 0 或没有 queue pressure，按固定倍数增加 arrival rate；
   在 planner plan 实际被执行前，不发布 scheduler 性能结论。
3. 比较 P95/P99 workflow latency 和 SLO ratio。
4. 对短串行 priority 分支做三组 ablation。
5. 保存 worker selection、global rank、planner revision、dispatch reason、fallback/override、snapshot、GPU 和 transfer telemetry。
6. 交替 baseline/HexAgenT 顺序或清空 worker cache，并至少重复运行，隔离 run-order 与 prefix-cache 影响。

### 11.1 Token-aware prefill admission 更新

Router admission 已从固定 request-count 软窗口改为两层 capacity：request count 仅保留为硬安全
上限；fresh prefill snapshot 报告 `max_prefill_tokens` 时，正常 admission 按尚未开始 prefill 的
input tokens 控制。默认每个 worker 保留一批 `max_prefill_tokens` 加 25% headroom，running
prefill 不占下一批 waiting window。snapshot 与 router inflight 通过 workflow/call identity 去重，
snapshot 缺失或过期时退化为 request-count credit。

16K prefill batch 下，约 7K-token BFCL 请求的下一批窗口约为 3 条，而约 770-token ShareGPT
请求约为 27 条；因此不能继续用统一的 32 条解释三种 workload 的 batching capacity。配置参数为
`--hexgent-prefill-token-headroom-percent`，默认 25。

ShareGPT 压力校准不能只提高 50-workflow trace 的 arrival rate，因为最大 active call 数和总 input
tokens 可能仍低于 4P token-window 总容量。下一轮应使用至少 128 个不同 workflows，按
3/6/9/12 workflows/s 做 staircase；只有同时观察到接近满批的 prefill tokens、稳定 decode batch、
非零 global wait 和非零 planner revision coverage，才能进入 baseline/HexAgenT 性能比较。详细记录见
`docs/plans/hexgent-migration/2026-08-04-token-aware-prefill-admission.md`。

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

## 13. RunPod Network Volume 退役与 A100 迁移

2026-08-04 对 CPU-only RunPod 的 `/workspace` 做了退役前审计。本地 checkout 包含更新的
HexAgenT 实现，因此代码必须以本地为准；远端 `/workspace/sglang` 只保留为历史运行证据，
不得反向覆盖本地。

不可重建的远端资产已备份到：

```text
real_trace_profiles/runpod_workspace_backup_20260804/
  runpod_hexgent_assets_20260804.tar.gz
  runpod_hexgent_assets_20260804.tar.gz.sha256
```

归档共 3,750 个成员、267,351,865 bytes。远端和本地 SHA-256 均为
`dbac2a8c6812367bb0466fc7f7bd1aa4a5ac4af1f101b4769f1bbe18e042e500`。已直接验证归档包含
2026-08-04 formal results/traces、50/50/20 Llama-8B 独占 profile、revisioned 4P+4D 日志、
SLO profiles、完整 H100 real-trace profiles、planner/worker/router/GPU telemetry、Mooncake
patch/overlay 和环境 package manifest。

模型缓存、venv、Cargo/Rust cache 没有复制，因为它们可重建；5.3GiB 的旧完整源码快照、旧
source backups 和 tmp 也被排除。本地代码已取代这些过时源码。完整资产边界和恢复顺序见
`docs/plans/hexgent-migration/2026-08-04-runpod-volume-retirement.md`。

下一台 8xA100 Pod 必须先同步本地权威代码并执行环境 preflight，再安装 A100/CUDA 匹配的
SGLang kernel、Mooncake 和 Rust router 依赖。Hopper 上构建的 Mooncake/SGLang 二进制只能
作为版本和诊断证据，不能直接假设在 A100 可用。模型下载完成后，应先用保留的相同 workflows
做 A100 独占 profile，生成模型/GPU 匹配的 service profile 与 SLO；随后完成 1P+1D smoke、
多 worker 验证、arrival-rate/admission 校准，最后才比较 baseline 与 HexAgenT。

### 13.1 归档清理

最初的 retirement archive 仍混有大量历史 scheduling 结果。2026-08-05 已重新整理为
`runpod_hexgent_assets_curated_20260805.tar.gz`。新归档保留 H100/Llama-8B profiles、formal traces、
SLO profile inputs、KV GPU-event/no-cache 与 CUDA peer 测量、capacity calibration、Mooncake
patch/overlay 和环境 manifest。归档中的 61 个 profile JSONL、47,020 行均解析成功，错误行和
zero-output 行均为 0。

所有旧 baseline/HexAgenT 性能结果均从 curated archive 删除，因为没有任何一轮使用当前完整的
revisioned plan、nonblocking fallback、safe priority update 与 token-aware admission 组合。已知问题
包括 revision-0-only fallback、错误 request/queue cap、10x5 重复 workflow、旧 transfer/capacity
模型、中间调试配置、固定运行顺序和 cache-state 污染。历史结论只保留在文档中，不得作为当前
scheduler 性能证据。详细清单见
`docs/plans/hexgent-migration/2026-08-05-runpod-asset-curation.md`。

最终 curated archive 包含 413 个成员、100,734,446 bytes，SHA-256 为
`3e2c6cd18728e065a64b472cf78284931d7b609d4242f3abff3edbd76b9a6f21`。相邻 `.sha256`
文件保存同一值。原 267,351,865-byte broad retirement archive 已被该归档取代，不得再用于
实验分析；当前执行环境拒绝文件删除操作，因此原包和临时解压目录的物理删除仍待完成。

## 14. 2026-08-05 A100 部署与算法语义审计

当前 planner 并未维护独立的 `H_w^P/C_w^P` 和 `H_w^D/C_w^D`。Prefill 与 decode
分别求解、分别计算 projected score，但共享同一组 workflow-level horizon/completion。
普通分支和 fan-out 分支均按较高 projected ratio 优先；短串行特殊分支仍采用
laxity/reveal order。包括短串行在内的所有低压力候选集（不超过 8 个候选）现在执行一次 bounded best-improving
pairwise swap；超过阈值的高压力路径不执行。完整代码证据见
`docs/plans/hexgent-migration/2026-08-05-a100-algorithm-audit-and-bringup.md`。

新的 8x A100-SXM4-80GB 节点任意 GPU 对均为 `NV12`。已校验的直接 CUDA P2P
在 256 MiB 下约为 0.96ms、261GiB/s；该值只代表纯 GPU copy，不代表 Mooncake 请求级
pipeline。Mooncake 第一次 native 测量因 build cache 的 `USE_INTRA_NVLINK=OFF` 无效，
已改为 `ON` 重编译，只有 target 数据校验和 native CUDA-event telemetry 同时通过后才能
发布 production transport 测量。

后续 clean build 同时启用 `USE_CUDA=ON` 和 `USE_INTRA_NVLINK=ON`，并用 Python 3.11
重新生成 binding。250 MiB/50 segments 的校验测量成功：native CUDA-event copy sum 约
1.29--1.31ms，Mooncake 同步 API wall-clock 中位数 4.292ms、P95 4.355ms。日志明确选择
Intra-Node NVLink。该值是空载单 transfer 结果；serving contention 仍必须单独测量。

RunPod runtime venv 与 Rust target 必须放在 container disk 的 `/opt`，否则 network-volume
NFS 会在 wheel 小文件安装或 LLVM output 写入时阻塞。源码、模型 cache、实验结果、补丁和
environment manifest 继续持久化在 `/workspace`。Hugging Face 已登录，Llama-3.1-8B-Instruct BF16 snapshot 已下载到
`/workspace/hub`。A100 TP1/BF16 独占 profile 已完成，使用固定 50 ShareGPT（227 calls）、50 BFCL
（487 calls）和 20 HotpotQA（636 calls），共 1,350 calls；所有请求级 timing 行均无 error。入口脚本为
`scripts/hexgent_real_trace_collectors/run_a100_llama8b_exclusive_profile.sh`，结果目录为
`/workspace/hexgent_e2e/a100_llama8b_bf16_profile_50_50_20_20260805`。请求级 decode 时间使用
`total_latency_ms - ttft_ms`；inter-token delta 仅为诊断字段。已修复 SSE event 跨网络 chunk 时
parser buffer 被重置的问题。权威 `exclusive_profile.jsonl` SHA-256 为
`97d0c6e8325dc711d77a210b43aa33ce9960bd501a1c438c098ab6370cc4e38b`，归档 SHA-256 为
`6f4c22d26cd9a833c320d0b33b756d31c2352e577fda01043c3385835835b5cc`。归档包含合并后的
prefill/decode service table 与 `latency_db.json`。

A100 1P+1D BF16 smoke 已跑通真实 Rust router、Python planner、prefill、Mooncake NVLink 与
decode。单 workflow 6 calls 全成功；低 credit 并发 smoke 21 calls 全成功，3 calls 获得非零
schedule revision，planner batch 最大 3 admissions，wall time 最大 4.65ms。当前源码审计同时确认：
global admission queue 实际仍使用固定 request-count credit，尚未包含文档所述 token-aware window。
默认 credit=64 会使轻负载全部 revision-0 fallback；正式 4P+4D 比较前必须修复并验证这一差异。

A100 1P+1D smoke 证据归档为
`/workspace/hexgent_e2e/a100_1p1d_smoke_20260805.tar.gz`，SHA-256 为
`00ef9cd33c7928aec325c7a92e19743b94f737cf849d6d49cf861c386e95b214`。验证后服务已停止，
不存在残留 GPU workload。

## 15. Token-aware Admission 恢复与全量 Profile Replay

2026-08-05 审计确认，token-aware prefill admission 和 revisioned execution-plan 的最新实现
此前位于 `D:\wt\hexgent-admission` 的未提交改动，而不是主工作区。现已将有效改动合并到
主工作区，并保留主工作区较新的 pairwise local search、SSE timing parser、Mooncake overlay
和 A100 启动逻辑。除该 worktree 外没有其他待合并的 HexAgenT worktree；`main` 和
`backup/pre-v0.5.9-20260411` 是不同上游历史，不应合并到当前 v0.5.9 开发线。

当前 token-aware admission 使用 fresh worker snapshot 中的 `max_prefill_tokens`，按尚未开始
prefill 的 request input tokens 预留下一批窗口；running prefill 不占下一批 credit，router
inflight 与 snapshot 按 workflow/call identity 去重，snapshot 缺失或过期时才回退到固定
request-count safety cap。A100 Pod 上 35 项 `hexgent_` Rust tests 全部通过，本地 Python
regression 为 149 passed。

正式 A100/Llama-8B replay 改为复用 H100/Llama-70B profile 的固定 workload shape 和
input/output length。权威原始数量为 ShareGPT 100 workflows/480 calls、BFCL 100/1213、
HotpotQA 20/636。ShareGPT 有 2 个原始 workflow 含空 user turn，无法原样通过 SGLang chat
validator；正式 replay 保留全部 100 个 workflow，但从首个非法 turn 开始截断这两个会话的
后缀，最终为 100 workflows/477 个可重放 calls。保留 calls 的 input/output length 不变，
也没有复制或补入 workflow。A100/Llama-8B latency 和 SLO 仍需重新独占测量，不能复用
70B/H100 latency。详细证据见
`docs/plans/hexgent-migration/2026-08-05-token-aware-admission-recovery-and-full-profile-replay.md`。

A100 exclusive profiler 另修复了进程生命周期问题：只终止 `launch_server` parent 会遗留
仍占约 69 GiB 的 `sglang::scheduler` 子进程，使下一次 server 在 load weights 时 OOM。
脚本现使用 `setsid` 创建独立 process group，并在 cleanup 中终止整个 group；该问题不能通过
降低 `mem_fraction_static` 掩盖。

正式 4P+4D workers 使用 `--disable-radix-cache`，因此 A100/Llama-8B exclusive profile
也必须禁用 radix cache。启用缓存的 ShareGPT turns 会复用历史前缀并低估 TTFT，不能作为
SLO 分母。Profiler 现默认设置 `DISABLE_RADIX_CACHE=1`；cleanup 会终止 profiler child，
对 server process group 先发 TERM，10 秒后仍未退出再发 SIGKILL。所有不匹配配置的 partial
结果均已删除。

## 16. A100 并行 Profile 与 2P+2D Provisional Replay

2026-08-05 使用同一台 8x A100-SXM4-80GB 节点并行完成 Llama-3.1-8B BF16
独占 profile，并在 GPU 4-7 启动 2P+2D。权威 profile 位于
`/workspace/hexgent_e2e/a100_llama8b_bf16_profile_final_20260805`，覆盖
ShareGPT 477、BFCL 1,213、HotpotQA 636，共 2,326 calls；无 error、无长度不匹配，且
profile 与 serving 均禁用 radix cache。

ShareGPT 3 workflows/s 的 provisional HexAgenT run 中，477 calls 全部沿用 fallback
revision，不能作为 planner 性能证据。后续必须通过 3/6/9/12 workflows/s staircase 找到
具有非零 planner revision coverage 的压力点，再进行正式 baseline/HexAgenT 比较。完整路径、
配置、有效性边界和新增脚本见
`docs/plans/hexgent-migration/2026-08-05-a100-parallel-profile-and-2p2d-provisional.md`。

最终 provisional P95 SLO ratio 为 ShareGPT `1.386x baseline / 1.397x HexAgenT`、
BFCL `20.884x / 25.060x`、HotpotQA `1.371x / 1.410x`。ShareGPT 在 12 workflows/s
时 decode GPU 平均利用率已达 `94.8%/92.3%`，但 planner coverage 仍为 0%，因为 admission
credit 只约束 prefill，而 prefill 平均利用率仅 `37.5%/36.9%`。BFCL coverage 为 29.6%，但
revision calls 的 admission wait 均值约 570ms、P95 2.17s。下一实现重点是 decode-aware
admission 与消除 plan wait；当前数据只能作为 debug evidence。远端/本地归档 SHA-256 为
`35efcff2a2a3d51e713803b8439b3ac0d8d30f81b54f3155f952bbc4501647e6`。

## 17. Pending-only Live Snapshot Alignment

2026-08-05 的 4P+2D BFCL 审计定位到新的 placement 根因：admission payload 可携带最多
1 秒前的 snapshot，而且部分 worker 刷新会整体替换已有 snapshot map。过载时该旧状态会让
planner 连续选择同一个 prefill worker，形成 queue skew 的正反馈。Router 现对已经无法立即
fallback-dispatch 的请求执行 pending-only snapshot refresh：25ms 内的 cache 直接复用，其余
P/D worker 并发 GET `/hexgent/pd_snapshot`，单 worker 上限 50ms；刷新结果增量合并，不再丢失
未返回 worker。刷新期间若真实 credit 恢复，请求会立即 fallback，且不会再向 sidecar 提交
陈旧 reservation。

该修复不改变 projected-ratio、pairwise local search 或 admission credit。Pod 上 Rust
`cargo test hexgent --lib` 为 36/36 通过。详细实现、实验与证据路径见
`docs/plans/hexgent-migration/2026-08-05-live-snapshot-alignment-and-planner-coverage.md`。

BFCL 50 个不同 workflow 的 clean 结果：4P+2D、0.3 workflow/s 下，baseline/HexAgenT
P95 SLO ratio 为 3.7298x/3.2403x，HexAgenT 改善 13.1%，planner coverage 7.6%，prefill
placement 为 110/121/138/118；0.4 workflow/s 已进入过载区，但 HexAgenT 仍将 P95 ratio
从 8.2418x 降至 6.2291x，改善 24.4%，coverage 29.0%，placement 为 114/120/131/122。
这反驳了“BFCL 上 projected-ratio 天然不如 FCFS”的假设；此前退化主要来自 stale state。

ShareGPT 2P+4D、24 workflows/s 的安全路径仍为 0% coverage，因为 prefill 有空闲 credit，
而 decode 已接近饱和。强制 planner-first 的诊断得到 100% coverage，但 P95 ratio 为 1.8189x，
差于同轮 baseline 1.4128x，并出现 102/142/171/62 的 decode placement skew。因此不能为了
coverage 阻塞 idle prefill；下一优化对象是 decode occupancy/remaining-work/reservation 对齐。

## 18. Mooncake 多 Entry Copy 的带宽口径

A100 校准中的 260.7GiB/s 与 189GiB/s 不冲突。前者是单次连续 256MiB
`cudaMemcpyPeerAsync`；后者是一次 Mooncake batch 内 50 个 5MiB entries 被 transport
逐项同步 copy 后的 CUDA-event elapsed sum。250MiB 按 direct 带宽约需 0.936ms，Mooncake
实测 1.293ms，约 0.357ms 差值相当于每 entry 约 7.1us，混合了小 copy 固定成本、带宽利用
不足和 telemetry 扰动。189GiB/s 是该多 entry 实现的有效 payload 吞吐，不是 NVLink 上限。

生产 KV 同样不是天然单块连续 allocation。Llama 3.1 8B 有 32 层，K/V 分开存放，单个连续
token run 通常至少对应 64 个 K/V-layer regions；最大 ShareGPT payload 254.625MiB 约为
2,037 tokens，每个 region 约 3.98MiB。`50 x 5MiB` 只是接近该数量级的受控校准，不是固定
production layout。详细代码边界和后续 1/32/64-entry 校准建议见
`docs/plans/hexgent-migration/2026-08-05-mooncake-segmentation-bandwidth-interpretation.md`。

## 19. A100 HotpotQA Fan-out 调度

2026-08-11 在 6x A100 上使用 Llama-3.1-8B BF16、1P+5D、20 个不同 HotpotQA workflows
完成新的 fan-out 调度审计。正式 runner 现在对 baseline 和 HexAgenT 各自执行 32-request、
并发度 8 的独立 warmup；此前没有逐模式 warmup 的正反顺序结果存在约 5% 冷启动偏差，不能
继续作为性能证据。

Scheduler 现在会记住已经 reveal fan-out 的 workflow。该 workflow 的后续短 calls 在 prefill
和 decode 阶段都继续使用 high-first projected SLO ratio，并跳过 bounded pairwise local
search；纯串行 workflow 的短请求分支不变。Replayer 同时增加 runtime namespace，避免多轮
实验复用 profile IDs 时继承旧 planner DAG state；结果 JSONL 仍使用原始 IDs。本地与 Pod
相关回归均为 143 passed。

`0.19 workflow/s` 的 baseline-first 和 HexAgenT-first 两轮隔离结果中，run-level P95 SLO
ratio 平均为 baseline `2.4674x`、HexAgenT `2.4530x`，改善 0.58%；max ratio 改善 3.44%，
但 mean/median 分别恶化 3.05%/2.68%。合并 40 个 workflow-runs 后，P95 从 2.5193x 降至
2.4874x，P99 从 2.7074x 降至 2.6214x。Planner coverage 为 10.2%--11.3%，planner P95
为 1.18--1.29ms；所有 P/D GPUs 的 utilization P95 均为 100%。结论是 tail objective 方向
正确但收益小、单轮方差仍大，不能声称所有 workflow latency 均改善。详细证据见
`docs/plans/hexgent-migration/2026-08-11-a100-hotpotqa-topology-and-scheduling.md`。

最终有效归档为 `real_trace_profiles/a100_hotpotqa_hexgent_valid_20260811.tar.gz`，SHA-256
为 `879bb6a0e14b1460f0f31be14edc70d513576dad43227655755fd8a8ec8c53ae`。归档不包含未逐模式
warmup、未使用 runtime namespace、revision watcher 失效或只完成单个 mode 的调试结果。

早期曾记录 `0.17 workflow/s`、关闭 pairwise local search 的 HexAgenT-only P95 ratio
`2.1740x`，并与其他运行的 baseline `2.2130x/2.3290x` 横向比较。该数字只用于提出 local
search 扰乱 fan-out barrier 的假设，不是正式 paired A/B：它缺少逐模式 warmup、runtime ID
隔离和同轮 baseline，并且压力点与最终 `0.19` 不同。不得用它替代上述最终结果或宣称更大的
性能提升。Planner CPU 非瓶颈的判断仍有效，最终 planner invocation P95 为 1.18--1.29ms。
上文归档的 pooled `1.27%/3.18%` 使用线性插值；当前统一 nearest-rank 口径为
`1.11%/2.61%`，样本没有变化。

## 20. Repository Curation and Current Entry Point

2026-08-12 已将当前权威复现入口收敛到 `docs/hexgent/README.md`，A100 三 trace 正式结果收敛
到 `docs/hexgent/2026-08-12-a100-three-trace-results.md`，保留 profile/result 资产及 SHA-256
列在 `real_trace_profiles/README.md`。

旧 Python `HexgentLoadBalancer`、`hexgent_vendor` simulator prototype、历史 Qwen timing
imputation scripts、错误 SLO/revision-0/未完成实验报告已经删除。当前 production serving
路径只包含 Rust PD router、Python planner sidecar、worker snapshot/mutation bridge、真实
trace profiler/replayer 和 Mooncake measurement tools。详细清理记录见
`docs/plans/hexgent-migration/2026-08-12-repository-curation-and-three-trace-summary.md`。
