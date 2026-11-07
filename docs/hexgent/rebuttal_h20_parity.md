# H20 parity audit and preflight

Audit date: 2026-11-07. Real starting revision: `af74dbfbb1561147a3853b1ae145ede06de91f13`. Simulator reference: `75d49e20e890d9c90644f836eafc8a9dd7469637` in read-only `D:/Github/HexAGenT`. This document records source inspection and CPU decisions; live H20 results remain pending authentication.

## Provenance and active execution path

The local submitted-paper PDF has SHA256 `eecff87f6d2df47dd6362ed46f15b6f78824541dbb87c3a93f25819efbef9f87` (17 pages). The review PDF has SHA256 `d29af91c16a6bc7da0d351f006ef4ba9aab91104e6f30f1a0a5dc412a2d8d59a`. The working rebuttal draft is a planning document, not measured evidence.

Current simulator benchmark entrypoints instantiate `SGLangHexAgenTScheduler` and then `AgenticPDExactScheduler`: `benchmarks/runner_core.py:108-118`, `benchmarks/run_sglang_comparison.py:27-38`, and `benchmarks/run_tool_kv_ablation.py:13,56`. Its core is `pd_disaggregated/sglang_scheduler_core.py`. At the starting revisions, normalized text matches real `python/sglang/hexgent/scheduler.py`, SHA256 `7a61c202ed30bc838462449c301d94fc09396239d25d761308834a8c3ae0c03a` (1176 lines). The legacy `evaluate.py` path's `PRPLSJointScheduler` uses another ordering and is a separate implementation.

The paper Section 5.4, Eq. 7 and Appendix C specify decreasing projected normalized completion. The current ratio path's `high_first` agrees. Both cores include additional workload-dependent ordering branches; core equality alone establishes neither paper parity for every branch nor identical end-to-end runtime semantics.

## Mechanism parity matrix

| Mechanism | Paper / simulator | Real path and evidence | Experiment consequence |
| --- | --- | --- | --- |
| Workflow priority | Paper Section 5.4: decreasing projected completion from arrival divided by longest revealed dependency path. Current simulator uses shared core. | `scheduler.py:47-51` defaults high_first; `573-583` includes short sequential call-laxity special case. | Record effective flags and workload branch. Keep legacy PRPLS distinct. |
| Joint P/D selection | Shared core enumerates P x D with projected availability; paper Sections 5.2-5.4 additionally describes feasible pair/capacity accounting. | `scheduler.py:483-530`; `PrefillStageState:128-132` carries P/D availability, without D capacity/demand. `planner.py` locks selected D for later decode ranking. | Current P-placement projection has a capacity boundary requiring the coordinator's separate fix review. |
| Placement lifecycle | Simulator adapter passes existing assignments as locks; paper running calls remain uninterrupted. | Global admission can revise pair before dispatch; snapshot calls have locked P/D (`planner.py:732-733`); router creates bootstrap at dispatch. Ownership after dispatch remains fixed. | Report fixed-pair serving. No live P/D migration claim. |
| Queue projection | Same core successively projects queues. Simulator asynchronous generation/stage guards are in `simulator.py:926-968`. | Full revisioned plan in planner, atomic router publication (`pd_router.rs:369-388`), worker monotonic revision guard (`hexgent_worker.py:44-58`). Local updates only affect compatible waiting queues. | Measure real RPC delay/rejections and fallback; simulator has no equivalent network execution failures. |
| Effective queue-update flag | Real policy is distinct from the legacy launcher variable. | `planner.py:50,53-64` reads `HEXGENT_ENABLE_LOCAL_PRIORITY_UPDATES`, default on. Launcher line135 sets old `HEXGENT_ENABLE_LOCAL_WAITING_MUTATIONS=0`, which current Python never reads. | Record the new variable's effective value. The old export cannot establish mutation disabled. |
| Pairwise refinement | Same core; <=8 calls, <=28 pair evaluations; fan-out cases skip refinement (`scheduler.py:233-254,750`). Paper Section 5.4 describes max-ratio, mean-ratio, total-latency ordering. | Source defaults pairwise on (`42-44`); README reproduction rule6 prescribes off. | Distinguish source default from prescribed experiment configuration and actual process environment. No historical effective value has been recovered. |
| KV reuse / transfer | Simulator has session cache home/global-copy/full/local/global modes (`simulator.py:264-268,634-641`). | Mooncake P-to-D transfer exists. Current real request metadata lacks equivalent prefix/cache-home semantics. README prescribes radix cache disabled. | Cache-disabled subset can align; byte estimates and fixed token growth do not demonstrate prefix reuse. Collect actual transfer bytes/durations. |
| Tool accounting | Simulator external tool events avoid GPU service and can update revealed horizon after completion (`simulator.py:369,402-407,599`). Paper Section 4.1/Appendix D uses observed completed-tool duration. | Replayer issues every trace node as a model request; BFCL tool node conversion (`workflow_trace.py:612-627`) creates LLM requests. There is no explicit CPU tool-delay path. | Label fixed-token LLM DAG replay. Tool-labeled nodes provide no real external-tool evidence. |
| Capacity / fallback | Decode core has token capacity and scheduled intervals. Coordinator is reviewing oversized-demand correctness separately. | D capacity comes from live snapshot / worker labels; missing snapshot can use default1m tokens (`pd_router.rs:1933-1936`). Router native fallback gates P credit (`424-428,528-566`), without planner D-feasibility proof. | Require live snapshots and verify every pilot call's input+target-output fits actual D pools; no guessed capacity. |
| Stale / coverage | Simulator plan generation guards are logical events. | Response revision/mode/timing support successful dispatch coverage and fallback. Worker stale reasons are lost by scheduler wrapper; router warnings do not provide a reliable denominator. | Report errors, revision>0/=0/missing separately; stale rate remains unmeasured until structured outcomes exist. |

Code line numbers in the matrix refer to the starting revision. Candidate patches move lines and must be identified separately.

## Deterministic CPU decision evidence

Run from this worktree:

```powershell
& .venv-h20/Scripts/python.exe docs/hexgent/rebuttal_h20_parity_probe.py --simulator-core D:/Github/HexAGenT/pd_disaggregated/sglang_scheduler_core.py
```

The initial full output is preserved at `artifacts/rebuttal-h20/parity-initial.json` and explicitly labels CPU provenance. Both cores produce the same decisions for all four tested flag settings:

| Order / refinement | Effective behavior | Selected order | Pair evaluations |
| --- | --- | --- | ---: |
| Both variables unset | high_first / on | high-risk, low-risk | 1 |
| high_first / 0 | high_first / off | high-risk, low-risk | 0 |
| low_first / 0 | low_first / off | low-risk, high-risk | 0 |
| high_first / 1 | high_first / on | high-risk, low-risk | 1 |

The high/low examples hold service cost constant and vary known workflow completion. They test a ratio branch, not every special case. Launch scripts `start_hexgent_pd_smoke_4p4d.sh` and `run_h100_formal_comparison.sh` do not set either flag; process environment and sourced env files determine actual runs.

## Candidate H1: consistent revealed horizon within one plan

Symptom: two same-workflow roots with estimated standalone service 11/101ms enter the same prefill rank with horizons 11/101ms, while the final revealed horizon is 101ms. Reversing input order produces 101/101ms. Published response metadata says 101ms for both, concealing the earlier ranking denominator difference.

Root cause: `planner.py:1181-1198` reveals each pending request and immediately copies the then-current workflow horizon into a `PrefillCallInfo`; earlier pending/waiting calls are not refreshed after the rest of the same known batch is revealed. The shared core is unchanged.

Proposed semantics: reveal all calls observed at this plan's cutoff, including waiting snapshots, then refresh the workflow horizon of every waiting/pending call from that consistent revealed state before ranking. Keep unknown future requests out. Preserve assignment locks and service/capacity estimates.

The coordinator authorized an independent experimental candidate and regression tests, pending review before canonical use. Required tests cover both sibling orders, existing waiting calls with newly revealed child, observed decode history, unknown future child, and single-call invariance. This patch affects fan-out priorities and projected completion decisions; performance results must identify original versus accepted revision. Candidate status and patch hash are maintained in `H20_STATUS.md`.

## Remote preflight

The user authorized existing aliases `TargetMachine2`, then `TargetMachine`. Both were attempted once with `BatchMode=yes`, `ConnectTimeout=12`, `StrictHostKeyChecking=yes`; both returned `fred@192.168.1.148: Permission denied (publickey,password)`. No remote command ran. The local agent already lists an ED25519 and an RSA key. The coordinator is helping the user install the existing public key through their manually authenticated terminal.

Status: authentication-blocked. H20 hostname, actual Fred directory, GPU inventory/availability, model, and runtime remain unverified. No remote files, permissions or processes were changed.

After the user confirms authentication is ready, one bounded read-only probe must capture hostname, `id`, `whoami`, `pwd`, HOME, directory ownership, GPU UUID/name/total+used memory/utilization, compute-process ownership, topology, driver/CUDA, installed runtime path/version, and model configuration. Verify actual `head_dim` (explicit config takes precedence), KV heads, layers, total/active MoE parameters, checkpoint revision, TP and weight/KV dtype. The current roofline estimator derives head_dim from hidden/heads, so Qwen configurations with explicit differing dimensions need the coordinator's model fix before calibration.

The existing multiworker launcher hardcodes TP1 and a single visible device per P/D worker (`191,218`); TP2 requires separate support. It also has shared PID/log defaults and process cleanup. Use only a new experiment-owned directory and PID set under the verified Fred root; audit cleanup before launching. Do not run default cleanup against existing services.

## Bounded pilot contract

Select an already-available feasible dense model after preflight. A small model pilot is reported using its actual model name and size. First use 1P+1D, cache disabled, explicit high_first, explicit pairwise off, identical TP/dtypes, and fixed input/output traces. This proves the serving path and calibrates stages; a single P/D pair offers no placement-choice experiment. Multiworker evaluation is conditional on free physical GPUs.

Freeze one chain and one fan-out/join subset before comparing methods. Build a new isolated profile on the same H20 topology and model/cache configuration. Separate calibration tasks from held-out tasks. Record input/profile hashes, independent source-task count, repeats, seed, revisions, all flags, arrival process and observation horizon. Calibration chooses two fixed loads based on observed service capacity before held-out method results are inspected.

Save all raw workflow rows, errors, request and response times, SLO, ratio, Req95/Req99 (nearest rank), fixed-scale violations, each planner invocation wall/CPU time, dispatch coverage/fallback/missing revisions, stale-action outcomes when available, snapshots, GPU samples, request-stage logs and measured KV transfers. Repeated copies of one trajectory are repeats, not independent tasks. Require expected trace/workflow/call coverage and zero execution errors before interpreting performance; preserve invalid/failed runs in the ledger.

Use `summarize_scheduling_replay.py` for SLO metrics. Its failure-preserving path leaves missing reference SLO null, measures from persisted workflow arrival for new records, and labels old records' earliest-request origin. The older `analyze_hexgent_1p1d_run.py:49-69` drops error calls and substitutes observed latency for missing isolated SLO; use its other telemetry only, not its SLO output.

Simulator-vs-real comparison must share trace/profile hash, model/revision, TP, dtypes, H20 parameters, topology/capacity, cache/tool convention, arrival process and scheduling flags. Report per-stage calibration separately from held-out end-to-end error and ordering agreement. H100/A100 retained archives supply historical inputs only; their latency is not H20 calibration.

## Current validation and missing evidence

An isolated `.venv-h20` resolves initial missing-FastAPI failures. The first combined local suite passed 129 tests with 13 FastAPI/Starlette deprecation warnings; later measurement/candidate tests are recorded in status. `cargo check --locked --offline` stopped because the tracked Cargo.lock needs updating; no lockfile was modified. Windows CPU tests do not validate Linux CUDA, Mooncake or the compiled gateway.

Real H20 replay, GPU memory feasibility, stage/transfer calibration, uncertainty across live repetitions, and simulator-vs-real measured errors remain missing. Reviewer evidence coverage: this audit supports implementation disclosure and E0/E4 preflight, while E4 performance claims require the missing real measurements; tool/cache benefit claims require additional supported execution paths.

## Accepted revisions and remote execution checkpoint (2026-11-07)

The starting-revision matrix, initial CPU probe, candidate descriptions and
initial SSH failures above are historical records from `af74dbf`. They remain
unchanged. Subsequent accepted corrections are identified below so that the
initial audit is not mistaken for the current implementation.

| Change | Accepted real commit or port record | Scope |
| --- | --- | --- |
| H1 | `33e34d5ccc4fbdd1f54ab87737ec6c70cf7580db` | Refresh every observed waiting/pending call's revealed horizon before ranking within one plan. |
| Q1-real | `502e70a04e93ee08d930cd6471463a4a43c0188d` | Honor explicit attention head dimension while retaining the existing TP convention. |
| R1_REAL_CAPACITY | `92e563764b11fad6df360352fdb232e3d547b7a8` | Preserve total decode capacity and account for occupied KV once through intervals. |
| C1 real port | `77400e9d6d40bb982f1decfc87856dd279598b70` | Pass known demand/total capacity before P/D assignment; preserve existing locks and omit infeasible destinations. |
| C2 fanout ownership real port | Exact two-hunk port of simulator `a2213882d25ab28d886c63cef72c18781efac5dd`; isolated real commit is recorded by the coordinator. | Mark fanout history only on its owning workflow; retain the mixed-batch refinement gate. |

The C2 real core matches the accepted simulator C1+C2 normalized text, SHA256
`cb71482755ba8f27c5234effe2e4e1c0897763d45e00001acfe9f592b3d139a3`.
Its H20 pure-core check recorded red 18 failed/6 passed, then green 24 passed,
with CPUs 9,11, single-thread numerical settings and a 2 GiB address-space limit.
These are CPU correctness checks. Full runtime and serving results have their
own evidence records. C2 ownership is separate from R1_REAL_CAPACITY; the early
snapshot-capacity proposal used the temporary label C2 before R1 was assigned.

Active work now runs on H20 in
`/home/fred/hexagent-rebuttal/worktrees/rebuttal-h20` under
[SERVER_WORKFLOW.md](/home/fred/hexagent-rebuttal/coordination/SERVER_WORKFLOW.md).
GitHub branch synchronization is authoritative; Windows worktrees serve as
mirrors. Synchronization is complete only after the coordinator verifies the
H20, GitHub and mirror revision identities. The earlier authentication-blocked
paragraphs remain initial-preflight provenance.


## Server migration and deployment checkpoint

On 2026-11-07 development and tests moved to the Fred-owned H20 worktree under `/home/fred/hexagent-rebuttal/worktrees/rebuttal-h20`. GitHub synchronization is coordinated through SERVER_WORKFLOW.md. Windows results remain separate provenance, and local pending files are preserved until a safe mirror reconciliation.

Accepted source ledger: H1 `33e34d5ccc`, Q1-real `502e70a04e`, R1 `92e563764b`, C1-real `77400e9d6d`, W1 `7687dbf1c`, C2 fan-out ownership `78833ed18d433e838b45e7d852a829e80137635b`. C1+C2 normalized core SHA256 is `cb71482755ba8f27c5234effe2e4e1c0897763d45e00001acfe9f592b3d139a3`, identical to simulator `a2213882d25ab28d886c63cef72c18781efac5dd`. This is the current core, distinct from the original matrix's starting revision.

Measurement commits: `898f9a936` preserves expected-trace completeness and fixed-scale failure bounds; `76003586b` records actual SSE token usage separately from legacy values and preserves failed profile rows; W2 `bd91b0094` logs received mutation outcomes without changing payloads/queue application. Runtime input and TP2 reference wrapper commit: `2e4b61464`. Logging overhead is included in real serving. Mutation counters cover received requests with results, not unobserved transport failures, and must be computed from per-run log slices.

Existing-image H20 build produced a 52 MiB custom gateway, SHA256 `a96b9c62a8bfaafc89703a2eb8dc2cdf801ef79ac4e6bede0c287261a9b76148`; 36 targeted Rust tests passed and custom Hexgent options were asserted. Fixed Rust 1.90.0 and locked dependencies were built within the approved 24 GiB/no-extra-swap, CPU1/3, 16 GiB tmpfs budget; compiler container was cleaned after success. The binary's Rust source is the starting gateway code, while accepted algorithm changes are Python planner/core changes. Source bundle and pending-patch provenance remain recorded separately.

The fixed subsets use ID-sorted calibration2/heldout4 workflows from ShareGPT and HotpotQA, totaling217calls. Historical hints/model IDs were removed. Actual local tokenizer preparation detected the installed default BatchEncoding return; v1 incorrectly counted its two fields and is retained as INVALID, never used for serving. Explicit `return_dict=False` produced v2 token-ID lengths. All217 prompts gained25 chat-template tokens relative to the archived declared lengths; messages and output targets were unchanged. These are local tokenizer counts; actual server SSE usage must still verify them. Runtime v2's largest input-plus-output budget is2234, below the proposed8192 context.

Only complete static model/nominal H20 configuration is injected into current calls. H20 labels can shadow request hardware dictionaries, so actual transfer calibration is frozen through planner transfer environment variables after independent calibration. The old MAX_INFLIGHT_TRANSFERS export is only read by snapshot projection, not enforced by the runtime; the new wrapper leaves it unset and declares the overlap assumption. No historical service/SLO/future-child fields enter online planning.

The planned reference uses actual dense Llama70B BF16 at TP2 for both P and D, four physical H20 GPUs in one pair. This validates correctness/priority and calibration; there is no multiworker placement choice. Frozen prompt/output-target DAG replay does not measure live agent task solving, external-tool execution, or prefix-cache reuse. Fresh per-call H20 streaming-exclusive times define the DAG critical-path denominator; this convention is distinct from H100 reference values and from a measured whole-workflow-alone latency with within-workflow resource contention. At this checkpoint no model-serving performance result is claimed.
