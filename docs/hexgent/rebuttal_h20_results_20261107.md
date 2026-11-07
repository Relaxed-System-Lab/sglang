# H20 pilot results and remaining blockers (2026-11-07)

The bounded real H20 pilot verified the accepted R2 timing correction and W3 HTTP-body correction, collected eight valid actual standalone workflow references, and completed four ShareGPT replay runs. It did not establish canonical scheduler gains: the two configured Ours ShareGPT runs used the admission-credit fastpath with canonical coverage 0/16 each. Hotpot burst failed in the KV transport path, and the corresponding Ours run failed during warmup before formal replay. Hotpot low was not run under the fixed budget rule. Canonical matched simulator experiments and the scheduler-gain objective remain incomplete.

All GPU work ended at **2026-11-07 17:44:26.602981 UTC**, before the client, worker and container deadlines. The owned container `hexagent-h20-r2w3-01a1b627` was removed, and all eight GPUs were observed at 0 MiB afterward. No rerun, retry, threshold change, simulator rollout or new GPU allocation followed the failure. GitHub synchronization remains with the coordinating chat; this local archive does not imply a push.

## Source, hardware and implementation

| Component | Actual version or setting |
| --- | --- |
| Runtime Python/measurement source | `08885ec27573ee1807b822e786ce95ebd06ec884` |
| R2 timing correction | `4307bf73b49f7ba02d643f53d4659e8fd90da51a` |
| W3 HTTP-body correction | `2e4a33a2e8ae6ca89346bf36ecb4064756ee85f6` |
| Gateway ELF | SHA256 `f52f2c1e02f32ca37effb60b3d51d306b1bb92f1f4b2ae8c3f13211b931e4831` |
| Rust build provenance | Built at R2; all 409 tracked gateway paths identical at runtime source revision |
| Shared scheduler core | SHA256 `cb71482755ba8f27c5234effe2e4e1c0897763d45e00001acfe9f592b3d139a3`, identical to accepted simulator core `a2213882d25ab28d886c63cef72c18781efac5dd` |
| Model | Existing dense `meta-llama/Llama-3.1-70B-Instruct`, BF16; checkpoint snapshot `945c8663693130f8be2ee66210e062158b2a9693` |
| Geometry | hidden8192, intermediate28672, layers80, heads64, KV heads8, head dimension128, vocab128256 |
| Topology | 1 P group on physical H20 4/5 TP2; 1 D group on 6/7 TP2; four physical GPUs |
| Actual capacity | 73197 KV tokens per group; max-running16; P max-prefill requests4; max-prefill tokens16384 |
| Other runtime settings | context/chunk8192, memory fraction0.82, BF16 KV, radix cache off, CUDA graph off, Mooncake TCP |
| Nominal estimator | 4096e9 B/s and FP16 148e12 operations/s, catalog values; batch1 in the separate serial-call diagnostic |
| Serving image | Existing `lmsysorg/sglang:dev`, image SHA256 `c0305ed60a44642322a2b1525fd025e42e656aebca0454266b16df5f7dddad2e` |
| Runtime libraries | Python3.12.3, torch2.11.0+cu130, transformers5.12.1, source-overlay SGLang v0.5.15 |

This 1P1D topology has no multi-instance placement choice. Replay uses fixed prompts, generation budgets and DAG dependencies with forced output length; it does not measure live task-solving, external-tool execution or prefix-cache reuse. The original independent-task provenance is unknown. All 217 sanitized prompt counts gained 25 chat-template tokens relative to the archived declared counts. Actual server usage verified the new counts. No historical service, SLO or future-child timing entered online planning.

## Verified correctness changes

R2 removes a double count in attained service. The old decode-side transfer wait included upstream prefill; summing it with prefill forward time counted prefill twice. The corrected same-clock P prefix and P transfer tail combine with D execution, excluding the separate D pipeline-wait diagnostic. Missing new components do not invoke the old sum. A real identity-matched preflight produced 692.3920828849077 ms from worker components and 692 ms in the Rust completion event, consistent with one final integer rounding. Synthetic coverage preserves zero/sub-ms values and checks that changing D pipeline wait does not change service.

W3 marks the two mutation request objects as FastAPI JSON bodies. The prior signatures treated `obj` as a missing query parameter and returned HTTP400. Actual TestClient coverage used the real request classes, converter, handlers and W2 logger: 6 valid requests failed before the fix, then all 10 valid/invalid cases passed. Live preflight exercised both mutation routes on P and D: HTTP200 with explicit business `not_found`, one DP aggregate per request. This proves routing to the handler, not a successful mutation of a real waiting request.

The rebuilt gateway passed 39 targeted Rust tests; the actual-image Python suite passed 397 tests. Exact executed replay/capture/diagnostic sources now reside in `scripts/hexgent_real_trace_collectors/h20_pilot`; the final packaged CPU suite passed 69 tests, with 2 FastAPI capture tests deselected because the CPU environment lacks FastAPI. All 5 capture tests had already passed in the actual image. These checks validate correctness and instrumentation separately from scheduling activation.

## Frozen workload, reference and load protocol

ID-sorted splits fixed two calibration and four heldout workflows per dataset before method results. ShareGPT heldout has 16 calls across `50SbvAt_0`, `6CWcc2w_0`, `6YMEjv4_0`, `88iCu0j_0`; HotpotQA heldout has 133 calls across `hotpotqa_10` through `_13`. The four mixed calibration workflows contain 68 calls. All 8 new standalone workflows and all 149 calls completed, with actual model/token usage, IDs, parents, arrival, child-release ordering and raw hashes independently checked. Each reference runs one workflow alone while preserving its internal fanout and resource contention. Old serial-call critical-path proxies are not used as formal reference SLOs.

One mixed four-workflow, zero-offset native calibration cohort set load only. SG makespan was 98258.22094726562 ms and Hotpot makespan 110048.982421875 ms; both include cross-dataset contention, so they are not pure capacity measurements. Low arrival rates were fixed as 0.5 times two workflows divided by each makespan: SG 0.010177265478241172 WF/s; Hotpot 0.00908686275868031 WF/s. Seed20261107 gives SG offsets `[0,81.70145120596857,128.72031406345522,290.7352186830157]` s and Hotpot `[0,91.50543823129867,144.1664569452152,325.6227790592575]` s. Burst offsets are all zero. Methods share exact frozen vectors.

Native uses fixed round-robin without the external admission gate. Configured Ours uses high_first, pairwise1, local-priority updates1 and projected_finish, admission enabled, 10000 ms planner timeout, 300000 ms maximum admission wait, 10 ms poll, queue cap64, headroom16 and token headroom25%. Both use the same workers, low-numeric-first priority and frozen transfer model. Each method gets the same one-call 39-input/8-output warmup, empty P/D gates, fresh owned Ours controllers and per-run worker log slices.

Before each pair, the guard required remaining time at least `1.25*2*(last planned arrival + maximum same-cell standalone latency)+60 seconds`. After an approved amendment, an insufficient pair is recorded as NOT_RUN_budget and later registered, wholly unattempted pairs may be considered under the unchanged deadline. Cases05/06 failed that rule (857.54 s available versus 1161.78 s required); cases07/08 passed it (347.72 s required). No case was selected using measured gains. There is one repetition and four workflows per complete cell; workflow-ratio P95 and P99 both equal the maximum of four ratios. Request Req95/Req99 are separate client-call latency percentiles in milliseconds; for16-call SG cases both select the maximum of16. No population-tail estimate or uncertainty across repetitions is established.

## Complete eight-case outcome ledger

Ratios use actual workflow end-to-end latency divided by its actual whole-workflow-alone reference. Coverage counts canonical positive-revision dispatches; the configured Ours label alone does not establish execution of the canonical policy.

| Case | Dataset/load/method | Formal success | Mean ratio | Workflow-ratio P95=P99 | Canonical coverage | Outcome |
| --- | --- | --- | ---: | ---: | --- | --- |
| 01 | SG low native | 4/4 WF,16/16 calls | 1.001001 | 1.002125 | Native | Complete |
| 02 | SG low configured Ours | 4/4 WF,16/16 calls | 0.999508 | 1.002149 | 0/16 | Credit fastpath data |
| 03 | SG burst configured Ours | 4/4 WF,16/16 calls | 1.054286 | 1.068082 | 0/16 | Credit fastpath data |
| 04 | SG burst native | 4/4 WF,16/16 calls | 1.049736 | 1.064405 | Native | Complete |
| 05 | Hotpot low configured Ours | Not run | n/a | n/a | n/a | NOT_RUN_budget |
| 06 | Hotpot low native | Not run | n/a | n/a | n/a | NOT_RUN_budget |
| 07 | Hotpot burst native | 1/4 WF,95/133 usage-valid calls | Incomplete cell | Incomplete cell | Native | 38 HTTP503 errors; transport failure |
| 08 | Hotpot burst configured Ours | Formal not started,0/133 calls | n/a | n/a | n/a | Warmup HTTP500/KVTransferError |

SG cases01–04 have alpha1 violation counts 3/4,2/4,4/4,4/4, respectively; all have zero alpha2 and alpha4 violations. For07, successful `hotpotqa_11` had 112567.46606445312 ms latency / 68896.486328125 ms reference = 1.6338636709039356. The other three workflows failed and retain null ratios. Failure-inclusive alpha1 violations are4/4 and alpha2/4 are3/4. Do not summarize the single successful workflow as a complete-cell comparison. Case08's warmup error is not a formal workflow sample; all formal ratios are missing. The planned block was counterbalanced, but its completed subset is not a complete balanced block.

Request latency tails, preserving the request population separately from workflow ratios:

| Case | Recorded requests | Req95 ms | Req99 ms | Population |
| --- | ---: | ---: | ---: | --- |
| 01 | 16 | 23066.053389 | 23066.053389 | All16 successful calls |
| 02 | 16 | 22912.650849 | 22912.650849 | All16 successful calls |
| 03 | 16 | 22994.721588 | 22994.721588 | All16 successful calls |
| 04 | 16 | 23014.574330 | 23014.574330 | All16 successful calls |
| 07 | 133 | 21524.620597 | 25332.969652 | All133 recorded outcomes, including38 HTTP503 errors |

Cases05/06/08 have no formal request percentile. Successful-HTTP-only request distributions are retained separately in the external ledger. [Case ledger](h20_20261107/cases.csv), [32-workflow ledger](h20_20261107/workflows.csv), and [provenance](h20_20261107/provenance.json) retain every registered case/workflow. Overall counts are17 successful workflows,3 failed,8 NOT_RUN_budget and4 NOT_STARTED_warmup_failure, with197 formal request rows and38 request errors; six warmup rows are excluded from formal statistics. `valid_measurement` in the CSV describes the saved completeness/usage gate, not canonical policy coverage.

Formal SG Ours telemetry contains 38 and 33 event-actor `/plan_pending_admissions` records. Every record has zero admissions/results; both runs have zero canonical planning invocations and zero observed worker mutation attempts. Their small wall/CPU times measure empty event passes, not solver overhead. Canonical solver overhead is **N/A**, not zero. Case08's saved summary contains 10 warmup telemetry records because no formal-start offset was established; those records are excluded from formal overhead and retained as warmup-only evidence. Exact per-workflow and per-case records are retained in the final-analysis ledger.

## Activation audit

The HTTP router uses the fresh reported P prefill request limit4, bounded by CLI cap64. Effective request capacity and headroom are therefore4; the queued-token target is16384 plus25%, or20480. Running requests are excluded from queued occupancy. Enqueue can immediately grant a native fallback with credit, including its class3 branch; a request already granted is explicitly not registered as pending with the sidecar. SG has serial calls within each workflow and four workflows in burst; observed peak active calls is4, with no sampled worker waiting queue. The saved events and zero pending admissions are consistent with this fastpath; they do not prove every per-request cached snapshot. SG low/burst peak active calls were1/4. Burst P logged up to3 new sequences and4 inflight requests; D ran up to4 with1 transfer, so concurrency existed even though the sampled waiting queue was0. Fresh snapshot credit capacity is4; without a fresh snapshot the router falls back to CLI64 and has no snapshot-derived token target. SG does not cover the queued scheduling mechanism; normal immediate grant alone is not an implementation bug. Hotpot Ours activation under its20-root pressure remains unknown because formal replay never started.

The configured admission path was enabled. Formal `/event` requests succeeded; no saved `/enqueue_admission`, `/admit` or `/schedule` calls establish a canonical planning attempt. These cases provide no evidence of a planner timeout or unreachable sidecar. Per-call fallback reasons were not structured in the formal raw rows, so the exact reason for each individual response cannot be reconstructed directly. A CPU event probe found that dispatch and prefill-completed events leave workflow state absent; root completion reconstructs the root, while a later fallback child completion does not reconstruct that child once the workflow already exists. This is a separate history-recovery observation, not a demonstrated cause of the SG fastpath or evidence about queued Hotpot behavior. Source-line evidence and the successful inline probe output are retained under `final-analysis/activation-audit`. A later command to save/run a standalone probe/test was rejected by automatic approval with the reason `blocked by policy`; it did not execute and was not retried. Consequently this archive contains the inline output/source evidence, not a separately rerunnable activation probe file.

## Observed transport failure

The first recorded operational error was at17:32:32.268730 UTC: Mooncake TCP could not connect to `192.168.1.148:16537`, reporting `Cannot assign requested address` (EADDRNOTAVAIL). The captured log contains13041 connect-error records. The observed sequence is native transfer failure, Python Mooncake failed-session blacklist at17:32:46, D Failed/KVTransferError HTTP500, then router circuit-open HTTP503 at17:32:56. The next Ours warmup failed on this already unhealthy KV path.

The failed session is blacklisted after a threshold of one recorded failure (`conn.py:1470–1492`), and future transfers short-circuit when that remote session is marked not alive (`1355–1371`). D receives Failed at1763–1768 and raises at2105–2117. Re-registration can clear the failed-session blacklist (`1664–1673`); optional send-probe recovery (`1840–1875`) defaults off (`environ.py:431–432`), and this run's actual enablement was not independently established. Native batch-transfer is called without a Python timeout argument; the bounded source audit did not locate the C++ timeout implementation. One failed room waited about18–19 seconds after entry; it did not show the configured300-second Python bootstrap/input timeout. A logged D transfer duration0 reflects a missing successful endpoint, not a measured zero wait.

Port exhaustion or source-port pressure is a hypothesis, not an established OS root cause. The later socket snapshot was not contemporaneous with the first error and cannot prove it. Local cgroup OOM/max events were all0, memory peak15762726912 bytes and nofile1048576. All engines still allocated GPU memory before the controlled stop. HTTP health200 did not prove the P-to-D KV transport was healthy. No sysctl, network, permission, model or scheduling threshold was changed.

## Separate heldout stage/service prediction diagnostic

This CPU-only analysis uses the earlier **a65d6f9** serial native call measurements and old gateway ELF `a96b9c62a8bfaafc89703a2eb8dc2cdf801ef79ac4e6bede0c287261a9b76148`. It is not R2/W3 pressure scheduling data. All149 heldout calls have exact serial-order, P/D-room, token/cache and single-chunk stage pairing, actual usage and zero errors. Two68-call calibration passes preceded heldout collection, so these are not cold-start measurements.

Predictions use the unchanged online nominal TP2/BF16 estimator, batch1, complete model geometry and nominal hardware values above. No P/D scales are fitted or applied. Transfer parameters were frozen from the earlier uniform68-call calibration pass: `T_ms=5.435876377145576+420.2114089420461*(fullgroup_bytes/1e9)`, effective full-group bandwidth2379754520.5106893 B/s. No transfer refit uses heldout data. The estimator file SHA256 is `42cded95817e809924b65090992d3772b24008c0ec95ef6165983a501ddd3407` at a65,08885 and the loaded source.

| Predicted component vs observation, all149 | Bias ms | MAE ms | RMSE ms | MAPE |
| --- | ---: | ---: | ---: | ---: |
| P prefix | -62.91 | 62.91 | 64.15 | 8.58% |
| Effective transfer surrogate | -0.93 | 5.57 | 7.01 | 2.74% |
| D forward | -1801.40 | 1801.40 | 2653.07 | 51.27% |
| Additive P+T+D service | -1865.24 | 1865.24 | 2693.90 | 35.75% |
| Same service prediction vs client total | -1876.87 | 1876.87 | 2701.79 | 35.99% |

| Dataset | N calls | P-prefix MAPE | Transfer MAPE | D-forward MAPE | Additive-service MAPE | Service vs client-total MAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 16 | 22.74% | 6.11% | 58.24% | 55.86% | 55.92% |
| HotpotQA | 133 | 6.88% | 2.33% | 50.44% | 33.33% | 33.59% |
| All calls | 149 | 8.58% | 2.74% | 51.27% | 35.75% | 35.99% |

P/D and total signed errors are negative for every call; all are retained. Transfer has79 negative and70 positive errors. The all-call aggregate weights each call equally, giving Hotpot133/149 weight; it is not dataset-balanced or workflow-balanced. MAPE averages per-call absolute relative errors. Bias uses prediction minus observation.

Observed P-prefix is logged P-forward minus reconstructed last-chunk handoff tail; the synchronous send/enqueue cost before the transfer timestamp remains in P-prefix. The effective transfer surrogate covers the post-send asynchronous communication/control/poll tail and is not a physical DMA bandwidth. D-forward excludes upstream D transfer wait. P-prefix and transfer rounding errors anticorrelate and cancel in additive service; maximum transfer rounding interval width is2.981009 ms. Client total separately includes network/bootstrap/queue/framework costs. All149 rows, provenance, raw and slice hashes, signed errors and parameter records are retained. Root independently recomputed all aggregates and checked hashes/formulas in24 passing checks. Decode error near51% and service error near36% do not support a claim that the model or simulator is accurately calibrated.

## Remaining work and separately reviewable proposals

Canonical matched targets remain03 SG-burst-Ours (blocked by zero policy coverage) and08 Hotpot-burst-Ours (blocked by failed warmup/no formal data). No alternative case was substituted and no actual simulator rollout was attempted. The adapter source and synthetic tests exist, but full runtime parity is false: packed prefill, batch-dependent decode, link contention, exact simulated remaining time, event/reveal order and real router/admission/mutation semantics differ.

Three independent follow-ups require review before implementation or a new live window: (1) define a bounded KV-path readiness check because HTTP health can remain green after transport failure; (2) audit native TCP connection/reuse and blacklist recovery semantics, distinguishing the observed EADDRNOTAVAIL sequence from its unproven OS root cause; (3) test whether genuinely queued calls reveal metadata promptly and can be reordered, and define a calibration-only queue/coverage preflight. Healthy transport is a separate prerequisite for any future replay of the originally registered Hotpot pressure case. Do not require every idle fastpath call to invoke the planner or acquire a positive revision. These are separate runtime and method questions. Do not lower thresholds, force planning after seeing heldout results, weaken the native baseline, fit P/D scales to the149 heldout calls or change network/system settings under this completed window.

A future network observation plan should bound duration, request count and storage, recording peer/source socket counts and errors at first failure, existing session lifecycle, P/D state and contemporaneous resource limits. A future activation proposal should show deterministic CPU before/after examples, verify the sign and ordering of core priority through the worker's low-numeric-first queue (not merely the presence of a priority field), and define an approved calibration-only gate before heldout replay. Neither proposal authorizes another run. The scheduler-gain and matched real/simulator objectives remain unachieved.

## Evidence paths and hashes

Shared root: `/home/fred/hexagent-rebuttal/artifacts/rebuttal-h20`.
New run (`RUN`): `live-r2w3-20261107-01a1b627` under that root. Old native profiles: `live-20261107-01a1b627`.

| Evidence under RUN unless stated | SHA256 or role |
| --- | --- |
| `actual-runtime-observations.json` | `d360d4e9575453c9dbeff880e09fcde49950087d1f17fe6ef8c749f70d57b375` |
| `reference-collection/workflow-references.jsonl` | `82fccd971a1e57eb121172eb0fced5f5732659c76338e0c12e4f22299dc7a1af` |
| `reference-collection/workflow-reference-contract.json` | `80dcc1d4647567c477f55840950b311b428e10b163f964bd20013dc3e8f24581` |
| `audit-metrics/qc-reference.json` | `10cf66d0e090b25ff5fa832eb04573c2c03ca4e157a0bb49a9fc1c59fa3fef7e` |
| `paired-pilot/run-plan.json` | `2a75a273be4696c6e529c127ec322dc5e1569ad276b91dae8414c8da47e79c78` |
| `paired-pilot/block-results.json`, continuation ledger and amendment | Complete attempted/not-run accounting; original block preserved |
| `audit-metrics/transport-failure-audit.json` | `cb1cffc0fa45916f254dfe5199a534fbbf14c53ccc187fd32f9a40d603cb455b` |
| `prestop-final-state.json`, `poststop-resource-release.json` | Failure/resource snapshots and successful resource release |
| `preflight/result.json`, `preflight/events.jsonl` | Live R2/W3 component proof |
| `analysis-service-prediction/results-v1/parameters.json` | `420f7569b96f0f2052c6d97e87891f15d4c282a9e9e7618ee6d89977608a2ce6` |
| `analysis-service-prediction/results-v1/results.json` | `1e2c8354ebedcf13789b836db4d2b77ecb28370fc40849a2631e67aa9d737ce6` |
| `analysis-service-prediction/results-v1/per-call.csv` and `.jsonl` | All149 calls, all errors retained |
| `final-analysis/activation-audit/audit.json` | `8b645a2d144146f74451fe7b1e3f929bb1a1b4d700c877fc33728cd992b6b801` |
| `final-analysis/activation-audit/audit.md` | `dea7d63935c74ba88fbc8c09c1b520efce34e0d63ee8351e8abf15d1c3a61928` |
| `final-analysis/` | Final8-case/per-WF ledger, packaged test log and independent arithmetic QC |
| `matched-sim-preparation/` | BLOCKED03/08 contracts/QC; no actual simulator output |

Exact executed drivers/tests/recipes are archived in Git; raw output remains outside Git. This report supersedes provisional reference conventions and pending-runtime statements in earlier audit/status checkpoints without rewriting their historical evidence.
