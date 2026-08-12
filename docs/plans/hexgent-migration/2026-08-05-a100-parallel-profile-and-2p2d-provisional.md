# A100 Parallel Profile and 2P+2D Provisional Replay

## Scope

This session used one 8x A100-SXM4-80GB RunPod node. Every GPU pair reports
`NV12`. Fixed workload shapes come from the H100/Llama-70B traces, while all
latencies and SLO denominators are remeasured with Llama-3.1-8B BF16 on A100.
Radix cache is disabled in both exclusive profiling and P-D serving.

## Parallel exclusive profiling

Three independent TP1 servers profiled ShareGPT, BFCL, and HotpotQA on GPUs
0, 1, and 2. After ShareGPT and HotpotQA completed, the first 635 valid BFCL
calls were retained and the remaining 578 calls were divided into three
disjoint canonical subsets. The final merge rejects duplicate, missing, extra,
or error rows and restores canonical workflow/call order.

Authoritative profile:

```text
/workspace/hexgent_e2e/a100_llama8b_bf16_profile_final_20260805/
```

Validated contents:

- ShareGPT: 100 workflows, 477 calls.
- BFCL: 100 workflows, 1,213 calls.
- HotpotQA: 20 workflows, 636 calls.
- Total: 2,326 calls; zero errors and zero input/output length mismatches.
- Model: `meta-llama/Llama-3.1-8B-Instruct`, BF16, TP1.
- GPU: A100-SXM4-80GB.
- Radix cache: disabled.

## Concurrent 2P+2D deployment

GPUs 4 and 5 are prefill workers; GPUs 6 and 7 are decode workers. The
deployment uses Mooncake intra-node NVLink, prefill `max_running_requests=32`,
decode `max_running_requests=128`, and disabled radix cache. Raw provisional
results and worker/router/planner/GPU telemetry are under:

```text
/workspace/hexgent_e2e/a100_2p2d_concurrent_20260805/
```

The launcher now accepts `PREFILL0_GPU`, `PREFILL1_GPU`, `DECODE0_GPU`, and
`DECODE1_GPU`, preserving the old `0/1/2/3` defaults. A provisional runner
records raw baseline/HexAgenT results without requiring an exclusive profile;
SLO summaries are generated only after the validated profile exists.

## Validity finding

At ShareGPT 3 workflows/s, all 477 HexAgenT-routed calls carried fallback
revisions. The planner endpoint was invoked, but no request remained in global
admission long enough to receive a published placement. Therefore that run is
not evidence of HexAgenT scheduling performance. The next step is a 3/6/9/12
workflow/s ShareGPT staircase. Acceptance requires non-zero planner revision
coverage as well as queue and GPU telemetry.

BFCL creates real admission pressure. Planner telemetry contained non-empty
admission batches and a representative one planned one admission in about
1.03 ms. BFCL planner revision coverage was 29.6%; HotpotQA coverage was only
0.94%.

The provisional P95 SLO ratios were:

| Trace | Baseline | HexAgenT | Planner coverage |
|---|---:|---:|---:|
| ShareGPT, 3 workflows/s | 1.386x | 1.397x | 0% |
| BFCL, 0.6 workflows/s | 20.884x | 25.060x | 29.6% |
| HotpotQA, 0.2 workflows/s | 1.371x | 1.410x | 0.94% |

These are debug results, not a positive HexAgenT result. For BFCL, calls with a
planner revision spent a mean 570 ms and P95 2.17 s in router admission. The
planner placement also favored prefill worker 1. For ShareGPT, a 6/9/12
workflows/s staircase still produced 0% planner coverage. At 12 workflows/s,
decode GPU utilization averaged 94.8% and 92.3%, while prefill averaged only
37.5% and 36.9%. This proves that prefill-only admission credit dispatches
requests through fallback even when decode is saturated. Admission must account
for decode backlog/capacity, and an admitted request must not wait seconds for a
plan.

The remote and local evidence archive is
`a100_llama8b_profile_and_2p2d_20260805.tar.gz`; SHA-256 is
`35efcff2a2a3d51e713803b8439b3ac0d8d30f81b54f3155f952bbc4501647e6`.

## Code and verification

Changed or added:

- `start_hexgent_pd_smoke_2p2d.sh`: explicit physical GPU assignment.
- `run_a100_llama8b_exclusive_profile.sh`: selected-trace execution through
  `TRACE_ORDER`, while retaining process-group cleanup.
- `run_2p2d_provisional_replay.sh`: raw replay and GPU/planner telemetry without
  premature SLO computation.
- Static launcher/runner tests in `test_hexgent_real_trace_collectors.py`.

Focused tests passed locally, including Bash syntax validation. Performance
results collected while profile servers shared the host CPU are provisional;
final baseline/HexAgenT comparisons should run without profilers and after the
decode-aware admission issue is corrected.
