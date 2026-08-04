# Llama-8B H100 4P+4D Fan-Out Replay Report

## Scope

This report records the controlled replay completed on 2026-08-03. It is a
diagnostic run, not a final performance claim: the six workloads were run once
in `baseline,hexgent` order, so hardware and process-order variance remain.

- Hardware: one RunPod host with eight H100 80 GB SXM GPUs.
- Serving topology: four TP1 prefill workers on GPU 0--3 and four TP1 decode
  workers on GPU 4--7.
- Model: `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8` checkpoint. Worker logs
  report BF16 runtime tensors and approximately 8.54 GB model memory per GPU.
- Trace replay: ten workflows each for ShareGPT, BFCL-v3, and LATS HotpotQA.
  Calls preserve trace dependencies and use their fixed input and output token
  lengths. Arrival rates are 6.0, 1.2, and 0.4 workflows/s respectively.
- Local capacities: prefill `max_running_requests=32` and
  `prefill_max_requests=32`; decode `max_running_requests=128`. These limits
  did not impose the earlier artificial four-request batching cap.
- P-D backend: Mooncake with the source-built intra-node NVLink overlay.

The complete raw artifact is
`real_trace_profiles/e2e/hexgent_h100_4p4d_fanout_parentpath_4p4d_20260803_r2.tar.gz`.
The persistent RunPod directory is
`/workspace/hexgent_e2e/llama8b_4p4d_fanout_20260803/results/fanout_parentpath_4p4d_20260803_r2`.

## SLO Definition

Each workflow SLO is the longest dependency path of its calls' isolated,
model-matched 4P+4D service latencies. The denominator comes from 456 unique,
zero-error rows in
`/workspace/hexgent_e2e/slo_profiles/llama8b_h100_4p4d/exclusive_profile.jsonl`.
It does not use the obsolete latency hint embedded in a trace and does not sum
parallel fan-out branches. `SLO ratio = workflow end-to-end latency / this DAG
critical-path SLO`.

## Results

All 912 replayed calls completed without an error.

| Trace | Scheduler | Mean ratio | P95 ratio | Mean workflow latency | P95 workflow latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | Baseline | 1.093 | 1.283 | 6,881 ms | 9,849 ms |
| ShareGPT | HexAgenT | 1.178 | 1.996 | 6,997 ms | 10,084 ms |
| BFCL-v3 | Baseline | 1.309 | 1.510 | 3,455 ms | 6,202 ms |
| BFCL-v3 | HexAgenT | 1.205 | 1.272 | 3,222 ms | 5,940 ms |
| HotpotQA | Baseline | 1.130 | 1.169 | 6,590 ms | 8,675 ms |
| HotpotQA | HexAgenT | 1.195 | 1.244 | 6,960 ms | 8,995 ms |

HexAgenT selected materially different worker pairs from the baseline: the
same pair was selected for only 1/44 ShareGPT calls, 5/93 BFCL calls, and
23/319 HotpotQA calls. Thus the result is not caused by both paths silently
using identical routing. In this homogeneous cluster, however, worker-pair
changes alone have limited scope for improvement.

## Diagnosed Bottleneck

The primary regression is the router's synchronous admission protocol, not
snapshot collection:

| Trace | HexAgenT admission wait P50 / P95 | Snapshot fetch P95 |
| --- | ---: | ---: |
| ShareGPT | 33.2 / 56.2 ms | 0.31 ms |
| BFCL-v3 | 17.4 / 28.5 ms | 0.19 ms |
| HotpotQA | 46.2 / 67.2 ms | 0.32 ms |

The current Rust router enqueues a request and then waits on sidecar
`/wait_admission` before creating bootstrap and dispatching. The planner's
batch solve itself is typically short, but the request is nevertheless held
until an actor cycle publishes a `dispatch_ready` result. HotpotQA has
multi-stage dependency paths, so these waits accumulate across its critical
path. In this run the per-call HotpotQA latency delta was +35.3 ms on average,
while its admission wait median was +46.2 ms.

The planner was updated in this run to identify already revealed siblings by
their common `parent_call_ids`, rather than requiring the future-looking
`child_count` hint. Its current-call critical-path pressure now includes the
longest revealed parent path. This is the correct online fan-out signal, but
it cannot offset the router's per-call synchronous wait.

## KV Transfer Evidence

The workers initialize a P2P handshake and log `Using Intra-Node NVLink
transport`. Decode request telemetry reports end-to-end `transfer_duration`
samples of approximately 28--96 ms for non-warmup requests. These are request
pipeline durations, not CUDA-copy timings: they include producer/consumer
coordination and queueing in addition to the bulk copy.

The same worker logs also contain `Unsupported memory type` messages from
`intranode_nvlink_transport.cpp` during transfer-engine initialization. They
occur before request execution and no replay request failed, but the current
logs do not provide per-request transferred bytes or CUDA events. Therefore
this run does **not** validate a raw NVLink bandwidth value and must not treat
the old planner default `10 ms + payload / 7 GB/s` as a physical-link model.

## Required Follow-Up

1. Replace synchronous `/wait_admission` in the router with a revisioned global
   admission queue. A request must always have a native PD fallback decision;
   the router should apply the latest published planner decision only when it
   receives prefill admission capacity and creates bootstrap. Planner solving
   must never add a per-call serving wait.
2. Preserve the new parent-edge metadata in worker snapshots. The router now
   injects `hexgent_parent_call_ids`, and the snapshot parser reads it; the
   rebuilt router must be used in the next pod run.
3. Instrument the Mooncake sender and decode receiver with request ID, bytes,
   queue-enter/ready timestamps, and CUDA events around the actual device copy.
   Fit payload-copy bandwidth, coordination overhead, and contention delay as
   separate planner terms.
4. Re-run at least both `baseline,hexgent` and `hexgent,baseline` orders after
   the non-blocking admission change. Report P95/P99 workflow SLO ratio before
   claiming a scheduling benefit.
