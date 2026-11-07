# Q1-real: explicit attention head dimension

Status: **accepted Q1-real**, commit
`502e70a04e93ee08d930cd6471463a4a43c0188d`. Date: 2026-11-07.

Base real revision: `33e34d5ccc` (accepted H1). Simulator architecture reference:
accepted Q1 commit `5cf51fdbc85293e1d3d4f7e3fed9fa0fcee5049a`.

## Evidence and root cause

The [official Qwen3-235B-A22B-Instruct-2507 config at immutable revision
ac9c66cc9b46af7306746a9250f23d47083d689e](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/raw/ac9c66cc9b46af7306746a9250f23d47083d689e/config.json)
was independently retrieved over HTTPS for this audit. Raw config SHA256:
`ad59a339665ee1c68cbff06ac72c9e59f5183ebfaf3698c98737e003a870e9e2`.
Its attention dimensions are residual width 4096, 64 query heads, 4 KV heads,
128 elements per head and 94 layers. The checkpoint declares BF16 weights.

Real `_ModelParams.head_dim` formerly derived 4096/64=64 even when the existing
model-config payload included `head_dim=128`. `_linear_work_components` also
used the residual width for Q/O and residual-width-derived KV width for K/V.
Both affect projected service; the incorrect head dimension additionally halves
the estimated attention KV bytes. The existing telemetry calibrator already
prioritizes the explicit config value, producing inconsistent planner and
calibration estimates for this configuration.

## Accepted semantics

Carry optional `attention_head_dim` through the existing model-config parser.
Use explicit config `head_dim` when supplied; preserve the derived dimension
when absent. No model-config precedence changes: worker config still precedes
request config, and the existing nested roofline forms remain supported.

For residual width h, head dimension d, query heads nq, KV heads nkv and TP p,
the per-rank attention projection shapes become:

```
Q: (h, nq*d/p)
K: (h, nkv*d/p)
V: (h, nkv*d/p)
O: (nq*d/p, h)
```

Validation boundary: TP greater than the number of KV heads requires a separate
check of runtime KV-head replication and physical per-rank storage. The
fractional `nkv/p` shape above describes this estimator's arithmetic; it does not
establish actual replicated weight/KV storage or model fit in that regime. The
explicit-head-dimension correction leaves this TP convention unchanged.

The attention and transfer expressions already consume `model.head_dim` and
therefore receive the same explicit dimension. Existing BF16/default precision,
FFN work, TP conventions, hardware inputs and transfer startup terms are
preserved. Hybrid cache keys already serialize complete model/roofline configs;
the candidate adds no cache layer or fallback framework.

## Minimal reproducer and affected quantities

Construct the existing model-config input with the official attention fields
and `head_dim=128`, then inspect `_model_params_from_payload(...).head_dim`.
The original returns 64; the candidate returns 128.

With BF16 KV and 1000 input tokens, full-group KV bytes are
`1000*2*94*4*128*2 = 192512000`. At the explicitly synthetic link rate 1e9
bytes/s and startup 5ms, the planner estimate changes from rounded 101ms to
198ms, matching the calibrator's model-config byte count. These values are
arithmetic checks, not measured transfer timings.

Affected quantities include projection FLOPs/weight traffic, attention work/KV
traffic, transfer estimates, and downstream planner decisions when those
estimates are used. Existing results retain their original implementation
version. A real Qwen deployment still requires measured model fit, all-expert
resident weight storage, actual TP/dtypes and runtime calibration.

The tests use Qwen attention geometry to isolate this correction. They retain
the estimator's existing dense FFN term and do not establish a full MoE model
or adopt the simulator's INT8/active-expert approximation for real execution.

## Validation

New test file: `test/unit/utils/test_hexgent_explicit_head_dim.py`.

- Red phase on unchanged estimator: **15 failed, 37 passed**. Failures cover
  four config entrypoints, Q/K/V/O work, attention work/KV traffic, TP1/2/4
  transfer/calibrator agreement, and changed-dimension cache results.
- Green phase: **52 passed** in the new file.
- Llama 8B/70B golden estimates were captured before implementation at the
  base revision for TP1/2/4 and A100/H100/H20 model parameters: 18 settings,
  each checked with absent and equivalent explicit dimensions (36 cases).
  Prefill, batch decode, transfer and packed prefill outputs are unchanged.
- Related regression: **153 passed**, with 13 existing FastAPI/Starlette
  deprecation warnings. JUnit: `artifacts/rebuttal-h20/q1-real-validation.xml`.

```powershell
& .venv-h20/Scripts/python.exe -m pytest test/unit/utils/test_hexgent_explicit_head_dim.py test/unit/utils/test_hexgent_planner.py test/unit/utils/test_hexgent_scheduler.py test/unit/utils/test_hexgent_service_profiler.py test/unit/utils/test_pd_telemetry_calibration.py -q --junitxml=artifacts/rebuttal-h20/q1-real-validation.xml
```

The validation above records the historical CPU candidate checks. H20 serving
measurements are tracked separately. Runtime changes are limited to
`python/sglang/hexgent/latency_estimator.py`; the coordinator accepted that file
and its regression test in commit `502e70a04e93ee08d930cd6471463a4a43c0188d`.
The independent candidate patch below remains provenance for the accepted change.

Patch: `artifacts/rebuttal-h20/q1-real-head-dim-candidate.patch`.
Patch SHA256: `dbbc01a1ed0d0eca8d5a74b5b30b960914348dadc9cf1d39bed619b5667b9cea`.
