# H20 pilot: exact executed sources (2026-11-07)

These files archive the sources executed for the bounded H20 pilot and its CPU diagnostic. `executed-source-manifest.json` binds every copied file to its original path and SHA256. The copies retain the executed bytes; they do not contain fixes made after observing results. Runtime source was `08885ec27573ee1807b822e786ce95ebd06ec884`. The compiled gateway used Rust revision `4307bf73b49f7ba02d643f53d4659e8fd90da51a`; all tracked gateway source bytes are identical at the runtime revision.

The result and limitations are recorded in [the final report](../../../docs/hexgent/rebuttal_h20_results_20261107.md). The eight-case block was incomplete: SG Ours requests took the credit fastpath, Hotpot burst encountered a transport failure, and Hotpot low was skipped by the fixed budget rule. These sources do not establish canonical scheduler gains or a matched simulator result.

## File roles

- `h20_run_support.py`: actual workflow replay API, incremental raw persistence and completeness/usage gates.
- `collect_workflow_references.py`: one mixed calibration burst for load selection, then eight actual whole-workflow-alone references; each workflow retains its internal DAG fanout.
- `run_paired_pilot.py`: frozen eight-case plan, same per-cell arrival vectors, controller ownership, uniform warmup, idle/capacity gates, per-run log slices and failure-preserving summaries. It writes a plan unless `--execute` is supplied.
- `run_budgeted_block.py`: executed top-level wrapper applying the predeclared pair budget rule. It stopped at the first insufficient pair.
- `continue_unattempted_pairs.py`: separately reviewed one-shot amendment; skip an insufficient pair and consider only later registered, wholly unattempted pairs under the unchanged deadline. It preserves the old block and writes an append-only decision ledger.
- `capture_sidecar_event.py` and `run_smoke_validation.py`: preflight body capture and one real model call proving R2 timing and W3 HTTP dispatch. Formal runs use the ordinary sidecar, not the capture wrapper.
- `aggregate_final.py`: post-runtime CPU-only recomputation of the complete eight-case/32-workflow ledger, with warmup/formal boundaries and failure-preserving metrics.
- `service_prediction_diagnostic.py`: CPU-only heldout per-call stage/service prediction analysis against the earlier a65 serial native measurements. It does not run a scheduler or fit P/D scales or transfer parameters.
- `commands.json`, `smoke-request.json`, `prospective-pilot-protocol.json`, `reference-r2-container-plan.json`: exact historical preflight/launch recipe and protocol. Plan fields such as `not_started` describe creation time, not the final runtime outcome.

`run_smoke_validation.py` and `run_budgeted_block.py` execute at module top level. Do not import them during checks. Their exact historical mount paths and environment dependencies are preserved intentionally. All live operational entrypoints require a separately authorized resource window; the original runtime has ended and its container was removed.

## CPU verification

The scripts and adjacent tests are self-contained apart from the actual source-root replay module, `aiohttp`, `pytest`, and the two optional FastAPI capture tests. The existing H20 simulator environment provides the first three. From the source root:

```bash
export H20_REPLAY_TEST_SOURCE="$PWD"
export PYTHONPATH="$PWD/scripts/hexgent_real_trace_collectors/h20_pilot"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
/home/fred/hexagent-rebuttal/venvs/sim/bin/python -m pytest -q -p no:cacheprovider \
  scripts/hexgent_real_trace_collectors/h20_pilot/test_*.py -k 'not fastapi'
```

The final archive passed 69 tests with 2 FastAPI tests deselected in the existing CPU environment. All 5 capture tests, including those 2 FastAPI tests, passed earlier in the actual serving image. The first archive subset passed61 tests; adding the final ledger's8 tests produced69. The already validated R2/W3 Python suite passed 397 tests and the rebuilt gateway passed 39 targeted Rust tests; their raw logs remain in shared artifacts.

## Historical runtime layout

The archived recipe mounted source read-only at `/src`, the existing model/cache read-only, shared artifacts read-write at `/artifacts`, and sanitized inputs read-only at `/inputs`. The scripts were copied to `/artifacts/r2-runtime-preparation`; source includes their exact hashes. Operational wrappers also read `/artifacts/reference-r2-container-launch.json`, `RUN_DIR/status.json` and explicit environment files. They must not be repurposed by refreshing old deadlines, IDs or hashes in place. A future experiment needs a new versioned plan and authorization.

The actual launcher is the existing `../start_h20_tp2_reference.sh`. The recipe requires the rebuilt ELF hash `f52f2c1e02f32ca37effb60b3d51d306b1bb92f1f4b2ae8c3f13211b931e4831`, existing Llama-3.1-70B BF16 checkpoint, physical H20 GPUs 4/5 (P TP2) and 6/7 (D TP2), Mooncake TCP, disabled radix cache/CUDA graph, and the frozen calibration transfer parameters. Hardware rates are nominal catalog values, not measured peak throughput.

Raw requests, JSONL results, logs, socket snapshots, runtime manifests and per-call diagnostic outputs stay outside Git under `/home/fred/hexagent-rebuttal/artifacts/rebuttal-h20`. The final report supplies exact paths and evidence hashes.

## CPU-only result regeneration

With the preserved external raw run available, `aggregate_final.py --run-root <RUN> --source-root <SOURCE_ROOT> --output-dir <FRESH_OUTPUT>` reconstructs the8-case/32-workflow ledger, validates the saved summaries and writes CSV/JSON/Markdown. It rejects existing result filenames to preserve prior evidence. Its executed hash and tests are in the manifest.

`service_prediction_diagnostic.py` accepts `--source-root`, `--old-run`, `--trace-dir`, `--approved-transfer` and `--output-dir`; use the original a65 profile run and approved frozen transfer JSON, not the R2/W3 pressure run as its old-run input. The original argv/resource limits are preserved externally in `RUN/analysis-service-prediction/execution-command.json`. The executed source hash is `dbc8f3bcea789f5223dc2df977c33d29bb9b71904b16937afaf2a33ae703d000`. Its original `used_in_this_predictor:false` metadata means no raw calibration-stage feature or fitted P/D scale; frozen transfer parameters still have the documented upstream68-call calibration source. The separately saved metadata clarification does not change the executed script or any numbers.

The activation audit is separate: only its successful inline probe output and source evidence were saved. A later standalone-probe/test save-and-run command was automatically rejected (`blocked by policy`), never executed or retried; no standalone activation probe/test is included or counted as passing.
