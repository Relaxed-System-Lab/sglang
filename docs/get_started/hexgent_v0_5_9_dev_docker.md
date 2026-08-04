# Hexgent Development Docker on SGLang v0.5.9

This repo is pinned to the upstream SGLang release tag `v0.5.9` for the current Hexgent integration work.

## Beginner Baseline

Build the local development image directly from this checkout:

```bash
docker build -f docker/Dockerfile --build-arg BRANCH_TYPE=local -t sglang:v0.5.9-local .
```

Or use the included compose file:

```bash
docker compose -f docker/compose.local-v0.5.9.yaml build
docker compose -f docker/compose.local-v0.5.9.yaml up
```

This gives a beginner-friendly single-server SGLang image from the checked-out `v0.5.9` source instead of `lmsysorg/sglang:latest`.

## Why This Is the Right Baseline

- It is reproducible: the source tree is `v0.5.9`, not a moving `latest` tag.
- It builds from the local checkout, so Hexgent-related edits in this repo are included automatically.
- It matches the framework version that the current router and worker changes target.

## Router Image for PD / Hexgent Work

The PD router lives in `sgl-model-gateway`. Build it separately with:

```bash
docker build -f docker/gateway.Dockerfile -t sglang-router:v0.5.9-local .
```

For Hexgent development, the router is now self-contained by default because the current `pd_disaggregated` scheduler code is vendored into `sgl-model-gateway/bindings/python/src/sglang_router/hexgent_vendor/`.

Default self-contained launch:

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --hexgent-router \
  --prefill http://prefill-host:30001 8998 \
  --decode http://decode-host:30002
```

You can still override the vendored scheduler with an external repo for experiments:

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --hexgent-router \
  --hexgent-repo-path /workspace/Hexgent \
  --prefill http://prefill-host:30001 8998 \
  --decode http://decode-host:30002
```

## Current Scope

Current code in this tree already adds:

- a Python `--hexgent-router` mode in the gateway
- in-process loading of `pd_disaggregated.schedulers.PRPLSJointScheduler`
- a vendored Hexgent scheduler copy so the repo/docker are self-contained by default
- request-aware PD dispatch plumbing so prefill and decode can receive different `priority` values
- decode-side local waiting-queue sorting through the existing SGLang scheduler policy

The remaining work is the richer live-state bridge between SGLang workers and the in-process Hexgent planner.

For trace-driven experiments, see [hexgent_trace_replay.md](D:/Github/sglang/docs/get_started/hexgent_trace_replay.md).
