# Hexgent Trace Conversion and Replay

The repo now includes a trace tool at [`python/sglang/hexgent/workflow_trace.py`](D:/Github/sglang/python/sglang/hexgent/workflow_trace.py).

It has three subcommands:

```bash
python -m sglang.hexgent.workflow_trace convert
python -m sglang.hexgent.workflow_trace build-profile
python -m sglang.hexgent.workflow_trace replay
```

## What This Solves

This toolchain is for trace-driven serving experiments where:

- workflows arrive according to a Poisson process
- each workflow contains one or more LLM calls
- each call carries exact input/output length information
- Hexgent reads estimated prefill/decode service times from a local lookup table

For service reproduction, the replayer sends:

- `conversation_id = workflow_id`
- `hexgent_call_id = call_id`
- `max_tokens = traced output length`
- `ignore_eos = true`

That last point is important: it makes SGLang generate up to the traced output budget instead of stopping early on EOS. This is the right mode when the experiment is about scheduling behavior rather than text quality.

## Upstream Trace Sources

The current Hexgent-side source mapping in [`trace_loader.py`](D:/HKUST/Hexgen2.5/Hexgent/utils/trace_loader.py) is:

- ShareGPT:
  - Hugging Face dataset: `anon8231489123/ShareGPT_Vicuna_unfiltered`
  - file used by Hexgent: `ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json`
- BFCL:
  - Hugging Face dataset: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`
  - Hexgent currently uses the BFCL v3 JSON files such as:
    - `BFCL_v3_exec_simple.json`
    - `BFCL_v3_exec_multiple.json`
    - `BFCL_v3_exec_parallel.json`
    - `BFCL_v3_exec_parallel_multiple.json`
    - `BFCL_v3_multi_turn_base.json`
    - `BFCL_v3_multi_turn_composite.json`
    - `BFCL_v3_multi_turn_long_context.json`
- LATS:
  - code/prompt source: `lapisrocks/LanguageAgentTreeSearch`
  - current Hexgent trace derivation source: Hugging Face dataset `hotpotqa/hotpot_qa`
  - file path used by Hexgent when downloading parquet directly: `distractor/validation-00000-of-00001.parquet`

Important caveat for BFCL:

- the BFCL dataset card explicitly says not to use generic Hugging Face `load_dataset(...)` for the main JSON files
- use line-by-line JSON loading or direct file download instead
- this matches your current `TraceLoader` behavior

Important caveat for LATS:

- the public LATS repo is the algorithm/code source, not a complete hosted trace corpus
- your current Hexgent loader builds LATS-style workflow trees from HotpotQA rows
- so for reproducible experiments, treat `hotpotqa/hotpot_qa` as the raw data source and `LanguageAgentTreeSearch` as the generation logic reference

## Step 1: Convert a Raw Trace

Supported input formats:

- `sharegpt`
- `sharegpt_hf`
- `bfcl_hf`
- `hotpotqa_hf`
- `openai_jsonl`
- `call_jsonl`
- `workflow_jsonl`

Canonical output is one workflow per JSONL line:

```json
{
  "workflow_id": "wf-1",
  "workflow_type": "sharegpt",
  "arrival_offset": null,
  "metadata": {},
  "calls": [
    {
      "call_id": "wf-1-call-0",
      "parents": [],
      "request": {
        "messages": [
          {"role": "user", "content": "hello"}
        ]
      },
      "input_len": 16,
      "output_len": 32,
      "planner_hints": {},
      "metadata": {}
    }
  ]
}
```

Example conversions:

```bash
python -m sglang.hexgent.workflow_trace convert \
  --input-path /data/sharegpt.json \
  --input-format sharegpt \
  --tokenizer-path meta-llama/Llama-3.1-8B-Instruct \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output-path /data/sharegpt_workflows.jsonl
```

Directly from the ShareGPT Hugging Face source:

```bash
python -m sglang.hexgent.workflow_trace convert \
  --input-format sharegpt_hf \
  --max-workflows 200 \
  --output-path /data/sharegpt_workflows.jsonl
```

Directly from the BFCL v3 Hugging Face files:

```bash
python -m sglang.hexgent.workflow_trace convert \
  --input-format bfcl_hf \
  --bfcl-files \
    BFCL_v3_exec_simple.json \
    BFCL_v3_exec_multiple.json \
    BFCL_v3_exec_parallel.json \
    BFCL_v3_exec_parallel_multiple.json \
    BFCL_v3_multi_turn_base.json \
    BFCL_v3_multi_turn_composite.json \
    BFCL_v3_multi_turn_long_context.json \
  --max-workflows 200 \
  --output-path /data/bfcl_workflows.jsonl
```

Directly from HotpotQA, converting rows into LATS-style workflows:

```bash
python -m sglang.hexgent.workflow_trace convert \
  --input-format hotpotqa_hf \
  --hotpot-config distractor \
  --hotpot-split validation \
  --max-workflows 200 \
  --output-path /data/lats_hotpot_workflows.jsonl
```

```bash
python -m sglang.hexgent.workflow_trace convert \
  --input-path /data/bfcl_calls.jsonl \
  --input-format call_jsonl \
  --output-path /data/bfcl_workflows.jsonl
```

`call_jsonl` is the easiest adapter format for BFCL-v3 and LATS-derived traces. Each line should already contain:

- `workflow_id`
- `call_id`
- `parents`
- `request`
- `input_len`
- `output_len`

## Step 2: Build a Service Profile Table

The profile table is a JSONL lookup keyed by:

- `model`
- `worker_class`
- `input_len`
- `output_len`

Each row stores:

- `estimated_prefill_time`
- `estimated_decode_time`
- `estimated_transfer_time`

Example row:

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "worker_class": "A100",
  "input_len": 512,
  "output_len": 128,
  "estimated_prefill_time": 0.021,
  "estimated_decode_time": 0.154,
  "estimated_transfer_time": 0.003
}
```

Normalize a raw profile file into JSONL with:

```bash
python -m sglang.hexgent.workflow_trace build-profile \
  --input-path /data/profile_rows.json \
  --output-path /data/profile_rows.jsonl
```

During replay, the router-facing request body gets these values under `hexgent`, so the in-process planner can read them directly.

## Step 2A: Measure Real Per-Call Service Tables

Use this path when you want measured serving stats for another scheduler, such as
`D:\Github\LLM4Scheduling\examples\agentic_pd_scheduling_optimizer`, instead of
router-side replay hints.

The measurement script runs trace calls one by one against a running OpenAI-compatible
SGLang endpoint. It does not use the Hexgent router, dispatcher, or scheduler. Point it
at one GPU class at a time and tag the result with `--worker-class`.

The output is intentionally split into two tables:

- prefill table rows keyed by `model`, `worker_class`, `input_len`, `output_len`, and `call_id`
- decode table rows keyed by the same fields

This matches the optimizer shape in `examples/agentic_pd_scheduling_optimizer`, where
prefill and decode latency are modeled separately.

Example for an A100 endpoint:

```bash
python python/sglang/hexgent/service_profiler.py profile \
  --trace-path /workspace/data/sharegpt_workflows.jsonl \
  --endpoint-url http://127.0.0.1:31000/v1/chat/completions \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --worker-class A100_80G \
  --phase both \
  --prefill-output /workspace/profiles/prefill_A100_80G.jsonl \
  --decode-output /workspace/profiles/decode_A100_80G.jsonl \
  --raw-output /workspace/profiles/raw_A100_80G.jsonl \
  --trials 3
```

Repeat the same command for each GPU class, changing only `--endpoint-url`,
`--worker-class`, and output filenames:

```bash
python python/sglang/hexgent/service_profiler.py profile \
  --trace-path /workspace/data/sharegpt_workflows.jsonl \
  --endpoint-url http://127.0.0.1:32000/v1/chat/completions \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --worker-class H20 \
  --phase both \
  --prefill-output /workspace/profiles/prefill_H20.jsonl \
  --decode-output /workspace/profiles/decode_H20.jsonl \
  --raw-output /workspace/profiles/raw_H20.jsonl \
  --trials 3
```

Combine the tables and fit the optimizer-compatible latency database:

```bash
cat /workspace/profiles/prefill_*.jsonl > /workspace/profiles/prefill_all.jsonl
cat /workspace/profiles/decode_*.jsonl > /workspace/profiles/decode_all.jsonl

python python/sglang/hexgent/service_profiler.py fit-latency-db \
  --prefill-path /workspace/profiles/prefill_all.jsonl \
  --decode-path /workspace/profiles/decode_all.jsonl \
  --output-path /workspace/profiles/latency_db_measured.json \
  --hot-prefill-scale 1.0
```

Then replace the simulated database in:

```text
D:\Github\LLM4Scheduling\examples\agentic_pd_scheduling_optimizer\data\latency_db.json
```

Timing caveat: the prefill table uses streaming time-to-first-token as the practical
prefill proxy. In a PD setup this includes prefill, transfer/handoff, router or HTTP
overhead, and the first decode step. The decode table is based on inter-token streaming
latencies and is closer to decode iteration time. If you need pure internal prefill and
decode kernel timings, instrument the worker internals instead of measuring through the
HTTP streaming API.

## Step 3: Replay Workflows Against the Hexgent Router

Example:

```bash
python -m sglang.hexgent.workflow_trace replay \
  --trace-path /data/sharegpt_workflows.jsonl \
  --router-url http://127.0.0.1:31000 \
  --profile-path /data/profile_rows.jsonl \
  --arrival-rate 4.0 \
  --seed 42 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --worker-class A100 \
  --output-path /data/replay_results.jsonl
```

If your local shell does not have the full `sglang` Python dependencies installed yet, the same tool can be invoked directly by file path:

```bash
python python/sglang/hexgent/workflow_trace.py convert ...
python python/sglang/hexgent/workflow_trace.py replay ...
```

Behavior:

- workflow arrivals are sampled from a Poisson process when `--arrival-rate` is set
- if a workflow has DAG structure, each call waits until all `parents` have completed
- requests are sent to `/v1/chat/completions` by default
- planner hints are merged from the canonical trace and the local profile table

If you want to use stored workflow arrival offsets instead of Poisson arrivals, omit `--arrival-rate`.

## RunPod Deployment Shape

Do not use the stock RunPod Hub `worker-sglang` image for this experiment. It does not contain the Hexgent router path or the worker-side queue mutation/snapshot changes from this repo.

The correct deployment path is:

1. build custom worker and router images from this repo
2. push them to a registry you control
3. create a RunPod Pod with enough GPUs
4. start multiple worker processes and one router process inside the Pod
5. convert traces and replay them against the router

### Step 0: Choose the Pod size

For `Llama-3.1-70B` with `TP=4`:

- one logical model instance uses `4` GPUs
- one prefill worker instance uses `4` GPUs
- one decode worker instance uses `4` GPUs

Minimum functional validation:

- `1` prefill worker
- `1` decode worker
- total `8` GPUs

Smallest setup that actually exercises routing choices:

- `2` prefill workers
- `2` decode workers
- total `16` GPUs

Recommended first real experiment:

- borrow a `16`-GPU Pod
- map GPUs as:
  - `0,1,2,3` -> prefill worker A
  - `4,5,6,7` -> prefill worker B
  - `8,9,10,11` -> decode worker A
  - `12,13,14,15` -> decode worker B

### Step 1: Build the custom images locally

From the repo root:

```bash
docker build -f docker/Dockerfile --build-arg BRANCH_TYPE=local -t fredpeng233/sglang-hexgent-worker:v0.5.9 .
docker build -f docker/gateway.Dockerfile -t fredpeng233/sglang-hexgent-router:v0.5.9 .
```

You need both images:

- worker image for SGLang model workers
- router image for the Hexgent-enabled PD router

### Docker build interruption behavior

Docker does not provide a true pause-and-resume mechanism for `docker build`.

What you can do safely:

- press `Ctrl-C` to cancel the build
- switch your network
- rerun the same `docker build` command

What happens after restart:

- layers that already finished normally are usually reused from the local Docker cache
- the layer that was still running when you interrupted the build usually has to be rebuilt
- if the interrupted step was downloading packages, that step may start over

Practical guidance:

- if you expect to change network soon, it is better to cancel and restart than to let the build fail unpredictably mid-download
- the worker image build requires network access for package and dependency downloads
- the router image also needs network access, but is usually smaller and faster to rebuild

### Step 2: Push to a registry

Example with Docker Hub:

```bash
docker login
docker push fredpeng233/sglang-hexgent-worker:v0.5.9
docker push fredpeng233/sglang-hexgent-router:v0.5.9
```

You can also use GHCR or another private/public registry that RunPod can pull from.

### Step 3: Create the RunPod Pod

On RunPod:

1. create a Pod, not a Serverless endpoint
2. use the worker image as the Pod image
3. allocate the GPU count chosen in Step 0
4. attach enough disk for:
   - model weights
   - converted workflow traces
   - profile tables
   - replay outputs and logs

### Step 4: Pull the router image inside the Pod

If the Pod launches from the worker image, also pull the router image:

```bash
docker pull fredpeng233/sglang-hexgent-router:v0.5.9
```

### Step 5: Start the worker processes

Always pin workers explicitly with `CUDA_VISIBLE_DEVICES`.

Example for `70B`, `TP=4`, `2P + 2D`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --tp-size 4 \
  --host 0.0.0.0 \
  --port 30001 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port 8998 \
  --enable-priority-scheduling \
  --schedule-policy fcfs
```

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --tp-size 4 \
  --host 0.0.0.0 \
  --port 30002 \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port 8999 \
  --enable-priority-scheduling \
  --schedule-policy fcfs
```

```bash
CUDA_VISIBLE_DEVICES=8,9,10,11 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --tp-size 4 \
  --host 0.0.0.0 \
  --port 30011 \
  --disaggregation-mode decode \
  --enable-priority-scheduling \
  --schedule-policy fcfs
```

```bash
CUDA_VISIBLE_DEVICES=12,13,14,15 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --tp-size 4 \
  --host 0.0.0.0 \
  --port 30012 \
  --disaggregation-mode decode \
  --enable-priority-scheduling \
  --schedule-policy fcfs
```

For the minimal `8`-GPU setup, start only one prefill worker and one decode worker.

### Step 6: Start the Hexgent router

Run the router in another shell or container:

```bash
python -m sglang_router.launch_router \
  --host 0.0.0.0 \
  --port 31000 \
  --pd-disaggregation \
  --hexgent-router \
  --hexgent-scheduler prp_ls_joint \
  --hexgent-mode async \
  --prefill http://127.0.0.1:30001 8998 \
  --prefill http://127.0.0.1:30002 8999 \
  --decode http://127.0.0.1:30011 \
  --decode http://127.0.0.1:30012
```

### Step 7: Convert traces inside the Pod

Convert directly from the upstream datasets:

```bash
python python/sglang/hexgent/workflow_trace.py convert \
  --input-format sharegpt_hf \
  --max-workflows 200 \
  --output-path /workspace/data/sharegpt_workflows.jsonl
```

```bash
python python/sglang/hexgent/workflow_trace.py convert \
  --input-format bfcl_hf \
  --max-workflows 200 \
  --output-path /workspace/data/bfcl_workflows.jsonl
```

```bash
python python/sglang/hexgent/workflow_trace.py convert \
  --input-format hotpotqa_hf \
  --max-workflows 200 \
  --output-path /workspace/data/lats_workflows.jsonl
```

### Step 8: Build or copy the service profile table

If you already have raw profile rows:

```bash
python python/sglang/hexgent/workflow_trace.py build-profile \
  --input-path /workspace/data/profile_rows_raw.json \
  --output-path /workspace/data/profile_rows.jsonl
```

For heterogeneous experiments, make sure the table includes separate worker classes such as `A100` and `H20`.

### Step 9: Run the replay experiment

Example:

```bash
python python/sglang/hexgent/workflow_trace.py replay \
  --trace-path /workspace/data/sharegpt_workflows.jsonl \
  --router-url http://127.0.0.1:31000 \
  --profile-path /workspace/data/profile_rows.jsonl \
  --arrival-rate 5.0 \
  --seed 42 \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --worker-class A100 \
  --output-path /workspace/data/sharegpt_replay_results.jsonl
```

Repeat with:

- `--arrival-rate 5.0`
- `--arrival-rate 10.0`
- `--arrival-rate 20.0`

to cover the three planned workload regimes.

### Step 10: Collect outputs

Keep:

- replay result JSONL
- router logs
- worker logs
- optional `/hexgent/pd_snapshot` outputs during execution

That is the minimum data needed for workflow-level completion analysis and debugging.

## Operating the Pod

### SSH into the RunPod Pod

Yes, you will normally use SSH to access the Pod. RunPod gives you a command like:

```bash
ssh root@YOUR_RUNPOD_IP -p YOUR_PORT -i ~/.ssh/id_ed25519
```

This is only for remote access to the machine. It is not part of SGLang itself.

### Use `tmux` instead of many SSH windows

For this workflow, `tmux` is strongly recommended.

A practical layout is:

- window 1: prefill workers
- window 2: decode workers
- window 3: router
- window 4: trace conversion and replay
- window 5: monitoring and log checks

Example:

```bash
tmux new -s hexgent
```

Inside `tmux`, create windows with:

```bash
Ctrl-b c
```

Switch windows with:

```bash
Ctrl-b 0
Ctrl-b 1
Ctrl-b 2
```

Detach without killing processes:

```bash
Ctrl-b d
```

Reconnect later:

```bash
tmux attach -t hexgent
```

### Do I need two terminals?

Yes, at minimum.

One terminal is for long-running services:

- prefill workers
- decode workers
- router

Another terminal is for:

- trace conversion
- replay commands
- `curl` checks
- log inspection

If you use `tmux`, this is much easier than opening many separate SSH sessions.

## Updating Code Without Rebuilding the Image

You do not need to rebuild the image for every Python code fix.

For most debugging iterations, it is faster to:

1. fix the code locally
2. copy the changed files to the Pod
3. restart only the affected process

This works especially well for:

- `python/sglang/...`
- `sgl-model-gateway/bindings/python/...`
- `python/sglang/hexgent/workflow_trace.py`
- `python/sglang/hexgent/service_profiler.py`
- docs or scripts

### Copy updated files to the Pod

You can use `scp` for a few files:

```bash
scp -P YOUR_PORT -i ~/.ssh/id_ed25519 \
  python/sglang/hexgent/workflow_trace.py \
  python/sglang/hexgent/service_profiler.py \
  root@YOUR_RUNPOD_IP:/workspace/sglang/python/sglang/hexgent/
```

Or `rsync` for larger syncs:

```bash
rsync -avz -e "ssh -p YOUR_PORT -i ~/.ssh/id_ed25519" \
  python/sglang/srt/ \
  root@YOUR_RUNPOD_IP:/workspace/sglang/python/sglang/srt/
```

### After syncing code

Restart only what changed:

- if you changed worker-side Python code:
  restart the affected worker processes
- if you changed router-side Python code:
  restart the router process
- if you changed replay tooling:
  no need to restart workers or router; just rerun the replay command

### When hot-patching is enough

Use direct file sync instead of rebuilding when:

- you changed Python logic
- you changed scheduling policy code
- you changed trace conversion or replay code
- you changed docs or scripts

### When you should rebuild the image

Rebuild when:

- you changed Dockerfiles
- you added or changed system packages
- you changed Python dependencies that are installed into the image
- you changed native build steps
- you want a clean reproducible snapshot for a final experiment run

## Practical Debugging Workflow

Recommended iteration loop:

1. Build and push the base images once.
2. Start the Pod from those images.
3. During debugging, sync Python changes directly to the Pod.
4. Restart only the changed services.
5. Rebuild the image only after the setup is stable or when dependencies change.
