# HexGen Real Trace Collectors

Collect request-only or replayed JSONL traces for:

- ShareGPT: <https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered>
- BFCL-v3: <https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard>
- LATS HotpotQA: <https://github.com/lapisrocks/LanguageAgentTreeSearch/tree/main>

The collectors target an OpenAI-compatible SGLang endpoint and write one JSON object per model call.
For BFCL and LATS, the most faithful collection path is the recording proxy:
run the official evaluation driver against the proxy, and let the proxy forward
to SGLang while recording every model call.

## Output Schema

Each JSONL row includes at least:

```json
{
  "dataset": "sharegpt",
  "workflow_id": "example",
  "call_id": "example_turn_0",
  "turn_index": 0,
  "messages": [{"role": "user", "content": "Hi"}],
  "prompt": null,
  "response_text": "Hello",
  "input_tokens": 7,
  "output_tokens": 3,
  "ttft_ms": 25.4,
  "total_latency_ms": 120.8,
  "error": null,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "endpoint": "http://127.0.0.1:30000/v1",
  "created_at": "2026-06-22T00:00:00+00:00"
}
```

Dataset-specific rows may also include `metadata` and `tools`.

## RunPod Setup

Start an SGLang OpenAI-compatible server in one shell:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

Set common variables in a second shell:

```bash
cd /workspace/sglang
export ENDPOINT=http://127.0.0.1:30000/v1
export MODEL=meta-llama/Llama-3.1-8B-Instruct
export OPENAI_API_KEY=EMPTY
mkdir -p /workspace/trace_outputs
```

## ShareGPT

Install the optional Hugging Face dependency if collecting directly from HF:

```bash
pip install datasets
```

Collect and replay 100 ShareGPT workflows with streaming TTFT:

```bash
python -m scripts.hexgent_real_trace_collectors sharegpt \
  --max-workflows 100 \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --stream \
  --output /workspace/trace_outputs/sharegpt_replayed.jsonl
```

Request-only mode, useful before the endpoint is up:

```bash
python -m scripts.hexgent_real_trace_collectors sharegpt \
  --max-workflows 100 \
  --output /workspace/trace_outputs/sharegpt_requests.jsonl
```

Offline/local ShareGPT JSON or JSONL:

```bash
python -m scripts.hexgent_real_trace_collectors sharegpt \
  --input /workspace/data/sharegpt.jsonl \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --output /workspace/trace_outputs/sharegpt_local_replayed.jsonl
```

ShareGPT conversations are preserved as workflows. Each assistant turn becomes one replay call with the prior user/assistant messages as context. Unlike BFCL and LATS, ShareGPT already contains the original assistant response. The collector uses that dataset assistant response as the recorded `response_text`, estimates its token length, and sends the local model a fixed-length generation request with `max_tokens = min_tokens = expected_output_tokens` and `ignore_eos = true`. The generated text is used only to measure TTFT and total latency; it is not stored as the trace response.

## BFCL-v3

Preferred source: run BFCL/Gorilla's official generator against the recording proxy.
The BFCL README documents `bfcl generate` and stores generated model responses
under `result/MODEL_NAME/...`; the proxy captures the same calls as JSONL trace
rows while BFCL runs.

```bash
git clone https://github.com/ShishirPatil/gorilla.git /workspace/gorilla
cd /workspace/gorilla/berkeley-function-call-leaderboard
pip install -e .
```

Start the proxy in a separate shell:

```bash
cd /workspace/sglang
python -m scripts.hexgent_real_trace_collectors proxy \
  --listen-host 127.0.0.1 \
  --listen-port 18080 \
  --target-endpoint "$ENDPOINT" \
  --trace-dataset bfcl-v3 \
  --output /workspace/trace_outputs/bfcl_v3_proxy_replayed.jsonl
```

Then configure the BFCL model handler to use `http://127.0.0.1:18080/v1`
as its OpenAI-compatible endpoint and run the BFCL categories you need. If the
official BFCL driver is not available, use the direct local-file collector below.

Direct replay of a local BFCL file:

```bash
python -m scripts.hexgent_real_trace_collectors bfcl \
  --input /workspace/gorilla/berkeley-function-call-leaderboard/path/to/bfcl_v3.jsonl \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --stream \
  --max-tokens 256 \
  --output /workspace/trace_outputs/bfcl_v3_replayed.jsonl
```

Request-only fallback:

```bash
python -m scripts.hexgent_real_trace_collectors bfcl \
  --input /workspace/data/bfcl_v3.jsonl \
  --output /workspace/trace_outputs/bfcl_v3_requests.jsonl
```

The direct collector normalizes BFCL function definitions into OpenAI `tools`
entries so SGLang receives tool-call prompts in an OpenAI-compatible shape.

## LATS HotpotQA

Preferred source: run the LATS HotPotQA experiment against the recording proxy.

```bash
git clone https://github.com/lapisrocks/LanguageAgentTreeSearch.git /workspace/LanguageAgentTreeSearch
cd /workspace/LanguageAgentTreeSearch/hotpot
pip install -r requirements.txt
```

Start the proxy in another shell:

```bash
cd /workspace/sglang
python -m scripts.hexgent_real_trace_collectors proxy \
  --listen-host 127.0.0.1 \
  --listen-port 18081 \
  --target-endpoint "$ENDPOINT" \
  --trace-dataset lats-hotpotqa \
  --output /workspace/trace_outputs/lats_hotpotqa_proxy_replayed.jsonl
```

Then set the LATS OpenAI base URL to `http://127.0.0.1:18081/v1` using the
variable supported by your installed OpenAI client or a small local patch, and
run `sh lats.sh`. If you cannot run the official LATS loop, direct replay of
local HotpotQA records is available:

LATS fanout can be recorded in two modes:

- `HEXGENT_RECORD_N_CHOICES=1` keeps the original `n_generate_sample > 1` batched request and lets the proxy split the returned choices into logical rows. This is the preferred service-profile mode because the shared prompt is prefilled once while each returned choice still becomes one logical fanout sibling with its own response text and output length.
- `HEXGENT_SPLIT_N_REQUESTS=1` sends one physical OpenAI request per generated trajectory. This is useful only when you explicitly need separately timed sibling requests; it is slower and should not be the default profile-table path.

For final workload traces, collect at least 20 HotpotQA workflows and keep `n_generate_sample=5` so each workflow contains real fanout groups. The suite can cap recursive LATS rollout with `LATS_ROLLOUT_MAX_DEPTH`; the current profiling default is `3`, which preserves multi-step tree search while avoiding model-dependent runaway expansion.

```bash
python -m scripts.hexgent_real_trace_collectors lats-hotpotqa \
  --input /workspace/LanguageAgentTreeSearch/hotpot/path/to/hotpotqa.jsonl \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --stream \
  --max-tokens 1024 \
  --output /workspace/trace_outputs/lats_hotpotqa_replayed.jsonl
```

Fallback mode when the LATS repo/data is unavailable: provide HotpotQA-style JSON/JSONL records with `id` or `_id`, `question`, optional `context`, and optional `answer`. The collector builds a ReAct/LATS-style prompt with `Thought:`, `Action:`, and `Answer:` instructions.

```bash
python -m scripts.hexgent_real_trace_collectors lats-hotpotqa \
  --input /workspace/data/hotpotqa_fallback.jsonl \
  --endpoint "$ENDPOINT" \
  --model "$MODEL" \
  --output /workspace/trace_outputs/lats_hotpotqa_fallback_replayed.jsonl
```

## Notes

- `input_tokens` and `output_tokens` use endpoint `usage` values when available.
- Request-only mode estimates `input_tokens` with a simple character heuristic and leaves response fields empty.
- Streaming mode requests `stream_options={"include_usage": true}` and records `ttft_ms` from the first streamed content delta.
- `--max-tokens` is optional but recommended for reproducible replay measurements.
- Errors are captured per row in `error` so long jobs can continue after individual request failures.
