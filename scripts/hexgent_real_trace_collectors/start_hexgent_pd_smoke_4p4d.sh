#!/usr/bin/env bash
set -euo pipefail

# Start four TP1 prefill workers on GPU0-3 and four TP1 decode workers on
# GPU4-7. Requests remain in HexAgenT global admission until a selected
# prefill worker has admission credit; bootstrap IDs are created afterwards.

VENV_DIR="${HEXGENT_VENV_DIR:-/workspace/hexgent_sglang_venv}"
SRC_DIR="${HEXGENT_SRC_DIR:-/workspace/sglang}"
RUN_DIR="${HEXGENT_RUN_DIR:-/workspace/hexgent_e2e/run_4p4d}"
LOG_DIR="${HEXGENT_LOG_DIR:-/workspace/hexgent_e2e/logs_4p4d}"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-hexgent-smoke-model}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
# Admission is controlled by the global queue below. Keep enough local
# headroom for prefill batching and a long-lived decode continuous batch.
PREFILL_MAX_RUNNING_REQUESTS="${PREFILL_MAX_RUNNING_REQUESTS:-32}"
DECODE_MAX_RUNNING_REQUESTS="${DECODE_MAX_RUNNING_REQUESTS:-128}"
# Keep the running pool large for continuous batching, but avoid starting a
# large number of TCP KV transfers concurrently. One transfer per decode
# worker matches the old good BFCL run and prevents transfer-bandwidth collapse.
DECODE_MAX_INFLIGHT_TRANSFERS="${DECODE_MAX_INFLIGHT_TRANSFERS:-1}"
# This controls the size of a single prefill batch, independently of the
# request pool retained by the worker.
PREFILL_MAX_BATCH_REQUESTS="${PREFILL_MAX_BATCH_REQUESTS:-32}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"

PLANNER_PORT="${PLANNER_PORT:-30999}"
PREFILL_PORT_BASE="${PREFILL_PORT_BASE:-31000}"
DECODE_PORT_BASE="${DECODE_PORT_BASE:-32000}"
BOOTSTRAP_PORT_BASE="${BOOTSTRAP_PORT_BASE:-8998}"
HEXGENT_ROUTER_PORT="${HEXGENT_ROUTER_PORT:-33000}"
BASELINE_ROUTER_PORT="${BASELINE_ROUTER_PORT:-33001}"
HEXGENT_FALLBACK_POLICY="${HEXGENT_FALLBACK_POLICY:-pd_policy}"
HEXGENT_ADMISSION_MAX_WAIT_MS="${HEXGENT_ADMISSION_MAX_WAIT_MS:-300000}"
HEXGENT_ADMISSION_POLL_INTERVAL_MS="${HEXGENT_ADMISSION_POLL_INTERVAL_MS:-20}"
# Keep the global queue as the source of truth, but do not make the local
# prefill admission credit smaller than the configured prefill request cap by
# default. A small credit (for example 1 or 4) would prevent a 32-request
# prefill batch from ever forming under sustained load.
HEXGENT_MAX_PREFILL_QUEUE_PER_WORKER="${HEXGENT_MAX_PREFILL_QUEUE_PER_WORKER:-${PREFILL_MAX_RUNNING_REQUESTS}}"
# Mooncake emits transfer-byte/throughput summaries when enabled.  The
# per-worker IB device is optional because older Pods may not expose RDMA.
MC_TE_METRIC="${MC_TE_METRIC:-1}"
MC_TE_METRIC_INTERVAL_SECONDS="${MC_TE_METRIC_INTERVAL_SECONDS:-1}"
MOONCAKE_IB_DEVICE_PREFIX="${MOONCAKE_IB_DEVICE_PREFIX:-}"
# Do not force an NVLINK/MNNVL pool by default.  On hosts without the
# matching transport Mooncake falls back to TCP, while a forced pool can add
# allocator/metadata failure modes. Enable it only after verifying P2P.
MOONCAKE_CUSTOM_MEM_POOL="${MOONCAKE_CUSTOM_MEM_POOL:-none}"
MC_FORCE_MNNVL="${MC_FORCE_MNNVL:-none}"
MOONCAKE_INTRANODE_NVLINK="${MOONCAKE_INTRANODE_NVLINK:-0}"
MOONCAKE_ENGINE_OVERLAY_DIR="${MOONCAKE_ENGINE_OVERLAY_DIR:-}"
# Measured TCP fallback on this Pod was roughly 0.2-0.4 GB/s, not the 50 GB/s
# nominal fabric value embedded in the replay trace. Override this when the
# transport telemetry proves that NVLink or RDMA is actually active.
if [[ "$MOONCAKE_INTRANODE_NVLINK" == "1" && -z "${TRANSFER_BANDWIDTH_BYTES_PER_S+x}" ]]; then
  # Raw same-host Mooncake P2P can exceed 300 GB/s after warm-up, but the
  # request-level P-D path on this H100 Pod measured about 7 GB/s plus a small
  # setup cost. Plan against that end-to-end value, not the microbenchmark.
  TRANSFER_BANDWIDTH_BYTES_PER_S=7000000000
else
  TRANSFER_BANDWIDTH_BYTES_PER_S="${TRANSFER_BANDWIDTH_BYTES_PER_S:-250000000}"
fi
TRANSFER_STARTUP_MS="${TRANSFER_STARTUP_MS:-10}"
# Emit per-request prefill/decode queue and transfer durations.  This is
# required to calibrate the planner against the actual PD transport rather
# than only inferring transfer time from an assumed bandwidth.
ENABLE_REQUEST_TIME_STATS_LOGGING="${ENABLE_REQUEST_TIME_STATS_LOGGING:-1}"
REQUEST_TIME_STATS_ARGS=()
if [[ "$ENABLE_REQUEST_TIME_STATS_LOGGING" == "1" || "$ENABLE_REQUEST_TIME_STATS_LOGGING" == "true" ]]; then
  REQUEST_TIME_STATS_ARGS+=(--enable-request-time-stats-logging)
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"
: >"$LOG_DIR/planner_invocations.jsonl"
source "$VENV_DIR/bin/activate"
if [[ -f /workspace/hexgent_e2e/env.sh ]]; then
  # shellcheck source=/dev/null
  source /workspace/hexgent_e2e/env.sh
fi
if [[ "$MOONCAKE_INTRANODE_NVLINK" == "1" ]]; then
  if [[ -z "$MOONCAKE_ENGINE_OVERLAY_DIR" || ! -f "$MOONCAKE_ENGINE_OVERLAY_DIR/mooncake/engine.so" ]]; then
    echo "MOONCAKE_INTRANODE_NVLINK=1 requires a source-built Mooncake overlay" >&2
    exit 1
  fi
  export MC_INTRANODE_NVLINK=1
  export SGLANG_MOONCAKE_TRANSFER_PROTOCOL=nvlink_intra
  # This selects the Mooncake intra-node NVLink transport for large KV data;
  # it does not enable the legacy MNNVL allocator.
  export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
  # Transfer the tiny CPU-resident auxiliary metadata through ZMQ/TCP. KV
  # cache pages remain on the direct NVLink transport above.
  export SGLANG_MOONCAKE_SEND_AUX_TCP=1
  export LD_LIBRARY_PATH="$MOONCAKE_ENGINE_OVERLAY_DIR/mooncake:${LD_LIBRARY_PATH:-}"
else
  unset MC_INTRANODE_NVLINK
  unset SGLANG_MOONCAKE_TRANSFER_PROTOCOL
  unset SGLANG_MOONCAKE_SEND_AUX_TCP
fi
export PYTHONPATH="${MOONCAKE_ENGINE_OVERLAY_DIR:+$MOONCAKE_ENGINE_OVERLAY_DIR:}$SRC_DIR/python:${PYTHONPATH:-}"
export MC_TE_METRIC MC_TE_METRIC_INTERVAL_SECONDS
export SGLANG_DISAGGREGATION_MAX_INFLIGHT_TRANSFERS="$DECODE_MAX_INFLIGHT_TRANSFERS"
# The NVLink setting applies only to bulk KV. Auxiliary metadata remains on
# CPU and uses the control-message path.
export HEXGENT_TRANSFER_BANDWIDTH_BYTES_PER_S="$TRANSFER_BANDWIDTH_BYTES_PER_S"
export HEXGENT_TRANSFER_STARTUP_MS="$TRANSFER_STARTUP_MS"
# Requests are mutable only in the global admission queue. Do not issue a
# worker-local requeue after bootstrap or KV transfer has started.
export HEXGENT_ENABLE_LOCAL_WAITING_MUTATIONS=0
if [[ "$MOONCAKE_INTRANODE_NVLINK" == "1" ]]; then
  export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
elif [[ -n "$MOONCAKE_CUSTOM_MEM_POOL" && "$MOONCAKE_CUSTOM_MEM_POOL" != "none" ]]; then
  export SGLANG_MOONCAKE_CUSTOM_MEM_POOL="$MOONCAKE_CUSTOM_MEM_POOL"
else
  unset SGLANG_MOONCAKE_CUSTOM_MEM_POOL
fi
if [[ "$MC_FORCE_MNNVL" == "unset" || "$MC_FORCE_MNNVL" == "none" ]]; then
  unset MC_FORCE_MNNVL
else
  export MC_FORCE_MNNVL
fi

stop_pid_file() {
  local pid_file="$1"
  if [[ -s "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    # Every long-lived service below is launched with setsid, so its PID is
    # also its process-group ID.  Stopping the whole group is required because
    # SGLang's launcher leaves scheduler/detokenizer children alive otherwise;
    # stale schedulers keep their bootstrap ports and GPU allocations.
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    rm -f "$pid_file"
  fi
}

for name in router baseline_router planner \
  prefill0 prefill1 prefill2 prefill3 decode0 decode1 decode2 decode3; do
  stop_pid_file "$RUN_DIR/${name}.pid"
done

setsid python -m sglang.hexgent.sidecar \
  --host 127.0.0.1 --port "$PLANNER_PORT" --log-level warning \
  --telemetry-path "$LOG_DIR/planner_invocations.jsonl" \
  >"$LOG_DIR/planner.log" 2>&1 &
echo $! >"$RUN_DIR/planner.pid"

start_prefill() {
  local index="$1"
  local port=$((PREFILL_PORT_BASE + index))
  local bootstrap_port=$((BOOTSTRAP_PORT_BASE + index))
  local physical_gpu="$index"
  local -a mooncake_ib_args=()
  if [[ -n "$MOONCAKE_IB_DEVICE_PREFIX" ]]; then
    mooncake_ib_args=(--mooncake-ib-device "${MOONCAKE_IB_DEVICE_PREFIX}${physical_gpu}")
  fi
  setsid env CUDA_VISIBLE_DEVICES="$index" python -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 --port "$port" --tp-size 1 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
    --disaggregation-bootstrap-port "$bootstrap_port" \
    --disaggregation-decode-tp 1 \
    "${mooncake_ib_args[@]}" \
    --context-length "$CONTEXT_LENGTH" \
    --max-running-requests "$PREFILL_MAX_RUNNING_REQUESTS" \
    --prefill-max-requests "$PREFILL_MAX_BATCH_REQUESTS" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" --disable-radix-cache \
    --enable-priority-scheduling \
    --schedule-low-priority-values-first \
    --trust-remote-code \
    "${REQUEST_TIME_STATS_ARGS[@]}" \
    >"$LOG_DIR/prefill${index}.log" 2>&1 &
  echo $! >"$RUN_DIR/prefill${index}.pid"
}

start_decode() {
  local index="$1"
  local gpu=$((index + 4))
  local port=$((DECODE_PORT_BASE + index))
  local -a mooncake_ib_args=()
  if [[ -n "$MOONCAKE_IB_DEVICE_PREFIX" ]]; then
    mooncake_ib_args=(--mooncake-ib-device "${MOONCAKE_IB_DEVICE_PREFIX}${gpu}")
  fi
  setsid env CUDA_VISIBLE_DEVICES="$gpu" python -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 --port "$port" --tp-size 1 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend "$TRANSFER_BACKEND" \
    --disaggregation-prefill-pp 1 \
    "${mooncake_ib_args[@]}" \
    --context-length "$CONTEXT_LENGTH" \
    --max-running-requests "$DECODE_MAX_RUNNING_REQUESTS" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" --disable-radix-cache \
    --enable-priority-scheduling \
    --schedule-low-priority-values-first \
    --trust-remote-code \
    "${REQUEST_TIME_STATS_ARGS[@]}" \
    >"$LOG_DIR/decode${index}.log" 2>&1 &
  echo $! >"$RUN_DIR/decode${index}.pid"
}

wait_endpoint() {
  local name="$1" url="$2" log_file="$3" pid_file="$4"
  for _ in $(seq 1 300); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name ready: $url"
      return 0
    fi
    if [[ -s "$pid_file" ]] && ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      tail -120 "$log_file" >&2
      return 1
    fi
    sleep 2
  done
  tail -120 "$log_file" >&2
  return 1
}

for index in 0 1 2 3; do
  start_prefill "$index"
  start_decode "$index"
done
for index in 0 1 2 3; do
  wait_endpoint "prefill${index}" \
    "http://127.0.0.1:$((PREFILL_PORT_BASE + index))/v1/models" \
    "$LOG_DIR/prefill${index}.log" "$RUN_DIR/prefill${index}.pid"
  wait_endpoint "decode${index}" \
    "http://127.0.0.1:$((DECODE_PORT_BASE + index))/v1/models" \
    "$LOG_DIR/decode${index}.log" "$RUN_DIR/decode${index}.pid"
done

router_workers=()
for index in 0 1 2 3; do
  router_workers+=(
    --prefill "http://127.0.0.1:$((PREFILL_PORT_BASE + index))" "$((BOOTSTRAP_PORT_BASE + index))"
  )
done
for index in 0 1 2 3; do
  router_workers+=(--decode "http://127.0.0.1:$((DECODE_PORT_BASE + index))")
done

setsid python -m sglang_router.launch_router \
  --host 127.0.0.1 --port "$HEXGENT_ROUTER_PORT" \
  --pd-disaggregation "${router_workers[@]}" \
  --hexgent-router \
  --hexgent-planner-url "http://127.0.0.1:${PLANNER_PORT}" \
  --hexgent-planner-timeout-ms 2000 \
  --hexgent-fallback-policy "$HEXGENT_FALLBACK_POLICY" \
  --hexgent-admission-queue \
  --hexgent-admission-max-wait-ms "$HEXGENT_ADMISSION_MAX_WAIT_MS" \
  --hexgent-admission-poll-interval-ms "$HEXGENT_ADMISSION_POLL_INTERVAL_MS" \
  --hexgent-max-prefill-queue-per-worker "$HEXGENT_MAX_PREFILL_QUEUE_PER_WORKER" \
  >"$LOG_DIR/router.log" 2>&1 &
echo $! >"$RUN_DIR/router.pid"

setsid python -m sglang_router.launch_router \
  --host 127.0.0.1 --port "$BASELINE_ROUTER_PORT" \
  --prometheus-port 29001 --pd-disaggregation "${router_workers[@]}" \
  >"$LOG_DIR/baseline_router.log" 2>&1 &
echo $! >"$RUN_DIR/baseline_router.pid"

wait_endpoint hexgent-router \
  "http://127.0.0.1:${HEXGENT_ROUTER_PORT}/v1/models" \
  "$LOG_DIR/router.log" "$RUN_DIR/router.pid"
wait_endpoint baseline-router \
  "http://127.0.0.1:${BASELINE_ROUTER_PORT}/v1/models" \
  "$LOG_DIR/baseline_router.log" "$RUN_DIR/baseline_router.pid"

echo "HexAgenT 4P4D deployment is ready."
echo "HexAgenT router: http://127.0.0.1:${HEXGENT_ROUTER_PORT}"
echo "Baseline router: http://127.0.0.1:${BASELINE_ROUTER_PORT}"
