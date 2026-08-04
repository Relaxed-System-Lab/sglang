use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::{DashMap, DashSet};
use futures_util::{future::join_all, StreamExt};
use memchr::memmem;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::oneshot;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};

use super::pd_types::{
    api_path, HexgentAdmissionPolicy, HexgentAdmissionRequest, HexgentAdmissionResponse,
    HexgentPlannerEvent, HexgentPlannerRequest, HexgentPlannerResponse, HexgentPlannerState,
    HexgentPlannerWorker, HexgentWorkerMutationAction,
};
use crate::{
    config::types::{HexgentConfig, HexgentFallbackPolicy, RetryConfig},
    core::{
        is_retryable_status, HashRing, RetryExecutor, Worker, WorkerLoadGuard, WorkerRegistry,
        WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{LoadBalancingPolicy, PolicyRegistry, SelectWorkerInfo},
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        classify::ClassifyRequest,
        common::{InputIds, StringOrArray},
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::RerankRequest,
    },
    routers::{
        error,
        grpc::utils::{error_type_from_status, route_to_endpoint},
        header_utils, RouterTrait,
    },
};

#[derive(Debug)]
pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub api_key: Option<String>,
    pub enable_igw: bool,
    pub hexgent: HexgentConfig,
    hexgent_snapshot_cache: Arc<DashMap<String, HexgentCachedSnapshot>>,
    hexgent_snapshot_refreshing: Arc<DashSet<String>>,
    hexgent_admission_queue: Arc<Mutex<HexgentGlobalAdmissionQueue>>,
    hexgent_admission_waiters:
        Arc<DashMap<HexgentAdmissionKey, oneshot::Sender<HexgentPlannerResponse>>>,
}

#[derive(Debug, Clone)]
struct HexgentCachedSnapshot {
    snapshot: Value,
    captured_at: Instant,
}

#[derive(Clone, Copy)]
struct HexgentSelectionTiming {
    snapshot_fetch: Duration,
    admission_wait: Duration,
}

impl HexgentSelectionTiming {
    fn from_admission_phase(snapshot_fetch: Duration, admission_wait: Duration) -> Self {
        Self {
            snapshot_fetch,
            admission_wait,
        }
    }
}

/// Stable identity for a request while it remains outside a worker.
///
/// A request can be re-planned freely until the router grants it prefill
/// admission and creates its bootstrap room.  Keeping this key independent of
/// a worker makes that boundary explicit.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct HexgentAdmissionKey {
    workflow_id: String,
    call_id: String,
}

impl HexgentAdmissionKey {
    fn new(workflow_id: impl Into<String>, call_id: impl Into<String>) -> Self {
        Self {
            workflow_id: workflow_id.into(),
            call_id: call_id.into(),
        }
    }
}

#[derive(Clone, Debug)]
struct HexgentAdmissionGrant {
    key: HexgentAdmissionKey,
    decision: HexgentPlannerResponse,
}

#[derive(Clone, Debug)]
struct HexgentQueuedAdmission {
    key: HexgentAdmissionKey,
    fallback_decision: HexgentPlannerResponse,
    planner_decision: Option<HexgentPlannerResponse>,
    fallback_eligible: bool,
    arrival_order: u64,
}

/// Router-owned global prefill admission state.
///
/// This deliberately owns only requests that have not been bootstrapped.  It
/// models prefill admission credit separately from the normal worker load,
/// whose lifetime extends through decode.  A newer planner revision replaces
/// the queued decision before dispatch; once a grant is issued the request is
/// immutable and follows the normal P-D lifecycle.
#[derive(Debug)]
struct HexgentGlobalAdmissionQueue {
    max_inflight_per_prefill: usize,
    next_arrival_order: u64,
    inflight_by_prefill: HashMap<String, usize>,
    pending: Vec<HexgentQueuedAdmission>,
}

impl HexgentGlobalAdmissionQueue {
    fn new(max_inflight_per_prefill: u64) -> Self {
        Self {
            max_inflight_per_prefill: usize::try_from(max_inflight_per_prefill)
                .unwrap_or(usize::MAX)
                .max(1),
            next_arrival_order: 0,
            inflight_by_prefill: HashMap::new(),
            pending: Vec::new(),
        }
    }

    fn enqueue(
        &mut self,
        key: HexgentAdmissionKey,
        fallback_decision: HexgentPlannerResponse,
    ) -> Vec<HexgentAdmissionGrant> {
        if let Some(existing) = self.pending.iter_mut().find(|entry| entry.key == key) {
            existing.fallback_decision = fallback_decision;
            return self.grant_available();
        }
        // Only an actually idle prefill worker may consume native fallback
        // immediately. Once that worker owns prefill work, retain subsequent
        // calls in the global queue until the planner publishes their batch
        // decisions. This preserves no-idle-worker latency while still giving
        // the planner ownership of the local prefill batch under load.
        let fallback_eligible = !self
            .inflight_by_prefill
            .contains_key(&fallback_decision.prefill_worker_url);
        self.next_arrival_order = self.next_arrival_order.saturating_add(1);
        self.pending.push(HexgentQueuedAdmission {
            key,
            fallback_decision,
            planner_decision: None,
            fallback_eligible,
            arrival_order: self.next_arrival_order,
        });
        self.grant_available()
    }

    fn publish_plan(
        &mut self,
        key: &HexgentAdmissionKey,
        decision: HexgentPlannerResponse,
    ) -> Vec<HexgentAdmissionGrant> {
        if let Some(entry) = self.pending.iter_mut().find(|entry| &entry.key == key) {
            let replace = entry
                .planner_decision
                .as_ref()
                .is_none_or(|current| decision.revision >= current.revision);
            if replace {
                entry.planner_decision = Some(decision);
            }
        }
        self.grant_available()
    }

    fn enable_fallback(&mut self, key: &HexgentAdmissionKey) -> Vec<HexgentAdmissionGrant> {
        if let Some(entry) = self.pending.iter_mut().find(|entry| &entry.key == key) {
            entry.fallback_eligible = true;
        }
        self.grant_available()
    }

    fn cancel(&mut self, key: &HexgentAdmissionKey) {
        self.pending.retain(|entry| &entry.key != key);
    }

    fn is_pending(&self, key: &HexgentAdmissionKey) -> bool {
        self.pending.iter().any(|entry| &entry.key == key)
    }

    fn release_prefill(&mut self, worker_url: &str) -> Vec<HexgentAdmissionGrant> {
        if let Some(inflight) = self.inflight_by_prefill.get_mut(worker_url) {
            *inflight = inflight.saturating_sub(1);
            if *inflight == 0 {
                self.inflight_by_prefill.remove(worker_url);
            }
        }
        self.grant_available()
    }

    fn has_credit(&self, worker_url: &str) -> bool {
        self.inflight_by_prefill
            .get(worker_url)
            .copied()
            .unwrap_or(0)
            < self.max_inflight_per_prefill
    }

    fn grant_available(&mut self) -> Vec<HexgentAdmissionGrant> {
        let mut grants = Vec::new();
        while let Some(index) = self.next_grant_index() {
            let entry = self.pending.remove(index);
            let decision = entry
                .planner_decision
                .unwrap_or(entry.fallback_decision);
            *self
                .inflight_by_prefill
                .entry(decision.prefill_worker_url.clone())
                .or_insert(0) += 1;
            grants.push(HexgentAdmissionGrant {
                key: entry.key,
                decision,
            });
        }
        grants
    }

    fn next_grant_index(&self) -> Option<usize> {
        self.pending
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                let decision = entry
                    .planner_decision
                    .as_ref()
                    .unwrap_or(&entry.fallback_decision);
                if (entry.planner_decision.is_some() || entry.fallback_eligible)
                    && self.has_credit(&decision.prefill_worker_url)
                {
                    Some((
                        index,
                        entry.planner_decision.is_none(),
                        decision.prefill_priority,
                        entry.arrival_order,
                    ))
                } else {
                    None
                }
            })
            .min_by_key(|(_, uses_fallback, priority, arrival_order)| {
                (*uses_fallback, *priority, *arrival_order)
            })
            .map(|(index, _, _, _)| index)
    }
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    model_id: Option<&'a str>,
    headers: Option<HeaderMap>,
    raw_request: Option<Value>,
}

struct HexgentStreamCompletionNotify {
    client: Client,
    planner_url: String,
    planner_timeout_ms: u64,
    original_request: Value,
    decision: HexgentPlannerResponse,
    start_time: Instant,
}

impl PDRouter {
    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        if let Some(worker_url) = first_worker_url {
            self.proxy_to_worker(worker_url, endpoint, headers).await
        } else {
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        }
    }

    async fn proxy_to_worker(
        &self,
        worker_url: String,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = format!("{}/{}", worker_url, endpoint);
        let mut request_builder = self.client.get(&url);

        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        error::internal_error(
                            "read_response_body_failed",
                            format!("Failed to read response body: {}", e),
                        )
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                // Use the status code to determine which error function to use
                match status {
                    StatusCode::BAD_REQUEST => error::bad_request(
                        "server_bad_request",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::NOT_FOUND => error::not_found(
                        "server_not_found",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                        "server_internal_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                        "server_unavailable",
                        format!("Server returned status: {}", res.status()),
                    ),
                    StatusCode::BAD_GATEWAY => error::bad_gateway(
                        "server_bad_gateway",
                        format!("Server returned status: {}", res.status()),
                    ),
                    _ => error::internal_error(
                        "server_error",
                        format!("Server returned status: {}", res.status()),
                    ),
                }
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                error::internal_error(
                    "proxy_request_failed",
                    format!("Failed to proxy request: {}", e),
                )
            }
        }
    }

    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        Ok(PDRouter {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            api_key: ctx.router_config.api_key.clone(),
            enable_igw: ctx.router_config.enable_igw,
            hexgent: ctx.router_config.hexgent.clone(),
            hexgent_snapshot_cache: Arc::new(DashMap::new()),
            hexgent_snapshot_refreshing: Arc::new(DashSet::new()),
            hexgent_admission_queue: Arc::new(Mutex::new(HexgentGlobalAdmissionQueue::new(
                ctx.router_config.hexgent.max_prefill_queue_per_worker,
            ))),
            hexgent_admission_waiters: Arc::new(DashMap::new()),
        })
    }

    fn handle_server_selection_error(error: String) -> Response {
        error!("Failed to select PD pair error={}", error);
        error::service_unavailable(
            "server_selection_failed",
            format!("No available servers: {}", error),
        )
    }

    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    fn get_generate_batch_size(req: &GenerateRequest) -> Option<usize> {
        // GenerateRequest doesn't support batch via arrays, only via input_ids
        if let Some(InputIds::Batch(batches)) = &req.input_ids {
            if !batches.is_empty() {
                return Some(batches.len());
            }
        }
        None
    }

    fn get_chat_batch_size(req: &ChatCompletionRequest) -> Option<usize> {
        if let Some(n) = req.n {
            if n > 1 {
                return Some(n as usize);
            }
        }
        None
    }

    fn get_completion_batch_size(req: &CompletionRequest) -> Option<usize> {
        if let StringOrArray::Array(arr) = &req.prompt {
            if !arr.is_empty() {
                return Some(arr.len());
            }
        }
        None
    }

    // Static key strings to avoid per-request allocations
    const BOOTSTRAP_HOST_KEY: &'static str = "bootstrap_host";
    const BOOTSTRAP_PORT_KEY: &'static str = "bootstrap_port";
    const BOOTSTRAP_ROOM_KEY: &'static str = "bootstrap_room";

    fn inject_bootstrap_into_value(
        mut original: Value,
        prefill_worker: &dyn Worker,
        batch_size: Option<usize>,
    ) -> Result<Value, String> {
        let obj = original
            .as_object_mut()
            .ok_or_else(|| "Request must be a JSON object".to_string())?;

        if let Some(n) = batch_size {
            let mut hosts = Vec::with_capacity(n);
            let mut ports = Vec::with_capacity(n);
            let mut rooms = Vec::with_capacity(n);
            for _ in 0..n {
                hosts.push(prefill_worker.bootstrap_host());
                ports.push(prefill_worker.bootstrap_port());
                rooms.push(super::pd_types::generate_room_id());
            }
            // Use static string keys to avoid per-request allocations
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::Array(hosts.into_iter().map(Value::from).collect()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                Value::Array(
                    ports
                        .into_iter()
                        .map(|p| match p {
                            Some(v) => Value::from(v),
                            None => Value::Null,
                        })
                        .collect(),
                ),
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::Array(rooms.into_iter().map(Value::from).collect()),
            );
        } else {
            // Use static string keys to avoid per-request allocations
            obj.insert(
                Self::BOOTSTRAP_HOST_KEY.to_string(),
                Value::from(prefill_worker.bootstrap_host()),
            );
            obj.insert(
                Self::BOOTSTRAP_PORT_KEY.to_string(),
                match prefill_worker.bootstrap_port() {
                    Some(v) => Value::from(v),
                    None => Value::Null,
                },
            );
            obj.insert(
                Self::BOOTSTRAP_ROOM_KEY.to_string(),
                Value::from(super::pd_types::generate_room_id()),
            );
        }
        Ok(original)
    }

    fn request_value_for_dispatch<T: Serialize>(
        original_request: &T,
        raw_request: Option<&Value>,
    ) -> Result<Value, serde_json::Error> {
        if let Some(raw_request) = raw_request {
            Ok(raw_request.clone())
        } else {
            serde_json::to_value(original_request)
        }
    }

    fn inject_routing_observability_headers(
        response: &mut Response,
        prefill_worker_url: &str,
        decode_worker_url: &str,
        decision: Option<&HexgentPlannerResponse>,
        selection_wall: Duration,
        hexgent_timing: Option<&HexgentSelectionTiming>,
    ) {
        fn insert(response: &mut Response, name: &'static str, value: &str) {
            if let Ok(value) = HeaderValue::from_str(value) {
                response
                    .headers_mut()
                    .insert(HeaderName::from_static(name), value);
            }
        }

        insert(response, "x-sglang-prefill-worker", prefill_worker_url);
        insert(response, "x-sglang-decode-worker", decode_worker_url);
        insert(
            response,
            "x-hexgent-router-mode",
            if decision.is_some() {
                "hexgent"
            } else {
                "baseline"
            },
        );
        insert(
            response,
            "x-hexgent-selection-wall-us",
            &selection_wall.as_micros().to_string(),
        );
        if let Some(decision) = decision {
            insert(
                response,
                "x-hexgent-schedule-revision",
                &decision.revision.to_string(),
            );
        }
        if let Some(timing) = hexgent_timing {
            insert(
                response,
                "x-hexgent-snapshot-fetch-us",
                &timing.snapshot_fetch.as_micros().to_string(),
            );
            insert(
                response,
                "x-hexgent-admission-wait-us",
                &timing.admission_wait.as_micros().to_string(),
            );
        }
    }

    async fn execute_dual_dispatch<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );
        // Clone request once outside the retry loop, then use Arc to share across attempts
        // This avoids O(retries) clones by sharing the same data
        let shared_request = Arc::new(original_request.clone());
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            {
                move |attempt: u32| {
                    // Clone Arc (cheap reference count increment) instead of cloning the entire request
                    let shared_request = Arc::clone(&shared_request);
                    let context = context.clone();
                    async move {
                        let mut json_request = match Self::request_value_for_dispatch(
                            shared_request.as_ref(),
                            context.raw_request.as_ref(),
                        ) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        let selection_started = Instant::now();
                        let (prefill, decode, hexgent_decision, hexgent_timing) = match self
                            .select_pd_pair_for_request(
                                context.route,
                                context.batch_size,
                                context.request_text.as_deref(),
                                context.model_id,
                                context.headers.as_ref(),
                                &json_request,
                            )
                            .await
                        {
                            Ok(pair) => pair,
                            Err(e) => {
                                return Self::handle_server_selection_error(e);
                            }
                        };
                        // Capture pair selection before the P-D request lifecycle. The
                        // response is only available after dispatch completes, so reading
                        // this clock below would incorrectly include model execution.
                        let selection_wall = selection_started.elapsed();

                        debug!(
                            "PD retry attempt {} using prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );

                        if let Some(decision) = hexgent_decision.as_ref() {
                            self.apply_hexgent_actions(decision).await;
                            Self::inject_hexgent_plan_into_value(&mut json_request, decision);
                        }

                        json_request = match Self::inject_bootstrap_into_value(
                            json_request,
                            prefill.as_ref(),
                            context.batch_size,
                        ) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        let mut response = self
                            .execute_dual_dispatch_internal(
                                headers,
                                json_request,
                                context,
                                Arc::clone(&prefill),
                                Arc::clone(&decode),
                                start_time,
                                hexgent_decision.as_ref(),
                            )
                            .await;

                        Self::inject_routing_observability_headers(
                            &mut response,
                            prefill.url(),
                            decode.url(),
                            hexgent_decision.as_ref(),
                            selection_wall,
                            hexgent_timing.as_ref(),
                        );

                        let status = response.status();
                        let not_error = status.is_success() || status.is_client_error();
                        prefill.record_outcome(not_error);
                        decode.record_outcome(not_error);

                        // Record worker errors for server errors (5xx)
                        if status.is_server_error() {
                            let error_type = error_type_from_status(status);
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_PREFILL,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                            Metrics::record_worker_error(
                                metrics_labels::WORKER_DECODE,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                        }

                        response
                    }
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                // Layer 3 worker metrics (PD mode uses both prefill and decode workers)
                Metrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_PREFILL, endpoint);
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_DECODE, endpoint);
            },
        )
        .await;

        // Record Layer 2 metrics
        let duration = start_time.elapsed();
        if response.status().is_success() {
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        let status = res.status();

        if context.is_stream {
            // Handle streaming error response
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let error_payload = match res.bytes().await {
                Ok(error_body) => {
                    if let Ok(error_json) = serde_json::from_slice::<Value>(&error_body) {
                        json!({ "message": error_json, "status": status.as_u16() })
                    } else {
                        json!({ "message": String::from_utf8_lossy(&error_body).to_string(), "status": status.as_u16() })
                    }
                }
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };

            let sse_data = format!(
                "data: {{'error': {}}}",
                serde_json::to_string(&error_payload).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            let decode_url = decode.url().to_string();
            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(decode_url),
                Some(response_headers),
                prefill,
                decode,
                None,
            )
        } else {
            // Handle non-streaming error response
            match res.bytes().await {
                Ok(error_body) => {
                    // Try to parse error message from body, fallback to status-based error
                    let error_message = if let Ok(error_json) =
                        serde_json::from_slice::<Value>(&error_body)
                    {
                        if let Some(msg) = error_json
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else if let Some(msg) = error_json.get("message").and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else {
                            String::from_utf8_lossy(&error_body).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&error_body).to_string()
                    };

                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_bad_request", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_not_found", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_internal_error", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_unavailable", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_bad_gateway", error_message)
                        }
                        _ => error::internal_error("decode_error", error_message),
                    }
                }
                Err(e) => {
                    let error_message = format!("Decode server error: {}", e);
                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match status_code {
                        StatusCode::BAD_REQUEST => {
                            error::bad_request("decode_read_failed", error_message)
                        }
                        StatusCode::NOT_FOUND => {
                            error::not_found("decode_read_failed", error_message)
                        }
                        StatusCode::INTERNAL_SERVER_ERROR => {
                            error::internal_error("decode_read_failed", error_message)
                        }
                        StatusCode::SERVICE_UNAVAILABLE => {
                            error::service_unavailable("decode_read_failed", error_message)
                        }
                        StatusCode::BAD_GATEWAY => {
                            error::bad_gateway("decode_read_failed", error_message)
                        }
                        _ => error::internal_error("decode_read_failed", error_message),
                    }
                }
            }
        }
    }

    // Internal method that performs the actual dual dispatch (without retry logic)
    async fn execute_dual_dispatch_internal(
        &self,
        headers: Option<&HeaderMap>,
        json_request: Value,
        context: PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        start_time: Instant,
        hexgent_decision: Option<&HexgentPlannerResponse>,
    ) -> Response {
        // For non-streaming: use guard for automatic load management
        // For streaming: load will be managed in create_streaming_response
        let _prefill_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(prefill.clone(), headers));
        let _decode_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone(), headers));

        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        // Build both requests
        let prefill_request = self.build_post_with_headers(
            &self.client,
            prefill.url(),
            context.route,
            &json_request,
            headers,
            false,
        );
        let decode_request = self.build_post_with_headers(
            &self.client,
            decode.url(),
            context.route,
            &json_request,
            headers,
            false,
        );

        // Send both requests concurrently. A successful prefill response is a
        // scheduling event in its own right: it frees prefill admission
        // capacity even while this request is still decoding.
        // Note: Using borrowed references avoids heap allocation
        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        let prefill_send = prefill_request.send();
        let decode_send = decode_request.send();
        tokio::pin!(prefill_send);
        tokio::pin!(decode_send);

        let mut prefill_result: Option<Result<reqwest::Response, reqwest::Error>> = None;
        let mut decode_result: Option<Result<reqwest::Response, reqwest::Error>> = None;
        while prefill_result.is_none() || decode_result.is_none() {
            tokio::select! {
                result = &mut prefill_send, if prefill_result.is_none() => {
                    let prefill_succeeded = matches!(
                        &result,
                        Ok(response) if response.status().is_success()
                    );
                    prefill_result = Some(result);
                    if let Some(decision) = hexgent_decision {
                        // Prefill admission ends at the prefill response, not
                        // at final decode completion.  The decode worker keeps
                        // its normal continuous-batching queue afterwards.
                        self.release_hexgent_global_prefill_admission(decision);
                    }
                    if prefill_succeeded {
                        self.notify_hexgent_prefill_completed(&json_request, hexgent_decision)
                            .await;
                    }
                }
                result = &mut decode_send, if decode_result.is_none() => {
                    decode_result = Some(result);
                }
            }
        }

        let prefill_result = prefill_result.expect("prefill request must complete");
        let decode_result = decode_result.expect("decode request must complete");

        events::RequestReceivedEvent {}.emit();

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                debug!("Decode response status: {}", status);

                if !status.is_success() {
                    error!(
                        "Decode server returned error status decode_url={} status={}",
                        decode.url(),
                        status
                    );
                    self.notify_hexgent_request_cancelled(
                        &json_request,
                        hexgent_decision,
                        "decode_http_error",
                    )
                    .await;

                    return self
                        .handle_decode_error_response(res, &context, prefill, decode)
                        .await;
                }

                // Process prefill response
                let prefill_body = if context.return_logprob {
                    match self
                        .process_prefill_response(
                            prefill_result,
                            prefill.url(),
                            context.return_logprob,
                        )
                        .await
                    {
                        Ok((_, body)) => body,
                        Err(error_response) => {
                            self.notify_hexgent_request_cancelled(
                                &json_request,
                                hexgent_decision,
                                "prefill_http_error",
                            )
                            .await;
                            return error_response;
                        }
                    }
                } else {
                    // Even if we don't need logprobs, we should check prefill status
                    match self
                        .process_prefill_response(prefill_result, prefill.url(), false)
                        .await
                    {
                        Ok((_, body)) => body,
                        Err(error_response) => {
                            self.notify_hexgent_request_cancelled(
                                &json_request,
                                hexgent_decision,
                                "prefill_http_error",
                            )
                            .await;
                            return error_response;
                        }
                    }
                };

                if context.is_stream {
                    // Streaming response
                    let prefill_logprobs = if context.return_logprob {
                        prefill_body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .and_then(|json| {
                                json.pointer("/meta_info/input_token_logprobs").cloned()
                            })
                    } else {
                        None
                    };

                    let response_headers = header_utils::preserve_response_headers(res.headers());
                    let hexgent_completion =
                        hexgent_decision
                            .cloned()
                            .map(|decision| HexgentStreamCompletionNotify {
                                client: self.client.clone(),
                                planner_url: self.hexgent.planner_url.clone(),
                                planner_timeout_ms: self.hexgent.planner_timeout_ms,
                                original_request: json_request.clone(),
                                decision,
                                start_time,
                            });

                    self.create_streaming_response(
                        res.bytes_stream(),
                        status,
                        prefill_logprobs,
                        context.return_logprob,
                        None,
                        Some(response_headers),
                        prefill,
                        decode,
                        hexgent_completion,
                    )
                } else {
                    // Non-streaming response
                    let response = if context.return_logprob {
                        self.process_non_streaming_response(
                            res,
                            status,
                            context.return_logprob,
                            prefill_body,
                        )
                        .await
                    } else {
                        // Direct passthrough when no logprobs needed
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(decode_body) => {
                                let mut response = Response::new(Body::from(decode_body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => {
                                error!("Failed to read decode response: {}", e);
                                error::internal_error(
                                    "read_response_failed",
                                    "Failed to read response",
                                )
                            }
                        }
                    };
                    drop(_prefill_guard);
                    drop(_decode_guard);
                    if response.status().is_success() {
                        self.notify_hexgent_completion(
                            &json_request,
                            hexgent_decision,
                            start_time,
                            response.status(),
                        )
                        .await;
                    } else {
                        self.notify_hexgent_request_cancelled(
                            &json_request,
                            hexgent_decision,
                            "decode_response_read_error",
                        )
                        .await;
                    }
                    response
                }
            }
            Err(e) => {
                error!(
                    decode_url = %decode.url(),
                    error = %e,
                    "Decode request failed"
                );
                self.notify_hexgent_request_cancelled(
                    &json_request,
                    hexgent_decision,
                    "decode_transport_error",
                )
                .await;
                error::bad_gateway("decode_server_error", format!("Decode server error: {}", e))
            }
        }
    }

    fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        self.hexgent.enabled
            || prefill_policy.needs_request_text()
            || decode_policy.needs_request_text()
    }

    fn current_time_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0)
    }

    fn refresh_hexgent_admission_state(
        payload: &mut HexgentPlannerRequest,
        current_time_ms: u64,
        worker_snapshots: Option<HashMap<String, Value>>,
    ) {
        payload.state.current_time_ms = current_time_ms;
        if worker_snapshots.is_some() {
            payload.state.worker_snapshots = worker_snapshots;
        }
    }

    fn max_tokens_from_request(original_request: &Value) -> Option<u64> {
        original_request
            .get("max_tokens")
            .and_then(Value::as_u64)
            .or_else(|| {
                original_request
                    .get("max_completion_tokens")
                    .and_then(Value::as_u64)
            })
            .or_else(|| {
                original_request
                    .get("sampling_params")
                    .and_then(|params| params.get("max_new_tokens"))
                    .and_then(Value::as_u64)
            })
    }

    fn hexgent_fallback_decision(
        original_request: &Value,
        prefill: &dyn Worker,
        decode: &dyn Worker,
        fallback_reason: String,
    ) -> HexgentPlannerResponse {
        let predicted_output_tokens = original_request
            .get("hexgent")
            .and_then(Value::as_object)
            .and_then(|metadata| metadata.get("predicted_output_tokens"))
            .and_then(Value::as_u64)
            .or_else(|| Self::max_tokens_from_request(original_request))
            .unwrap_or(0);
        HexgentPlannerResponse {
            revision: 0,
            workflow_horizon_ms: None,
            prefill_worker_id: prefill.url().to_string(),
            prefill_worker_url: prefill.url().to_string(),
            decode_worker_id: decode.url().to_string(),
            decode_worker_url: decode.url().to_string(),
            // The worker-local scheduler treats this as a best-effort fallback
            // request. A later planner revision never mutates started work.
            prefill_priority: i64::MAX,
            decode_priority: i64::MAX,
            planned_decode_instance: decode.url().to_string(),
            predicted_output_tokens,
            actions: vec![],
            fallback_reason: Some(fallback_reason),
        }
    }

    fn hexgent_admission_key(payload: &HexgentPlannerRequest) -> Result<HexgentAdmissionKey, String> {
        let metadata = payload
            .hexgent
            .as_object()
            .ok_or_else(|| "hexgent metadata must be an object".to_string())?;
        let workflow_id = metadata
            .get("workflow_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "hexgent metadata is missing workflow_id".to_string())?;
        let call_id = metadata
            .get("call_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "hexgent metadata is missing call_id".to_string())?;
        Ok(HexgentAdmissionKey::new(workflow_id, call_id))
    }

    fn deliver_hexgent_admission_grants(
        waiters: &DashMap<HexgentAdmissionKey, oneshot::Sender<HexgentPlannerResponse>>,
        grants: Vec<HexgentAdmissionGrant>,
    ) {
        for grant in grants {
            if let Some((_, sender)) = waiters.remove(&grant.key) {
                let _ = sender.send(grant.decision);
            }
        }
    }

    fn enqueue_hexgent_global_admission(
        &self,
        key: HexgentAdmissionKey,
        fallback_decision: HexgentPlannerResponse,
    ) -> oneshot::Receiver<HexgentPlannerResponse> {
        let (sender, receiver) = oneshot::channel();
        self.hexgent_admission_waiters.insert(key.clone(), sender);
        let grants = self
            .hexgent_admission_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .enqueue(key, fallback_decision);
        Self::deliver_hexgent_admission_grants(&self.hexgent_admission_waiters, grants);
        receiver
    }

    fn is_hexgent_global_admission_pending(&self, key: &HexgentAdmissionKey) -> bool {
        self.hexgent_admission_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_pending(key)
    }

    fn publish_hexgent_global_plan(
        queue: &Mutex<HexgentGlobalAdmissionQueue>,
        waiters: &DashMap<HexgentAdmissionKey, oneshot::Sender<HexgentPlannerResponse>>,
        key: &HexgentAdmissionKey,
        decision: HexgentPlannerResponse,
    ) {
        let grants = queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .publish_plan(key, decision);
        Self::deliver_hexgent_admission_grants(waiters, grants);
    }

    fn enable_hexgent_global_fallback(
        queue: &Mutex<HexgentGlobalAdmissionQueue>,
        waiters: &DashMap<HexgentAdmissionKey, oneshot::Sender<HexgentPlannerResponse>>,
        key: &HexgentAdmissionKey,
    ) {
        let grants = queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .enable_fallback(key);
        Self::deliver_hexgent_admission_grants(waiters, grants);
    }

    fn release_hexgent_global_prefill_admission(
        &self,
        decision: &HexgentPlannerResponse,
    ) {
        let grants = self
            .hexgent_admission_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .release_prefill(&decision.prefill_worker_url);
        Self::deliver_hexgent_admission_grants(&self.hexgent_admission_waiters, grants);
    }

    fn cancel_hexgent_global_admission(&self, key: &HexgentAdmissionKey) {
        self.hexgent_admission_waiters.remove(key);
        self.hexgent_admission_queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .cancel(key);
    }

    fn spawn_hexgent_admission_planner(
        &self,
        key: HexgentAdmissionKey,
        payload: HexgentPlannerRequest,
    ) {
        let client = self.client.clone();
        let planner_url = self.hexgent.planner_url.trim_end_matches('/').to_string();
        let planner_timeout = Duration::from_millis(self.hexgent.planner_timeout_ms.max(1));
        let max_wait_ms = self.hexgent.admission_max_wait_ms.max(1);
        let policy = HexgentAdmissionPolicy {
            max_prefill_queue_per_worker: self.hexgent.max_prefill_queue_per_worker,
            retry_after_ms: self.hexgent.admission_poll_interval_ms,
        };
        let queue = Arc::clone(&self.hexgent_admission_queue);
        let waiters = Arc::clone(&self.hexgent_admission_waiters);

        tokio::spawn(async move {
            let enqueue_endpoint = format!("{}/enqueue_admission", planner_url);
            let admission_request = HexgentAdmissionRequest {
                request: payload.clone(),
                policy,
            };
            let enqueue = tokio::time::timeout(
                planner_timeout,
                client.post(enqueue_endpoint).json(&admission_request).send(),
            )
            .await;
            match enqueue {
                Ok(Ok(response)) if response.status().is_success() => {}
                Ok(Ok(response)) => {
                    warn!(
                        "HexAgenT admission enqueue returned HTTP {}; enabling fallback for {}/{}",
                        response.status(), key.workflow_id, key.call_id
                    );
                    Self::enable_hexgent_global_fallback(&queue, &waiters, &key);
                    return;
                }
                Ok(Err(err)) => {
                    warn!(
                        "HexAgenT admission enqueue failed for {}/{}: {}; enabling fallback",
                        key.workflow_id, key.call_id, err
                    );
                    Self::enable_hexgent_global_fallback(&queue, &waiters, &key);
                    return;
                }
                Err(_) => {
                    warn!(
                        "HexAgenT admission enqueue timed out for {}/{}; enabling fallback",
                        key.workflow_id, key.call_id
                    );
                    Self::enable_hexgent_global_fallback(&queue, &waiters, &key);
                    return;
                }
            }

            let wait_endpoint = format!("{}/wait_admission", planner_url);
            let wait = tokio::time::timeout(
                Duration::from_millis(max_wait_ms.saturating_add(planner_timeout.as_millis() as u64)),
                client
                    .get(wait_endpoint)
                    .query(&[
                        ("workflow_id", key.workflow_id.as_str()),
                        ("call_id", key.call_id.as_str()),
                        ("timeout_ms", &max_wait_ms.to_string()),
                    ])
                    .send(),
            )
            .await;
            match wait {
                Ok(Ok(response)) if response.status().is_success() => {
                    match response.json::<HexgentAdmissionResponse>().await {
                        Ok(admission) if admission.dispatch_ready => {
                            if let Some(schedule) = admission.schedule {
                                Self::publish_hexgent_global_plan(&queue, &waiters, &key, schedule);
                            } else {
                                warn!(
                                    "HexAgenT marked {}/{} dispatch-ready without a schedule; enabling fallback",
                                    key.workflow_id, key.call_id
                                );
                                Self::enable_hexgent_global_fallback(&queue, &waiters, &key);
                            }
                        }
                        Ok(admission) => {
                            warn!(
                                "HexAgenT admission wait expired for {}/{} reason={}; enabling fallback",
                                key.workflow_id, key.call_id, admission.reason
                            );
                            Self::enable_hexgent_global_fallback(&queue, &waiters, &key);
                        }
                        Err(err) => {
                            warn!(
                                "HexAgenT admission response decode failed for {}/{}: {}; enabling fallback",
                                key.workflow_id, key.call_id, err
                            );
                            Self::enable_hexgent_global_fallback(&queue, &waiters, &key);
                        }
                    }
                }
                Ok(Ok(response)) => {
                    warn!(
                        "HexAgenT admission wait returned HTTP {} for {}/{}; enabling fallback",
                        response.status(), key.workflow_id, key.call_id
                    );
                    Self::enable_hexgent_global_fallback(&queue, &waiters, &key);
                }
                Ok(Err(err)) => {
                    warn!(
                        "HexAgenT admission wait failed for {}/{}: {}; enabling fallback",
                        key.workflow_id, key.call_id, err
                    );
                    Self::enable_hexgent_global_fallback(&queue, &waiters, &key);
                }
                Err(_) => {
                    warn!(
                        "HexAgenT admission wait timed out for {}/{}; enabling fallback",
                        key.workflow_id, key.call_id
                    );
                    Self::enable_hexgent_global_fallback(&queue, &waiters, &key);
                }
            }
        });
    }

    fn worker_label_f64(worker: &Arc<dyn Worker>, key: &str, default: f64) -> f64 {
        worker
            .metadata()
            .labels
            .get(key)
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn worker_label_u64(worker: &Arc<dyn Worker>, key: &str, default: u64) -> u64 {
        worker
            .metadata()
            .labels
            .get(key)
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn worker_payload(workers: &[Arc<dyn Worker>], worker_type: &str) -> Vec<HexgentPlannerWorker> {
        workers
            .iter()
            .map(|worker| HexgentPlannerWorker {
                id: worker.url().to_string(),
                url: worker.url().to_string(),
                worker_type: worker_type.to_string(),
                hardware: worker.metadata().labels.get("hardware").cloned(),
                available_at_ms: 0,
                capacity_tokens: Self::worker_label_u64(
                    worker,
                    "hexgent_capacity_tokens",
                    1_000_000,
                ),
                prefill_base_ms: Self::worker_label_f64(worker, "hexgent_prefill_base_ms", 2.0),
                prefill_ms_per_input_token: Self::worker_label_f64(
                    worker,
                    "hexgent_prefill_ms_per_input_token",
                    0.5,
                ),
                decode_base_ms: Self::worker_label_f64(worker, "hexgent_decode_base_ms", 1.0),
                decode_ms_per_output_token: Self::worker_label_f64(
                    worker,
                    "hexgent_decode_ms_per_output_token",
                    10.0,
                ),
                transfer_ms_per_token: Self::worker_label_f64(
                    worker,
                    "hexgent_transfer_ms_per_token",
                    0.0,
                ),
            })
            .collect()
    }

    fn find_worker_by_url(workers: &[Arc<dyn Worker>], url: &str) -> Option<Arc<dyn Worker>> {
        workers.iter().find(|worker| worker.url() == url).cloned()
    }

    fn hexgent_reassign_payload(action: &HexgentWorkerMutationAction) -> Value {
        json!({
            "request_id": action.request_id.clone(),
            "target_queue_name": action.target_queue_name.clone(),
            "priority": action.priority,
            "schedule_revision": action.schedule_revision,
            "expected_revision": action.expected_revision,
            "planned_decode_instance": action.planned_decode_instance.clone(),
            "requeue_locally": true,
        })
    }

    async fn apply_hexgent_actions(&self, decision: &HexgentPlannerResponse) {
        for action in &decision.actions {
            let endpoint = api_path(&action.worker_url, "/hexgent/reassign_waiting");
            let payload = Self::hexgent_reassign_payload(action);
            let result = tokio::time::timeout(
                Duration::from_millis(self.hexgent.planner_timeout_ms),
                self.client.post(endpoint).json(&payload).send(),
            )
            .await;
            match result {
                Ok(Ok(response)) if response.status().is_success() => {}
                Ok(Ok(response)) => {
                    warn!(
                        "HexAgenT worker mutation returned HTTP {} for request {}",
                        response.status(),
                        action.request_id
                    );
                }
                Ok(Err(err)) => {
                    warn!(
                        "HexAgenT worker mutation failed for request {}: {}",
                        action.request_id, err
                    );
                }
                Err(_) => {
                    warn!(
                        "HexAgenT worker mutation timed out for request {}",
                        action.request_id
                    );
                }
            }
        }
    }

    fn hexgent_snapshot_timeout(&self) -> Duration {
        Duration::from_millis(self.hexgent.planner_timeout_ms.min(50).max(1))
    }

    fn fetch_hexgent_snapshot_before_admission(&self) -> bool {
        !self.hexgent.admission_queue_enabled
    }

    fn cached_hexgent_worker_snapshot(&self, worker_url: &str, max_age: Duration) -> Option<Value> {
        let entry = self.hexgent_snapshot_cache.get(worker_url)?;
        (entry.captured_at.elapsed() <= max_age).then(|| entry.snapshot.clone())
    }

    fn cached_hexgent_worker_snapshots_for_admission(
        &self,
        workers: &[Arc<dyn Worker>],
    ) -> Option<HashMap<String, Value>> {
        let max_staleness = Duration::from_millis(
            self.hexgent
                .snapshot_max_staleness_ms
                .max(self.hexgent.snapshot_refresh_interval_ms),
        );
        let timeout_duration = self.hexgent_snapshot_timeout();
        let mut snapshots = HashMap::new();
        for worker in workers {
            if let Some(snapshot) = self.cached_hexgent_worker_snapshot(worker.url(), max_staleness)
            {
                snapshots.insert(worker.url().to_string(), snapshot);
            }
            // Cache misses and stale entries refresh out of band. Admission
            // can still dispatch a safe native fallback while the next plan
            // waits for this live state instead of blocking an idle worker.
            if self
                .hexgent_snapshot_age(worker.url())
                .is_none_or(|age| age >= Duration::from_millis(self.hexgent.snapshot_refresh_interval_ms))
            {
                self.refresh_hexgent_worker_snapshot_in_background(
                    worker.url().to_string(),
                    timeout_duration,
                );
            }
        }
        (!snapshots.is_empty()).then_some(snapshots)
    }

    fn cache_hexgent_worker_snapshot(&self, worker_url: String, snapshot: Value) {
        self.hexgent_snapshot_cache.insert(
            worker_url,
            HexgentCachedSnapshot {
                snapshot,
                captured_at: Instant::now(),
            },
        );
    }

    fn hexgent_snapshot_age(&self, worker_url: &str) -> Option<Duration> {
        self.hexgent_snapshot_cache
            .get(worker_url)
            .map(|entry| entry.captured_at.elapsed())
    }

    fn refresh_hexgent_worker_snapshot_in_background(
        &self,
        worker_url: String,
        timeout_duration: Duration,
    ) {
        if !self.hexgent_snapshot_refreshing.insert(worker_url.clone()) {
            return;
        }

        let endpoint = api_path(&worker_url, "/hexgent/pd_snapshot");
        let client = self.client.clone();
        let cache = Arc::clone(&self.hexgent_snapshot_cache);
        let refreshing = Arc::clone(&self.hexgent_snapshot_refreshing);
        tokio::spawn(async move {
            let result = tokio::time::timeout(timeout_duration, async {
                let response = client
                    .get(&endpoint)
                    .send()
                    .await
                    .map_err(|err| format!("request failed: {}", err))?;
                if !response.status().is_success() {
                    return Err(format!("returned HTTP {}", response.status()));
                }
                response
                    .json::<Value>()
                    .await
                    .map_err(|err| format!("decode failed: {}", err))
            })
            .await;
            if let Ok(Ok(snapshot)) = result {
                cache.insert(
                    worker_url.clone(),
                    HexgentCachedSnapshot {
                        snapshot,
                        captured_at: Instant::now(),
                    },
                );
            }
            refreshing.remove(&worker_url);
        });
    }

    async fn fetch_hexgent_worker_snapshot(
        &self,
        worker: Arc<dyn Worker>,
        timeout_duration: Duration,
    ) -> Option<(String, Value)> {
        let worker_url = worker.url().to_string();
        let endpoint = api_path(&worker_url, "/hexgent/pd_snapshot");
        let result = tokio::time::timeout(timeout_duration, async {
            let response = self
                .client
                .get(&endpoint)
                .send()
                .await
                .map_err(|err| format!("request failed: {}", err))?;
            if !response.status().is_success() {
                return Err(format!("returned HTTP {}", response.status()));
            }
            response
                .json::<Value>()
                .await
                .map_err(|err| format!("decode failed: {}", err))
        })
        .await;

        match result {
            Ok(Ok(snapshot)) => {
                self.cache_hexgent_worker_snapshot(worker_url.clone(), snapshot.clone());
                Some((worker_url, snapshot))
            }
            Ok(Err(err)) => {
                if let Some(snapshot) =
                    self.cached_hexgent_worker_snapshot(&worker_url, Duration::from_secs(1))
                {
                    debug!(
                        "HexAgenT worker snapshot fetch failed for {}; using recent cache: {}",
                        endpoint, err
                    );
                    Some((worker_url, snapshot))
                } else {
                    warn!(
                        "HexAgenT worker snapshot fetch failed for {}: {}",
                        endpoint, err
                    );
                    None
                }
            }
            Err(_) => {
                if let Some(snapshot) =
                    self.cached_hexgent_worker_snapshot(&worker_url, Duration::from_secs(1))
                {
                    debug!(
                        "HexAgenT worker snapshot fetch timed out for {}; using recent cache",
                        endpoint
                    );
                    Some((worker_url, snapshot))
                } else {
                    warn!("HexAgenT worker snapshot fetch timed out for {}", endpoint);
                    None
                }
            }
        }
    }

    async fn fetch_hexgent_worker_snapshots(
        &self,
        workers: &[Arc<dyn Worker>],
    ) -> Option<HashMap<String, Value>> {
        let timeout_duration = self.hexgent_snapshot_timeout();
        let refresh_after = Duration::from_millis(self.hexgent.snapshot_refresh_interval_ms);
        let max_staleness = Duration::from_millis(
            self.hexgent
                .snapshot_max_staleness_ms
                .max(self.hexgent.snapshot_refresh_interval_ms),
        );
        let mut snapshots = HashMap::new();
        let mut stale_workers = Vec::new();
        for worker in workers {
            let worker_url = worker.url().to_string();
            if let Some(snapshot) = self.cached_hexgent_worker_snapshot(&worker_url, max_staleness)
            {
                snapshots.insert(worker_url, snapshot);
                if self
                    .hexgent_snapshot_age(worker.url())
                    .is_some_and(|age| age >= refresh_after)
                {
                    self.refresh_hexgent_worker_snapshot_in_background(
                        worker.url().to_string(),
                        timeout_duration,
                    );
                }
            } else {
                stale_workers.push(Arc::clone(worker));
            }
        }
        snapshots.extend(
            join_all(
                stale_workers
                    .into_iter()
                    .map(|worker| self.fetch_hexgent_worker_snapshot(worker, timeout_duration)),
            )
            .await
            .into_iter()
            .flatten(),
        );

        (!snapshots.is_empty()).then_some(snapshots)
    }

    async fn fetch_hexgent_stage_timing(
        client: &Client,
        worker_url: &str,
        workflow_id: &str,
        call_id: &str,
        stage: &str,
        timeout_duration: Duration,
    ) -> Option<Value> {
        let endpoint = api_path(worker_url, "/hexgent/request_timing");
        let result = tokio::time::timeout(timeout_duration, async {
            let response = client
                .get(&endpoint)
                .query(&[("workflow_id", workflow_id), ("call_id", call_id)])
                .send()
                .await
                .map_err(|err| format!("request failed: {}", err))?;
            if !response.status().is_success() {
                return Err(format!("returned HTTP {}", response.status()));
            }
            response
                .json::<Value>()
                .await
                .map_err(|err| format!("decode failed: {}", err))
        })
        .await;

        match result {
            Ok(Ok(timing)) => timing
                .get("timings")
                .and_then(Value::as_array)
                .and_then(|timings| timings.iter().find_map(|entry| entry.get(stage)))
                .cloned(),
            Ok(Err(err)) => {
                debug!(
                    "HexAgenT {} timing fetch failed for {}: {}",
                    stage, endpoint, err
                );
                None
            }
            Err(_) => {
                debug!("HexAgenT {} timing fetch timed out for {}", stage, endpoint);
                None
            }
        }
    }

    async fn collect_hexgent_service_timing(
        client: &Client,
        original_request: &Value,
        decision: &HexgentPlannerResponse,
        timeout_duration: Duration,
    ) -> Option<Value> {
        let hexgent = original_request.get("hexgent")?.as_object()?;
        let workflow_id = hexgent.get("workflow_id")?.as_str()?;
        let call_id = hexgent.get("call_id")?.as_str()?;
        let (prefill, decode) = tokio::join!(
            Self::fetch_hexgent_stage_timing(
                client,
                &decision.prefill_worker_url,
                workflow_id,
                call_id,
                "prefill",
                timeout_duration,
            ),
            Self::fetch_hexgent_stage_timing(
                client,
                &decision.decode_worker_url,
                workflow_id,
                call_id,
                "decode",
                timeout_duration,
            ),
        );
        let prefill_execution_ms = prefill?.get("prefill_execution_ms")?.as_u64()?;
        let decode = decode?;
        let transfer_ms = decode.get("transfer_ms")?.as_u64()?;
        let decode_execution_ms = decode.get("decode_execution_ms")?.as_u64()?;
        let measured_service_ms = prefill_execution_ms
            .saturating_add(transfer_ms)
            .saturating_add(decode_execution_ms);

        Some(json!({
            "prefill_execution_ms": prefill_execution_ms,
            "transfer_ms": transfer_ms,
            "decode_execution_ms": decode_execution_ms,
            "measured_service_ms": measured_service_ms,
        }))
    }

    fn hexgent_completion_event_payload(
        original_request: &Value,
        decision: &HexgentPlannerResponse,
        end_to_end_latency_ms: u64,
        measured_timing: Option<Value>,
        timestamp_ms: u64,
        status: StatusCode,
    ) -> Option<Value> {
        let hexgent = original_request.get("hexgent")?.as_object()?;
        let workflow_id = hexgent.get("workflow_id")?.as_str()?.to_string();
        let call_id = hexgent.get("call_id")?.as_str()?.to_string();
        let measured_service_ms = measured_timing
            .as_ref()
            .and_then(|timing| timing.get("measured_service_ms"))
            .and_then(Value::as_u64);
        let event = HexgentPlannerEvent {
            event_type: "call_completed".to_string(),
            workflow_id: Some(workflow_id),
            call_id: Some(call_id),
            // Router wall time includes admission and worker queueing. It is
            // retained separately for workflow completion accounting, never as
            // planner service telemetry.
            service_ms: None,
            end_to_end_latency_ms: Some(end_to_end_latency_ms),
            measured_service_ms,
            timestamp_ms: Some(timestamp_ms),
            payload: json!({
                "status_code": status.as_u16(),
                "end_to_end_latency_ms": end_to_end_latency_ms,
                "stage_timing": measured_timing,
                "prefill_worker_id": decision.prefill_worker_id.clone(),
                "prefill_worker_url": decision.prefill_worker_url.clone(),
                "decode_worker_id": decision.decode_worker_id.clone(),
                "decode_worker_url": decision.decode_worker_url.clone(),
                "prefill_priority": decision.prefill_priority,
                "decode_priority": decision.decode_priority,
                "planned_decode_instance": decision.planned_decode_instance.clone(),
                "schedule_revision": decision.revision,
                "predicted_output_tokens": decision.predicted_output_tokens,
                "fallback_reason": decision.fallback_reason.clone(),
                "hexgent_metadata": Value::Object(hexgent.clone()),
            }),
        };
        serde_json::to_value(event).ok()
    }

    fn hexgent_prefill_completion_event_payload(
        original_request: &Value,
        decision: &HexgentPlannerResponse,
        timestamp_ms: u64,
    ) -> Option<Value> {
        let hexgent = original_request.get("hexgent")?.as_object()?;
        let workflow_id = hexgent.get("workflow_id")?.as_str()?.to_string();
        let call_id = hexgent.get("call_id")?.as_str()?.to_string();
        let event = HexgentPlannerEvent {
            event_type: "prefill_completed".to_string(),
            workflow_id: Some(workflow_id),
            call_id: Some(call_id),
            service_ms: None,
            end_to_end_latency_ms: None,
            measured_service_ms: None,
            timestamp_ms: Some(timestamp_ms),
            payload: json!({
                "prefill_worker_id": decision.prefill_worker_id.clone(),
                "prefill_worker_url": decision.prefill_worker_url.clone(),
                "decode_worker_id": decision.decode_worker_id.clone(),
                "decode_worker_url": decision.decode_worker_url.clone(),
                "schedule_revision": decision.revision,
            }),
        };
        serde_json::to_value(event).ok()
    }

    fn hexgent_request_cancelled_event_payload(
        original_request: &Value,
        decision: &HexgentPlannerResponse,
        reason: &str,
        timestamp_ms: u64,
    ) -> Option<Value> {
        let hexgent = original_request.get("hexgent")?.as_object()?;
        let workflow_id = hexgent.get("workflow_id")?.as_str()?.to_string();
        let call_id = hexgent.get("call_id")?.as_str()?.to_string();
        let event = HexgentPlannerEvent {
            event_type: "request_cancelled".to_string(),
            workflow_id: Some(workflow_id),
            call_id: Some(call_id),
            service_ms: None,
            end_to_end_latency_ms: None,
            measured_service_ms: None,
            timestamp_ms: Some(timestamp_ms),
            payload: json!({
                "reason": reason,
                "schedule_revision": decision.revision,
                "prefill_worker_url": decision.prefill_worker_url.clone(),
                "decode_worker_url": decision.decode_worker_url.clone(),
            }),
        };
        serde_json::to_value(event).ok()
    }

    async fn notify_hexgent_prefill_completed(
        &self,
        original_request: &Value,
        decision: Option<&HexgentPlannerResponse>,
    ) {
        let Some(decision) = decision else {
            return;
        };
        let Some(payload) = Self::hexgent_prefill_completion_event_payload(
            original_request,
            decision,
            Self::current_time_ms(),
        ) else {
            return;
        };
        let endpoint = format!("{}/event", self.hexgent.planner_url.trim_end_matches('/'));
        let result = tokio::time::timeout(
            Duration::from_millis(self.hexgent.planner_timeout_ms),
            self.client.post(endpoint.clone()).json(&payload).send(),
        )
        .await;
        match result {
            Ok(Ok(response)) if response.status().is_success() => {}
            Ok(Ok(response)) => warn!(
                "HexAgenT prefill-completion event returned HTTP {}",
                response.status()
            ),
            Ok(Err(err)) => warn!("HexAgenT prefill-completion event failed: {}", err),
            Err(_) => warn!(
                "HexAgenT prefill-completion event timed out for {}",
                endpoint
            ),
        }
    }

    async fn notify_hexgent_request_cancelled(
        &self,
        original_request: &Value,
        decision: Option<&HexgentPlannerResponse>,
        reason: &str,
    ) {
        let Some(decision) = decision else {
            return;
        };
        let Some(payload) = Self::hexgent_request_cancelled_event_payload(
            original_request,
            decision,
            reason,
            Self::current_time_ms(),
        ) else {
            return;
        };
        let endpoint = format!("{}/event", self.hexgent.planner_url.trim_end_matches('/'));
        let result = tokio::time::timeout(
            Duration::from_millis(self.hexgent.planner_timeout_ms),
            self.client.post(endpoint.clone()).json(&payload).send(),
        )
        .await;
        match result {
            Ok(Ok(response)) if response.status().is_success() => {}
            Ok(Ok(response)) => warn!(
                "HexAgenT request-cancelled event returned HTTP {}",
                response.status()
            ),
            Ok(Err(err)) => warn!("HexAgenT request-cancelled event failed: {}", err),
            Err(_) => warn!(
                "HexAgenT request-cancelled event timed out for {}",
                endpoint
            ),
        }
    }

    async fn notify_hexgent_stream_cancelled(
        completion: &HexgentStreamCompletionNotify,
        reason: &str,
    ) {
        let Some(payload) = Self::hexgent_request_cancelled_event_payload(
            &completion.original_request,
            &completion.decision,
            reason,
            Self::current_time_ms(),
        ) else {
            return;
        };
        let endpoint = format!("{}/event", completion.planner_url.trim_end_matches('/'));
        let result = tokio::time::timeout(
            Duration::from_millis(completion.planner_timeout_ms),
            completion
                .client
                .post(endpoint.clone())
                .json(&payload)
                .send(),
        )
        .await;
        match result {
            Ok(Ok(response)) if response.status().is_success() => {}
            Ok(Ok(response)) => warn!(
                "HexAgenT streaming request-cancelled event returned HTTP {}",
                response.status()
            ),
            Ok(Err(err)) => warn!("HexAgenT streaming request-cancelled event failed: {}", err),
            Err(_) => warn!(
                "HexAgenT streaming request-cancelled event timed out for {}",
                endpoint
            ),
        }
    }

    async fn notify_hexgent_completion(
        &self,
        original_request: &Value,
        decision: Option<&HexgentPlannerResponse>,
        start_time: Instant,
        status: StatusCode,
    ) {
        if !status.is_success() {
            return;
        }
        let Some(decision) = decision else {
            return;
        };
        let end_to_end_latency_ms = start_time.elapsed().as_millis() as u64;
        let timing_timeout = Duration::from_millis(self.hexgent.planner_timeout_ms.min(100).max(1));
        let measured_timing = Self::collect_hexgent_service_timing(
            &self.client,
            original_request,
            decision,
            timing_timeout,
        )
        .await;
        let Some(payload) = Self::hexgent_completion_event_payload(
            original_request,
            decision,
            end_to_end_latency_ms,
            measured_timing,
            Self::current_time_ms(),
            status,
        ) else {
            return;
        };
        let endpoint = format!("{}/event", self.hexgent.planner_url.trim_end_matches('/'));
        let result = tokio::time::timeout(
            Duration::from_millis(self.hexgent.planner_timeout_ms),
            self.client.post(endpoint.clone()).json(&payload).send(),
        )
        .await;
        match result {
            Ok(Ok(response)) if response.status().is_success() => {}
            Ok(Ok(response)) => {
                warn!(
                    "HexAgenT completion event returned HTTP {}",
                    response.status()
                );
            }
            Ok(Err(err)) => {
                warn!("HexAgenT completion event failed: {}", err);
            }
            Err(_) => {
                warn!("HexAgenT completion event timed out for {}", endpoint);
            }
        }
    }

    async fn notify_hexgent_admission_cancel(&self, payload: &HexgentPlannerRequest, reason: &str) {
        let Some(hexgent) = payload.hexgent.as_object() else {
            return;
        };
        let Some(workflow_id) = hexgent.get("workflow_id").and_then(Value::as_str) else {
            return;
        };
        let Some(call_id) = hexgent.get("call_id").and_then(Value::as_str) else {
            return;
        };
        let event = HexgentPlannerEvent {
            event_type: "admission_cancelled".to_string(),
            workflow_id: Some(workflow_id.to_string()),
            call_id: Some(call_id.to_string()),
            service_ms: None,
            end_to_end_latency_ms: None,
            measured_service_ms: None,
            timestamp_ms: Some(Self::current_time_ms()),
            payload: json!({ "reason": reason }),
        };
        let endpoint = format!("{}/event", self.hexgent.planner_url.trim_end_matches('/'));
        let _ = tokio::time::timeout(
            Duration::from_millis(self.hexgent.planner_timeout_ms),
            self.client.post(endpoint).json(&event).send(),
        )
        .await;
    }

    async fn request_hexgent_schedule(
        &self,
        payload: HexgentPlannerRequest,
    ) -> Result<HexgentPlannerResponse, String> {
        let endpoint = format!(
            "{}/schedule_request",
            self.hexgent.planner_url.trim_end_matches('/')
        );
        let response = tokio::time::timeout(
            Duration::from_millis(self.hexgent.planner_timeout_ms),
            self.client.post(endpoint).json(&payload).send(),
        )
        .await
        .map_err(|_| "planner request timed out".to_string())?
        .map_err(|err| format!("planner request failed: {}", err))?;
        if !response.status().is_success() {
            return Err(format!("planner returned HTTP {}", response.status()));
        }
        response
            .json::<HexgentPlannerResponse>()
            .await
            .map_err(|err| format!("planner response decode failed: {}", err))
    }

    fn hexgent_admission_status_query(
        payload: &HexgentPlannerRequest,
    ) -> Result<Vec<(&'static str, String)>, String> {
        let hexgent = payload
            .hexgent
            .as_object()
            .ok_or_else(|| "hexgent metadata must be an object".to_string())?;
        let workflow_id = hexgent
            .get("workflow_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "hexgent metadata is missing workflow_id".to_string())?;
        let call_id = hexgent
            .get("call_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "hexgent metadata is missing call_id".to_string())?;
        Ok(vec![
            ("workflow_id", workflow_id.to_string()),
            ("call_id", call_id.to_string()),
        ])
    }

    fn hexgent_admission_wait_query(
        payload: &HexgentPlannerRequest,
        timeout_ms: u64,
    ) -> Result<Vec<(&'static str, String)>, String> {
        let mut query = Self::hexgent_admission_status_query(payload)?;
        query.push(("timeout_ms", timeout_ms.to_string()));
        Ok(query)
    }

    async fn enqueue_hexgent_admission(
        &self,
        payload: &HexgentPlannerRequest,
    ) -> Result<(), String> {
        let endpoint = format!(
            "{}/enqueue_admission",
            self.hexgent.planner_url.trim_end_matches('/')
        );
        let admission_payload = HexgentAdmissionRequest {
            request: payload.clone(),
            policy: HexgentAdmissionPolicy {
                max_prefill_queue_per_worker: self.hexgent.max_prefill_queue_per_worker,
                retry_after_ms: self.hexgent.admission_poll_interval_ms,
            },
        };
        let response = tokio::time::timeout(
            self.hexgent_snapshot_timeout(),
            self.client.post(endpoint).json(&admission_payload).send(),
        )
        .await
        .map_err(|_| "planner admission enqueue timed out".to_string())?
        .map_err(|err| format!("planner admission enqueue failed: {}", err))?;
        if !response.status().is_success() {
            return Err(format!(
                "planner admission enqueue returned HTTP {}",
                response.status()
            ));
        }
        Ok(())
    }

    async fn request_hexgent_admission_status(
        &self,
        payload: &HexgentPlannerRequest,
    ) -> Result<HexgentAdmissionResponse, String> {
        let endpoint = format!(
            "{}/admission_status",
            self.hexgent.planner_url.trim_end_matches('/')
        );
        let query = Self::hexgent_admission_status_query(payload)?;
        let response = tokio::time::timeout(
            self.hexgent_snapshot_timeout(),
            self.client.get(endpoint).query(&query).send(),
        )
        .await
        .map_err(|_| "planner admission status timed out".to_string())?
        .map_err(|err| format!("planner admission status failed: {}", err))?;
        if !response.status().is_success() {
            return Err(format!(
                "planner admission status returned HTTP {}",
                response.status()
            ));
        }
        response
            .json::<HexgentAdmissionResponse>()
            .await
            .map_err(|err| format!("planner admission status decode failed: {}", err))
    }

    async fn wait_for_hexgent_admission_status(
        &self,
        payload: &HexgentPlannerRequest,
    ) -> Result<HexgentAdmissionResponse, String> {
        let endpoint = format!(
            "{}/wait_admission",
            self.hexgent.planner_url.trim_end_matches('/')
        );
        let max_wait_ms = self.hexgent.admission_max_wait_ms.max(1);
        let query = Self::hexgent_admission_wait_query(payload, max_wait_ms)?;
        let response = tokio::time::timeout(
            Duration::from_millis(max_wait_ms.saturating_add(self.hexgent.planner_timeout_ms)),
            self.client.get(endpoint).query(&query).send(),
        )
        .await
        .map_err(|_| "planner admission wait timed out".to_string())?
        .map_err(|err| format!("planner admission wait failed: {}", err))?;
        if !response.status().is_success() {
            return Err(format!(
                "planner admission wait returned HTTP {}",
                response.status()
            ));
        }
        response
            .json::<HexgentAdmissionResponse>()
            .await
            .map_err(|err| format!("planner admission wait decode failed: {}", err))
    }

    async fn wait_for_hexgent_admission(
        &self,
        mut payload: HexgentPlannerRequest,
        snapshot_workers: &[Arc<dyn Worker>],
    ) -> Result<(HexgentPlannerResponse, HexgentSelectionTiming), String> {
        let snapshot_started = Instant::now();
        let worker_snapshots = self.fetch_hexgent_worker_snapshots(snapshot_workers).await;
        let snapshot_fetch = snapshot_started.elapsed();
        Self::refresh_hexgent_admission_state(
            &mut payload,
            Self::current_time_ms(),
            worker_snapshots,
        );
        let admission_started = Instant::now();
        self.enqueue_hexgent_admission(&payload).await?;
        let admission = self.wait_for_hexgent_admission_status(&payload).await?;
        let admission_wait = admission_started.elapsed();
        if admission.dispatch_ready {
            let schedule = admission
                .schedule
                .ok_or_else(|| "planner admission marked ready without schedule".to_string())?;
            return Ok((
                schedule,
                HexgentSelectionTiming::from_admission_phase(snapshot_fetch, admission_wait),
            ));
        }
        self.notify_hexgent_admission_cancel(&payload, "admission_wait_expired")
            .await;
        Err(format!(
            "planner admission wait expired after {} ms; last reason={} queue_position={}",
            self.hexgent.admission_max_wait_ms, admission.reason, admission.queue_position
        ))
    }

    fn inject_hexgent_plan_into_value(json_request: &mut Value, decision: &HexgentPlannerResponse) {
        let Some(obj) = json_request.as_object_mut() else {
            return;
        };
        let workflow_id = obj
            .get("hexgent")
            .and_then(Value::as_object)
            .and_then(|hexgent| hexgent.get("workflow_id"))
            .cloned();
        if let Some(workflow_id) = workflow_id {
            obj.insert("workflow_id".to_string(), workflow_id);
        }
        if let Some(hexgent) = obj.get("hexgent").and_then(Value::as_object).cloned() {
            let mappings = [
                ("call_id", "hexgent_call_id"),
                ("workflow_arrival_ms", "hexgent_workflow_arrival_ms"),
                ("workflow_horizon_ms", "hexgent_workflow_horizon_ms"),
                ("parent_call_ids", "hexgent_parent_call_ids"),
                ("reveal_sequence", "hexgent_reveal_sequence"),
                ("critical_path_ms", "hexgent_critical_path_ms"),
                ("child_count", "hexgent_child_count"),
                ("input_tokens", "hexgent_input_tokens"),
                ("estimated_prefill_time", "hexgent_estimated_prefill_time"),
                ("estimated_decode_time", "hexgent_estimated_decode_time"),
                ("estimated_transfer_time", "hexgent_estimated_transfer_time"),
                ("workflow_completion_ms", "hexgent_workflow_completion_ms"),
                (
                    "workflow_attained_service_ms",
                    "hexgent_workflow_attained_service_ms",
                ),
            ];
            for (source_key, target_key) in mappings {
                if let Some(value) = hexgent.get(source_key).cloned() {
                    obj.insert(target_key.to_string(), value);
                }
            }
            if let Some(value) = hexgent.get("predicted_output_tokens").cloned() {
                obj.insert("hexgent_predicted_output_tokens".to_string(), value);
            }
        }
        if let Some(workflow_horizon_ms) = decision.workflow_horizon_ms {
            obj.insert(
                "hexgent_workflow_horizon_ms".to_string(),
                Value::from(workflow_horizon_ms),
            );
        }
        obj.insert(
            "priority".to_string(),
            Value::from(decision.prefill_priority),
        );
        obj.insert(
            "hexgent_prefill_priority".to_string(),
            Value::from(decision.prefill_priority),
        );
        obj.insert(
            "hexgent_decode_priority".to_string(),
            Value::from(decision.decode_priority),
        );
        obj.insert(
            "hexgent_schedule_revision".to_string(),
            Value::from(decision.revision),
        );
        obj.insert(
            "planned_decode_instance".to_string(),
            Value::from(decision.planned_decode_instance.clone()),
        );
        obj.insert(
            "hexgent_predicted_output_tokens".to_string(),
            Value::from(decision.predicted_output_tokens),
        );
    }

    async fn select_pd_pair_for_request(
        &self,
        route: &'static str,
        batch_size: Option<usize>,
        request_text: Option<&str>,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
        original_request: &Value,
    ) -> Result<
        (
            Arc<dyn Worker>,
            Arc<dyn Worker>,
            Option<HexgentPlannerResponse>,
            Option<HexgentSelectionTiming>,
        ),
        String,
    > {
        if !self.hexgent.enabled {
            let (prefill, decode) = self.select_pd_pair(request_text, model_id, headers).await?;
            return Ok((prefill, decode, None, None));
        }
        if batch_size.unwrap_or(1) > 1 {
            let (prefill, decode) = self.select_pd_pair(request_text, model_id, headers).await?;
            return Ok((prefill, decode, None, None));
        }

        match self
            .try_select_pd_pair_with_hexgent(
                route,
                request_text,
                model_id,
                headers,
                original_request,
            )
            .await
        {
            Ok(selection) => Ok(selection),
            Err(err) => {
                warn!("HexAgenT planner selection failed: {}", err);
                match self.hexgent.fallback_policy {
                    HexgentFallbackPolicy::FailClosed => Err(err),
                    HexgentFallbackPolicy::PdPolicy => {
                        let (prefill, decode) =
                            self.select_pd_pair(request_text, model_id, headers).await?;
                        let decision = Self::hexgent_fallback_decision(
                            original_request,
                            prefill.as_ref(),
                            decode.as_ref(),
                            err,
                        );
                        Ok((prefill, decode, Some(decision), None))
                    }
                }
            }
        }
    }

    async fn try_select_pd_pair_with_hexgent(
        &self,
        route: &'static str,
        request_text: Option<&str>,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
        original_request: &Value,
    ) -> Result<
        (
            Arc<dyn Worker>,
            Arc<dyn Worker>,
            Option<HexgentPlannerResponse>,
            Option<HexgentSelectionTiming>,
        ),
        String,
    > {
        let Some(hexgent) = original_request.get("hexgent").cloned() else {
            return Err("request is missing hexgent metadata".to_string());
        };

        let prefill_workers = if let Some(model) = model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };
        let decode_workers = if let Some(model) = model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        if prefill_workers.is_empty() || decode_workers.is_empty() {
            return Err("no prefill/decode workers available for HexAgenT".to_string());
        }

        let snapshot_workers: Vec<Arc<dyn Worker>> = prefill_workers
            .iter()
            .chain(decode_workers.iter())
            .cloned()
            .collect();
        let direct_snapshot_started = Instant::now();
        let worker_snapshots = if self.fetch_hexgent_snapshot_before_admission() {
            self.fetch_hexgent_worker_snapshots(&snapshot_workers).await
        } else {
            self.cached_hexgent_worker_snapshots_for_admission(&snapshot_workers)
        };
        let direct_snapshot_fetch = direct_snapshot_started.elapsed();

        let payload = HexgentPlannerRequest {
            route,
            model: model_id.map(str::to_string),
            request_text: request_text.unwrap_or_default().to_string(),
            max_tokens: Self::max_tokens_from_request(original_request),
            hexgent,
            prefill_workers: Self::worker_payload(&prefill_workers, "prefill"),
            decode_workers: Self::worker_payload(&decode_workers, "decode"),
            state: HexgentPlannerState {
                current_time_ms: Self::current_time_ms(),
                worker_snapshots,
            },
        };
        let (decision, timing) = if self.hexgent.admission_queue_enabled {
            let (fallback_prefill, fallback_decode) =
                self.select_pd_pair(request_text, model_id, headers).await?;
            let fallback = Self::hexgent_fallback_decision(
                original_request,
                fallback_prefill.as_ref(),
                fallback_decode.as_ref(),
                "planner_decision_pending".to_string(),
            );
            let key = Self::hexgent_admission_key(&payload)?;
            let receiver = self.enqueue_hexgent_global_admission(key.clone(), fallback);
            // An idle prefill worker can consume native fallback immediately.
            // Do not submit a sidecar admission for that already-dispatched
            // request: it cannot affect placement and could later create a
            // stale reservation after the request has completed.
            if self.is_hexgent_global_admission_pending(&key) {
                self.spawn_hexgent_admission_planner(key.clone(), payload.clone());
            }
            let admission_started = Instant::now();
            let decision = tokio::time::timeout(
                Duration::from_millis(self.hexgent.admission_max_wait_ms.max(1)),
                receiver,
            )
            .await
            .map_err(|_| {
                self.cancel_hexgent_global_admission(&key);
                format!(
                    "global prefill admission timed out after {} ms",
                    self.hexgent.admission_max_wait_ms
                )
            })?
            .map_err(|_| "global prefill admission was cancelled".to_string())?;
            let admission_wait = admission_started.elapsed();
            (
                decision,
                HexgentSelectionTiming::from_admission_phase(
                    direct_snapshot_fetch,
                    admission_wait,
                ),
            )
        } else {
            let admission_started = Instant::now();
            let decision = self.request_hexgent_schedule(payload).await?;
            (
                decision,
                HexgentSelectionTiming::from_admission_phase(
                    direct_snapshot_fetch,
                    admission_started.elapsed(),
                ),
            )
        };
        let prefill = Self::find_worker_by_url(&prefill_workers, &decision.prefill_worker_url)
            .or_else(|| Self::find_worker_by_url(&prefill_workers, &decision.prefill_worker_id))
            .ok_or_else(|| {
                format!(
                    "planner selected unknown prefill worker {}",
                    decision.prefill_worker_url
                )
            })?;
        let decode = Self::find_worker_by_url(&decode_workers, &decision.decode_worker_url)
            .or_else(|| Self::find_worker_by_url(&decode_workers, &decision.decode_worker_id))
            .ok_or_else(|| {
                format!(
                    "planner selected unknown decode worker {}",
                    decision.decode_worker_url
                )
            })?;
        Ok((prefill, decode, Some(decision), Some(timing)))
    }

    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        debug!(
            "Selecting PD pair: enable_igw={}, model_id={:?}, effective_model_id={:?}",
            self.enable_igw, model_id, effective_model_id
        );

        let prefill_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };

        let decode_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model(model)
                .iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .cloned()
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        // Get cached hash ring for consistent hashing
        let hash_ring = self
            .worker_registry
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let prefill = Self::pick_worker_by_policy_arc(
            &prefill_workers,
            &*prefill_policy,
            request_text,
            headers,
            hash_ring.clone(),
            "prefill",
        )
        .await?;

        let decode = Self::pick_worker_by_policy_arc(
            &decode_workers,
            &*decode_policy,
            request_text,
            headers,
            hash_ring,
            "decode",
        )
        .await?;

        // Record worker selection metrics (Layer 3)
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_HTTP,
            model,
            prefill_policy.name(),
        );
        Metrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_HTTP,
            model,
            decode_policy.name(),
        );

        Ok((prefill, decode))
    }

    async fn pick_worker_by_policy_arc(
        workers: &[Arc<dyn Worker>],
        policy: &dyn LoadBalancingPolicy,
        request_text: Option<&str>,
        headers: Option<&HeaderMap>,
        hash_ring: Option<Arc<HashRing>>,
        worker_type: &str,
    ) -> Result<Arc<dyn Worker>, String> {
        if workers.is_empty() {
            return Err(format!(
                "No {} workers available. Please check if {} servers are configured and healthy.",
                worker_type, worker_type
            ));
        }

        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {} workers (all circuits open or unhealthy)",
                worker_type
            ));
        }

        let selected_idx = policy
            .select_worker(
                &available_workers,
                &SelectWorkerInfo {
                    request_text,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                },
            )
            .await
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        Ok(available_workers[selected_idx].clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        decode_url: Option<String>,
        headers: Option<HeaderMap>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        hexgent_completion: Option<HexgentStreamCompletionNotify>,
    ) -> Response {
        use crate::core::AttachedBody;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            futures_util::pin_mut!(stream);
            let mut completed_normally = false;
            let mut cancellation_reason = "decode_stream_terminated";
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let is_done = memmem::find(&chunk, b"data: [DONE]").is_some();

                        let result = if return_logprob && prefill_logprobs.is_some() {
                            Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                .unwrap_or(chunk)
                        } else {
                            chunk
                        };

                        if tx.send(Ok(result)).is_err() {
                            cancellation_reason = "client_stream_disconnected";
                            break;
                        }

                        if is_done {
                            if let Some(completion) = &hexgent_completion {
                                let end_to_end_latency_ms =
                                    completion.start_time.elapsed().as_millis() as u64;
                                let timing_timeout = Duration::from_millis(
                                    completion.planner_timeout_ms.min(100).max(1),
                                );
                                let measured_timing = Self::collect_hexgent_service_timing(
                                    &completion.client,
                                    &completion.original_request,
                                    &completion.decision,
                                    timing_timeout,
                                )
                                .await;
                                if let Some(payload) = Self::hexgent_completion_event_payload(
                                    &completion.original_request,
                                    &completion.decision,
                                    end_to_end_latency_ms,
                                    measured_timing,
                                    Self::current_time_ms(),
                                    status,
                                ) {
                                    let endpoint = format!(
                                        "{}/event",
                                        completion.planner_url.trim_end_matches('/')
                                    );
                                    if let Err(err) =
                                        completion.client.post(endpoint).json(&payload).send().await
                                    {
                                        warn!(
                                            error = %err,
                                            "Failed to notify HexAgenT planner about streaming completion"
                                        );
                                    }
                                }
                            }
                            completed_normally = true;
                            break;
                        }
                    }
                    Err(e) => {
                        if let Some(ref url) = decode_url {
                            error!("Stream error from decode server {}: {}", url, e);
                        }
                        let _ = tx.send(Err(format!("Stream error: {}", e)));
                        cancellation_reason = "decode_stream_error";
                        break;
                    }
                }
            }
            if !completed_normally {
                if let Some(completion) = &hexgent_completion {
                    Self::notify_hexgent_stream_cancelled(completion, cancellation_reason).await;
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let guards = vec![
            WorkerLoadGuard::new(prefill, headers.as_ref()),
            WorkerLoadGuard::new(decode, headers.as_ref()),
        ];

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        AttachedBody::wrap_response(response, guards)
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_non_streaming_response(
        &self,
        res: reqwest::Response,
        status: StatusCode,
        return_logprob: bool,
        prefill_body: Option<bytes::Bytes>,
    ) -> Response {
        let response = res.bytes().await;
        let decode_body = match response {
            Ok(decode_body) => decode_body,
            Err(e) => {
                error!("Failed to read decode response: {}", e);
                return error::internal_error("read_response_failed", "Failed to read response");
            }
        };

        if !return_logprob {
            return (status, decode_body).into_response();
        }

        let Some(prefill_body) = prefill_body else {
            return (status, decode_body).into_response();
        };

        // Merge logprobs from prefill and decode
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(&prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse responses for logprob merging");
            return (status, decode_body).into_response();
        };

        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);

        // Return merged response
        match serde_json::to_vec(&decode_json) {
            Ok(body) => (status, body).into_response(),
            Err(e) => {
                error!("Failed to serialize merged response: {}", e);
                (status, decode_body).into_response()
            }
        }
    }

    // Helper to process prefill response and extract body if needed for logprobs
    async fn process_prefill_response(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        prefill_url: &str,
        return_logprob: bool,
    ) -> Result<(StatusCode, Option<bytes::Bytes>), Response> {
        // Check prefill result first - it's critical for disaggregated mode
        let prefill_response = match prefill_result {
            Ok(response) => response,
            Err(e) => {
                error!(
                    "Prefill server failed (CRITICAL) prefill_url={} error={}. Decode will timeout without prefill KV cache.",
                    prefill_url,
                    e
                );

                // Return error immediately - don't wait for decode to timeout
                return Err(error::bad_gateway(
                    "prefill_server_error",
                    format!(
                        "Prefill server error: {}. This will cause decode timeout.",
                        e
                    ),
                ));
            }
        };

        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Check if prefill succeeded
        if !prefill_status.is_success() {
            // Get error body from prefill
            let error_msg = prefill_response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            error!(
                "Prefill server returned error status prefill_url={} status={} body={}",
                prefill_url, prefill_status, error_msg
            );

            // Map prefill_status to appropriate error function
            let error_response = match prefill_status {
                StatusCode::BAD_REQUEST => error::bad_request(
                    "prefill_bad_request",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::NOT_FOUND => error::not_found(
                    "prefill_not_found",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::INTERNAL_SERVER_ERROR => error::internal_error(
                    "prefill_internal_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::SERVICE_UNAVAILABLE => error::service_unavailable(
                    "prefill_unavailable",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                StatusCode::BAD_GATEWAY => error::bad_gateway(
                    "prefill_bad_gateway",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
                _ => error::internal_error(
                    "prefill_error",
                    format!("Prefill server error ({}): {}", prefill_status, error_msg),
                ),
            };
            return Err(error_response);
        }

        // Read prefill body if needed for logprob merging
        let prefill_body = if return_logprob {
            match prefill_response.bytes().await {
                Ok(body) => Some(body),
                Err(e) => {
                    warn!("Failed to read prefill response body for logprobs: {}", e);
                    None
                }
            }
        } else {
            // For non-logprob requests, just consume the response without storing
            debug!("Consuming prefill response body (non-logprob request)");
            match prefill_response.bytes().await {
                Ok(_) => debug!("Prefill response consumed successfully"),
                Err(e) => warn!("Error consuming prefill response: {}", e),
            }
            None
        };

        Ok((prefill_status, prefill_body))
    }

    fn build_post_with_headers(
        &self,
        client: &Client,
        url: &str,
        route: &'static str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let mut request = client.post(api_path(url, route)).json(json_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
                if header_utils::should_forward_request_header(name.as_str()) {
                    if let Ok(val) = value.to_str() {
                        request = request.header(name, val);
                    }
                }
            }
        }
        request
    }

    // Helper to merge logprobs from prefill and decode responses
    // Optimized to avoid double cloning by taking ownership of decode array
    fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        if let (Some(prefill_meta), Some(decode_meta)) = (
            prefill_json.get("meta_info"),
            decode_json.get_mut("meta_info"),
        ) {
            if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                prefill_meta.get("input_token_logprobs"),
                decode_meta.get_mut("input_token_logprobs"),
            ) {
                if let Some(prefill_arr) = prefill_logprobs.as_array() {
                    // Take ownership of decode array to avoid cloning it
                    let decode_arr = std::mem::take(decode_logprobs);
                    if let Value::Array(decode_vec) = decode_arr {
                        // Pre-allocate merged array with exact capacity
                        let mut merged = Vec::with_capacity(prefill_arr.len() + decode_vec.len());
                        merged.extend(prefill_arr.iter().cloned());
                        merged.extend(decode_vec);
                        decode_meta["input_token_logprobs"] = Value::Array(merged);
                        return true;
                    }
                }
            }
        }
        false
    }

    // Simple helper to merge logprobs in streaming responses
    // Optimized to reduce allocations in the merge path
    fn merge_streaming_logprobs(
        prefill_logprobs: Option<Value>,
        decode_chunk: &[u8],
    ) -> Result<bytes::Bytes, ()> {
        // Skip non-data chunks
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        if !chunk_str.starts_with("data: ") || chunk_str.contains("[DONE]") {
            return Err(());
        }

        // Parse JSON from chunk
        let json_str = chunk_str.trim_start_matches("data: ").trim();
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        // Merge prefill logprobs if available
        if let Some(ref p_logprobs) = prefill_logprobs {
            if let Some(meta) = decode_json.get_mut("meta_info") {
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    if let Some(p_arr) = p_logprobs.as_array() {
                        // Take ownership of decode array to avoid cloning it
                        let decode_arr = std::mem::take(d_logprobs);
                        if let Value::Array(d_vec) = decode_arr {
                            // Pre-allocate merged array with exact capacity
                            let mut merged = Vec::with_capacity(p_arr.len() + d_vec.len());
                            merged.extend(p_arr.iter().cloned());
                            merged.extend(d_vec);
                            *d_logprobs = Value::Array(merged);
                        }
                    }
                }
            }
        }

        // Re-serialize
        let merged_str = format!(
            "data: {}\n\n",
            serde_json::to_string(&decode_json).unwrap_or_default()
        );
        Ok(bytes::Bytes::from(merged_str))
    }
}

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // Select a random worker pair using the policy
        let (prefill, decode) = match self.select_pd_pair(None, None, None).await {
            Ok(pair) => pair,
            Err(e) => {
                return error::service_unavailable(
                    "no_healthy_worker_pair",
                    format!("No healthy worker pair available: {}", e),
                );
            }
        };

        let prefill_url = format!("{}/health_generate", prefill.url());
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client
                .get(format!("{}/health_generate", decode.url()))
                .send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            error::service_unavailable(
                "health_generate_failed",
                format!("Health generate failed: {:?}", errors),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server to match sglang's server info format
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("get_server_info", None)
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("get_model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.return_logprob.unwrap_or(false);

        let request_text = if self.policies_need_request_text() {
            body.text.as_deref().map(|s| s.to_string())
        } else {
            None
        };

        let batch_size = Self::get_generate_batch_size(body);

        let context = PDRequestContext {
            route: "/generate",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
            raw_request: None,
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_generate_with_raw(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
        raw_body: Option<&Value>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.return_logprob.unwrap_or(false);

        let request_text = if self.policies_need_request_text() {
            body.text.as_deref().map(|s| s.to_string())
        } else {
            None
        };

        let batch_size = Self::get_generate_batch_size(body);

        let context = PDRequestContext {
            route: "/generate",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
            raw_request: raw_body.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() {
            body.messages.first().and_then(|msg| match msg {
                ChatMessage::User { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::Developer { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::System { content, .. } => Some(content.to_simple_string()),
                _ => None,
            })
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_chat_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
            raw_request: None,
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_chat_with_raw(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
        raw_body: Option<&Value>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() {
            body.messages.first().and_then(|msg| match msg {
                ChatMessage::User { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::Developer { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::System { content, .. } => Some(content.to_simple_string()),
                _ => None,
            })
        } else {
            None
        };

        let batch_size = Self::get_chat_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
            raw_request: raw_body.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_completion_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
            raw_request: None,
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_completion_with_raw(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
        raw_body: Option<&Value>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        let batch_size = Self::get_completion_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned(),
            raw_request: raw_body.cloned(),
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Extract text for cache-aware routing
        let req_text = if self.policies_need_request_text() {
            Some(body.query.clone())
        } else {
            None
        };

        let context = PDRequestContext {
            route: "/v1/rerank",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            request_text: req_text,
            model_id,
            headers: headers.cloned(),
            raw_request: None,
        };

        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/embeddings; returning bad request");
        error::bad_request(
            "pd_unsupported_embeddings",
            "PD mode does not support /v1/embeddings",
        )
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        let _ = (headers, body, model_id);
        warn!("PD mode does not support /v1/classify; returning bad request");
        error::bad_request(
            "pd_unsupported_classify",
            "PD mode does not support /v1/classify",
        )
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};
    use serde::Serialize;

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            api_key: Some("test_api_key".to_string()),
            enable_igw: false,
            hexgent: HexgentConfig::default(),
            hexgent_snapshot_cache: Arc::new(DashMap::new()),
            hexgent_snapshot_refreshing: Arc::new(DashSet::new()),
            hexgent_admission_queue: Arc::new(Mutex::new(HexgentGlobalAdmissionQueue::new(1))),
            hexgent_admission_waiters: Arc::new(DashMap::new()),
        }
    }

    fn test_hexgent_decision(
        revision: u64,
        prefill_worker_url: &str,
        prefill_priority: i64,
    ) -> HexgentPlannerResponse {
        HexgentPlannerResponse {
            revision,
            workflow_horizon_ms: Some(100),
            prefill_worker_id: prefill_worker_url.to_string(),
            prefill_worker_url: prefill_worker_url.to_string(),
            decode_worker_id: "http://decode".to_string(),
            decode_worker_url: "http://decode".to_string(),
            prefill_priority,
            decode_priority: prefill_priority,
            planned_decode_instance: "http://decode".to_string(),
            predicted_output_tokens: 16,
            actions: vec![],
            fallback_reason: None,
        }
    }

    #[test]
    fn test_hexgent_global_admission_releases_latest_planned_priority() {
        let mut queue = HexgentGlobalAdmissionQueue::new(1);
        let first = HexgentAdmissionKey::new("wf", "running");
        let low_priority = HexgentAdmissionKey::new("wf", "low");
        let high_priority = HexgentAdmissionKey::new("wf", "high");

        let granted = queue.enqueue(
            first.clone(),
            test_hexgent_decision(1, "http://prefill", 100),
        );
        assert_eq!(granted.len(), 1);
        assert_eq!(granted[0].key, first);

        assert!(queue
            .enqueue(
                low_priority.clone(),
                test_hexgent_decision(1, "http://prefill", 100),
            )
            .is_empty());
        assert!(queue
            .enqueue(
                high_priority.clone(),
                test_hexgent_decision(1, "http://prefill", 100),
            )
            .is_empty());

        assert!(queue
            .publish_plan(
                &low_priority,
                test_hexgent_decision(2, "http://prefill", 20),
            )
            .is_empty());
        assert!(queue
            .publish_plan(
                &high_priority,
                test_hexgent_decision(3, "http://prefill", 5),
            )
            .is_empty());

        let granted = queue.release_prefill("http://prefill");
        assert_eq!(granted.len(), 1);
        assert_eq!(granted[0].key, high_priority);
        assert_eq!(granted[0].decision.revision, 3);
        assert_eq!(granted[0].decision.prefill_priority, 5);
    }

    #[test]
    fn test_hexgent_global_admission_does_not_retain_immediately_granted_fallback() {
        let mut queue = HexgentGlobalAdmissionQueue::new(1);
        let key = HexgentAdmissionKey::new("wf", "immediate");

        let granted = queue.enqueue(
            key.clone(),
            test_hexgent_decision(0, "http://prefill0", 0),
        );

        assert_eq!(granted.len(), 1);
        assert!(!queue.is_pending(&key));
    }

    #[test]
    fn test_hexgent_global_admission_holds_fallback_when_its_prefill_is_busy() {
        let mut queue = HexgentGlobalAdmissionQueue::new(32);
        let running = HexgentAdmissionKey::new("wf", "running");
        let queued = HexgentAdmissionKey::new("wf", "queued");
        let fallback = test_hexgent_decision(0, "http://prefill0", 0);

        assert_eq!(queue.enqueue(running, fallback.clone()).len(), 1);
        assert!(queue.enqueue(queued.clone(), fallback).is_empty());
        assert!(queue.is_pending(&queued));
    }

    #[test]
    fn test_hexgent_snapshot_cache_only_returns_fresh_entries() {
        let router = create_test_pd_router();
        let worker_url = "http://prefill";
        router.hexgent_snapshot_cache.insert(
            worker_url.to_string(),
            HexgentCachedSnapshot {
                snapshot: json!({"queues": {"waiting": []}}),
                captured_at: Instant::now(),
            },
        );

        assert_eq!(
            router.cached_hexgent_worker_snapshot(worker_url, Duration::from_millis(25)),
            Some(json!({"queues": {"waiting": []}})),
        );

        router.hexgent_snapshot_cache.insert(
            worker_url.to_string(),
            HexgentCachedSnapshot {
                snapshot: json!({"queues": {"waiting": ["stale"]}}),
                captured_at: Instant::now() - Duration::from_secs(1),
            },
        );
        assert!(router
            .cached_hexgent_worker_snapshot(worker_url, Duration::from_millis(25))
            .is_none());
    }

    #[test]
    fn test_hexgent_admission_fetches_snapshot_only_inside_admission_loop() {
        let mut router = create_test_pd_router();
        assert!(router.fetch_hexgent_snapshot_before_admission());

        router.hexgent.admission_queue_enabled = true;
        assert!(!router.fetch_hexgent_snapshot_before_admission());
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        let healthy_worker = create_test_worker(
            "http://healthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let unhealthy_worker = create_test_worker(
            "http://unhealthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            false,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(unhealthy_worker));
        router.worker_registry.register(Arc::from(healthy_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[test]
    fn test_inject_hexgent_plan_into_value_adds_worker_metadata() {
        let mut request = serde_json::json!({
            "model": "test-model",
            "hexgent": {
                "workflow_id": "wf-1",
                "call_id": "call-1",
                "workflow_arrival_ms": 10,
                "workflow_horizon_ms": 1000,
                "parent_call_ids": ["parent-1"],
                "reveal_sequence": 2,
                "input_tokens": 33,
                "predicted_output_tokens": 44,
                "estimated_prefill_time": 11.5,
                "estimated_decode_time": 22.5,
                "estimated_transfer_time": 3.5
            }
        });
        let decision = HexgentPlannerResponse {
            revision: 7,
            workflow_horizon_ms: Some(35),
            prefill_worker_id: "http://prefill".to_string(),
            prefill_worker_url: "http://prefill".to_string(),
            decode_worker_id: "http://decode".to_string(),
            decode_worker_url: "http://decode".to_string(),
            prefill_priority: 3,
            decode_priority: 5,
            planned_decode_instance: "http://decode".to_string(),
            predicted_output_tokens: 64,
            actions: vec![],
            fallback_reason: None,
        };

        PDRouter::inject_hexgent_plan_into_value(&mut request, &decision);

        assert_eq!(request["workflow_id"], "wf-1");
        assert_eq!(request["hexgent_call_id"], "call-1");
        assert_eq!(request["hexgent_workflow_arrival_ms"], 10);
        assert_eq!(
            request["hexgent_parent_call_ids"],
            serde_json::json!(["parent-1"])
        );
        assert_eq!(request["hexgent_workflow_horizon_ms"], 35);
        assert_eq!(request["hexgent_reveal_sequence"], 2);
        assert_eq!(request["hexgent_input_tokens"], 33);
        assert_eq!(request["hexgent_estimated_prefill_time"], 11.5);
        assert_eq!(request["hexgent_estimated_decode_time"], 22.5);
        assert_eq!(request["hexgent_estimated_transfer_time"], 3.5);
        assert_eq!(request["priority"], 3);
        assert_eq!(request["hexgent_prefill_priority"], 3);
        assert_eq!(request["hexgent_decode_priority"], 5);
        assert_eq!(request["hexgent_schedule_revision"], 7);
        assert_eq!(request["planned_decode_instance"], "http://decode");
        assert_eq!(request["hexgent_predicted_output_tokens"], 64);
    }

    #[test]
    fn test_request_value_for_dispatch_preserves_raw_hexgent_metadata() {
        #[derive(Serialize)]
        struct TypedRequest {
            model: String,
        }

        let typed = TypedRequest {
            model: "test-model".to_string(),
        };
        let raw = serde_json::json!({
            "model": "test-model",
            "hexgent": {
                "workflow_id": "wf-raw",
                "call_id": "call-raw",
                "predicted_output_tokens": 48
            }
        });

        let dispatch_value =
            PDRouter::request_value_for_dispatch(&typed, Some(&raw)).expect("raw body is valid");

        assert_eq!(dispatch_value["hexgent"]["workflow_id"], "wf-raw");
        assert_eq!(dispatch_value["hexgent"]["call_id"], "call-raw");
        assert_eq!(dispatch_value["hexgent"]["predicted_output_tokens"], 48);
    }

    #[test]
    fn test_max_tokens_from_request_supports_chat_max_completion_tokens() {
        let request = serde_json::json!({
            "model": "test-model",
            "max_completion_tokens": 96
        });

        assert_eq!(PDRouter::max_tokens_from_request(&request), Some(96));
    }

    #[test]
    fn test_hexgent_reassign_payload_uses_worker_mutation_schema() {
        let action = HexgentWorkerMutationAction {
            worker_url: "http://decode".to_string(),
            request_id: "call-1".to_string(),
            target_queue_name: "decode".to_string(),
            priority: 11,
            planned_decode_instance: Some("http://decode".to_string()),
            schedule_revision: 17,
            expected_revision: Some(16),
            target_worker_id: Some("decode-1".to_string()),
            target_worker_url: Some("http://decode-target".to_string()),
        };

        let payload = PDRouter::hexgent_reassign_payload(&action);

        assert_eq!(payload["request_id"], "call-1");
        assert_eq!(payload["target_queue_name"], "decode");
        assert_eq!(payload["priority"], 11);
        assert_eq!(payload["planned_decode_instance"], "http://decode");
        assert_eq!(payload["schedule_revision"], 17);
        assert_eq!(payload["expected_revision"], 16);
        assert_eq!(payload["requeue_locally"], true);
    }

    #[test]
    fn test_hexgent_worker_mutation_action_reads_target_metadata() {
        let action: HexgentWorkerMutationAction = serde_json::from_value(serde_json::json!({
            "worker_url": "http://prefill-source",
            "request_id": "call-1",
            "target_queue_name": "prefill_waiting",
            "priority": 11,
            "planned_decode_instance": "decode-1",
            "schedule_revision": 17,
            "expected_revision": 16,
            "target_worker_id": "prefill-target",
            "target_worker_url": "http://prefill-target"
        }))
        .expect("mutation action should deserialize");

        assert_eq!(action.expected_revision, Some(16));
        assert_eq!(action.target_worker_id.as_deref(), Some("prefill-target"));
        assert_eq!(
            action.target_worker_url.as_deref(),
            Some("http://prefill-target")
        );
    }

    #[test]
    fn test_hexgent_planner_request_serializes_worker_snapshots() {
        let mut worker_snapshots = HashMap::new();
        worker_snapshots.insert(
            "http://prefill".to_string(),
            serde_json::json!({
                "mode": "prefill",
                "queues": {
                    "waiting": [
                        {
                            "rid": "call-1",
                            "input_tokens": 16
                        }
                    ]
                }
            }),
        );

        let payload = HexgentPlannerRequest {
            route: "/v1/chat/completions",
            model: Some("test-model".to_string()),
            request_text: "hello".to_string(),
            max_tokens: Some(32),
            hexgent: serde_json::json!({
                "workflow_id": "wf-1",
                "call_id": "call-1"
            }),
            prefill_workers: vec![],
            decode_workers: vec![],
            state: HexgentPlannerState {
                current_time_ms: 123,
                worker_snapshots: Some(worker_snapshots),
            },
        };

        let value = serde_json::to_value(payload).expect("planner request serializes");

        assert_eq!(value["state"]["current_time_ms"], 123);
        assert_eq!(
            value["state"]["worker_snapshots"]["http://prefill"]["queues"]["waiting"][0]["rid"],
            "call-1"
        );
    }

    #[test]
    fn test_hexgent_admission_poll_refreshes_live_cluster_state() {
        let mut payload = HexgentPlannerRequest {
            route: "/v1/chat/completions",
            model: Some("test-model".to_string()),
            request_text: "hello".to_string(),
            max_tokens: Some(8),
            hexgent: serde_json::json!({
                "workflow_id": "wf-refresh",
                "call_id": "call-0"
            }),
            prefill_workers: vec![],
            decode_workers: vec![],
            state: HexgentPlannerState {
                current_time_ms: 10,
                worker_snapshots: Some(HashMap::from([(
                    "http://prefill".to_string(),
                    serde_json::json!({"running": [{"rid": "old"}]}),
                )])),
            },
        };
        let refreshed_snapshots = HashMap::from([(
            "http://prefill".to_string(),
            serde_json::json!({"running": []}),
        )]);

        PDRouter::refresh_hexgent_admission_state(&mut payload, 25, Some(refreshed_snapshots));

        assert_eq!(payload.state.current_time_ms, 25);
        assert_eq!(
            payload.state.worker_snapshots,
            Some(HashMap::from([(
                "http://prefill".to_string(),
                serde_json::json!({"running": []}),
            )]))
        );

        PDRouter::refresh_hexgent_admission_state(&mut payload, 30, None);

        assert_eq!(payload.state.current_time_ms, 30);
        assert_eq!(
            payload.state.worker_snapshots,
            Some(HashMap::from([(
                "http://prefill".to_string(),
                serde_json::json!({"running": []}),
            )]))
        );
    }

    #[test]
    fn test_hexgent_admission_request_wraps_planner_request_and_policy() {
        let payload = HexgentPlannerRequest {
            route: "/v1/chat/completions",
            model: Some("test-model".to_string()),
            request_text: "hello".to_string(),
            max_tokens: Some(32),
            hexgent: serde_json::json!({
                "workflow_id": "wf-1",
                "call_id": "call-1"
            }),
            prefill_workers: vec![],
            decode_workers: vec![],
            state: HexgentPlannerState {
                current_time_ms: 123,
                worker_snapshots: None,
            },
        };
        let admission = HexgentAdmissionRequest {
            request: payload,
            policy: HexgentAdmissionPolicy {
                max_prefill_queue_per_worker: 2,
                retry_after_ms: 17,
            },
        };

        let value = serde_json::to_value(admission).expect("admission request serializes");

        assert_eq!(value["request"]["hexgent"]["workflow_id"], "wf-1");
        assert_eq!(value["policy"]["max_prefill_queue_per_worker"], 2);
        assert_eq!(value["policy"]["retry_after_ms"], 17);
    }

    #[test]
    fn test_hexgent_admission_status_query_uses_workflow_and_call_ids() {
        let payload = HexgentPlannerRequest {
            route: "/v1/chat/completions",
            model: Some("test-model".to_string()),
            request_text: "hello".to_string(),
            max_tokens: Some(32),
            hexgent: serde_json::json!({
                "workflow_id": "wf status",
                "call_id": "call/1"
            }),
            prefill_workers: vec![],
            decode_workers: vec![],
            state: HexgentPlannerState {
                current_time_ms: 123,
                worker_snapshots: None,
            },
        };

        assert_eq!(
            PDRouter::hexgent_admission_status_query(&payload).expect("valid metadata"),
            vec![
                ("workflow_id", "wf status".to_string()),
                ("call_id", "call/1".to_string()),
            ]
        );
    }

    #[test]
    fn test_hexgent_admission_wait_query_carries_the_admission_deadline() {
        let payload = HexgentPlannerRequest {
            route: "/v1/chat/completions",
            model: Some("test-model".to_string()),
            request_text: "hello".to_string(),
            max_tokens: Some(64),
            hexgent: serde_json::json!({
                "workflow_id": "wf wait",
                "call_id": "call/1",
            }),
            prefill_workers: vec![],
            decode_workers: vec![],
            state: HexgentPlannerState {
                current_time_ms: 0,
                worker_snapshots: None,
            },
        };

        assert_eq!(
            PDRouter::hexgent_admission_wait_query(&payload, 300_000).expect("valid metadata"),
            vec![
                ("workflow_id", "wf wait".to_string()),
                ("call_id", "call/1".to_string()),
                ("timeout_ms", "300000".to_string()),
            ]
        );
    }

    #[test]
    fn test_hexgent_fallback_decision_preserves_the_selected_pd_pair() {
        let request = serde_json::json!({
            "max_tokens": 64,
            "hexgent": {
                "workflow_id": "wf-fallback",
                "call_id": "call-0",
                "predicted_output_tokens": 12
            }
        });
        let prefill = create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode = create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        let decision = PDRouter::hexgent_fallback_decision(
            &request,
            prefill.as_ref(),
            decode.as_ref(),
            "planner_grace_expired".to_string(),
        );

        assert_eq!(decision.prefill_worker_url, "http://prefill");
        assert_eq!(decision.decode_worker_url, "http://decode");
        assert_eq!(decision.predicted_output_tokens, 12);
        assert_eq!(
            decision.fallback_reason.as_deref(),
            Some("planner_grace_expired")
        );
    }

    #[test]
    fn test_hexgent_completion_event_payload_uses_request_and_decision_metadata() {
        let request = serde_json::json!({
            "hexgent": {
                "workflow_id": "wf-1",
                "call_id": "call-1"
            }
        });
        let decision = HexgentPlannerResponse {
            revision: 7,
            workflow_horizon_ms: None,
            prefill_worker_id: "http://prefill".to_string(),
            prefill_worker_url: "http://prefill".to_string(),
            decode_worker_id: "http://decode".to_string(),
            decode_worker_url: "http://decode".to_string(),
            prefill_priority: 3,
            decode_priority: 5,
            planned_decode_instance: "http://decode".to_string(),
            predicted_output_tokens: 64,
            actions: vec![],
            fallback_reason: None,
        };

        let payload = PDRouter::hexgent_completion_event_payload(
            &request,
            &decision,
            42,
            None,
            1234,
            StatusCode::OK,
        )
        .expect("hexgent metadata should produce an event payload");

        assert_eq!(payload["event_type"], "call_completed");
        assert_eq!(payload["workflow_id"], "wf-1");
        assert_eq!(payload["call_id"], "call-1");
        assert!(payload.get("service_ms").is_none());
        assert_eq!(payload["end_to_end_latency_ms"], 42);
        assert!(payload.get("measured_service_ms").is_none());
        assert_eq!(payload["timestamp_ms"], 1234);
        assert_eq!(payload["payload"]["status_code"], 200);
        assert_eq!(payload["payload"]["prefill_worker_url"], "http://prefill");
        assert_eq!(payload["payload"]["decode_worker_url"], "http://decode");
        assert_eq!(payload["payload"]["schedule_revision"], 7);
        assert_eq!(payload["payload"]["predicted_output_tokens"], 64);
        assert_eq!(
            payload["payload"]["hexgent_metadata"]["workflow_id"],
            "wf-1"
        );
    }

    #[test]
    fn test_hexgent_prefill_completion_event_preserves_the_plan_identity() {
        let request = serde_json::json!({
            "hexgent": {
                "workflow_id": "wf-prefill",
                "call_id": "call-0"
            }
        });
        let decision = HexgentPlannerResponse {
            revision: 9,
            workflow_horizon_ms: None,
            prefill_worker_id: "prefill-0".to_string(),
            prefill_worker_url: "http://prefill".to_string(),
            decode_worker_id: "decode-0".to_string(),
            decode_worker_url: "http://decode".to_string(),
            prefill_priority: 2,
            decode_priority: 4,
            planned_decode_instance: "decode-0".to_string(),
            predicted_output_tokens: 16,
            actions: vec![],
            fallback_reason: None,
        };

        let payload = PDRouter::hexgent_prefill_completion_event_payload(&request, &decision, 1234)
            .expect("hexgent metadata should produce an event payload");

        assert_eq!(payload["event_type"], "prefill_completed");
        assert_eq!(payload["workflow_id"], "wf-prefill");
        assert_eq!(payload["call_id"], "call-0");
        assert_eq!(payload["timestamp_ms"], 1234);
        assert_eq!(payload["payload"]["schedule_revision"], 9);
        assert_eq!(payload["payload"]["prefill_worker_url"], "http://prefill");
    }

    #[test]
    fn test_hexgent_request_cancellation_event_releases_the_published_plan() {
        let request = serde_json::json!({
            "hexgent": {
                "workflow_id": "wf-cancel",
                "call_id": "call-0"
            }
        });
        let decision = HexgentPlannerResponse {
            revision: 10,
            workflow_horizon_ms: None,
            prefill_worker_id: "prefill-0".to_string(),
            prefill_worker_url: "http://prefill".to_string(),
            decode_worker_id: "decode-0".to_string(),
            decode_worker_url: "http://decode".to_string(),
            prefill_priority: 2,
            decode_priority: 4,
            planned_decode_instance: "decode-0".to_string(),
            predicted_output_tokens: 16,
            actions: vec![],
            fallback_reason: None,
        };

        let payload = PDRouter::hexgent_request_cancelled_event_payload(
            &request,
            &decision,
            "decode_transport_error",
            1234,
        )
        .expect("hexgent metadata should produce a cancellation payload");

        assert_eq!(payload["event_type"], "request_cancelled");
        assert_eq!(payload["workflow_id"], "wf-cancel");
        assert_eq!(payload["call_id"], "call-0");
        assert_eq!(payload["timestamp_ms"], 1234);
        assert_eq!(payload["payload"]["reason"], "decode_transport_error");
        assert_eq!(payload["payload"]["schedule_revision"], 10);
    }

    #[test]
    fn test_hexgent_completion_event_payload_uses_only_measured_stage_service() {
        let request = serde_json::json!({
            "hexgent": {
                "workflow_id": "wf-telemetry",
                "call_id": "call-telemetry"
            }
        });
        let decision = HexgentPlannerResponse {
            revision: 1,
            workflow_horizon_ms: None,
            prefill_worker_id: "http://prefill".to_string(),
            prefill_worker_url: "http://prefill".to_string(),
            decode_worker_id: "http://decode".to_string(),
            decode_worker_url: "http://decode".to_string(),
            prefill_priority: 1,
            decode_priority: 1,
            planned_decode_instance: "http://decode".to_string(),
            predicted_output_tokens: 8,
            actions: vec![],
            fallback_reason: None,
        };
        let stage_timing = serde_json::json!({
            "prefill_execution_ms": 12,
            "transfer_ms": 3,
            "decode_execution_ms": 45,
            "measured_service_ms": 60,
        });

        let payload = PDRouter::hexgent_completion_event_payload(
            &request,
            &decision,
            250,
            Some(stage_timing),
            999,
            StatusCode::OK,
        )
        .expect("hexgent metadata should produce an event payload");

        assert_eq!(payload["end_to_end_latency_ms"], 250);
        assert_eq!(payload["measured_service_ms"], 60);
        assert_eq!(payload["payload"]["stage_timing"]["transfer_ms"], 3);
    }

    #[test]
    fn test_inject_routing_observability_headers() {
        let mut response = StatusCode::OK.into_response();
        let decision = HexgentPlannerResponse {
            revision: 7,
            workflow_horizon_ms: None,
            prefill_worker_id: "prefill-0".to_string(),
            prefill_worker_url: "http://prefill:31000".to_string(),
            decode_worker_id: "decode-0".to_string(),
            decode_worker_url: "http://decode:32000".to_string(),
            prefill_priority: 3,
            decode_priority: 5,
            planned_decode_instance: "decode-0".to_string(),
            predicted_output_tokens: 64,
            actions: vec![],
            fallback_reason: None,
        };

        PDRouter::inject_routing_observability_headers(
            &mut response,
            "http://prefill:31000",
            "http://decode:32000",
            Some(&decision),
            Duration::from_micros(2500),
            Some(&HexgentSelectionTiming {
                snapshot_fetch: Duration::from_micros(800),
                admission_wait: Duration::from_micros(1600),
            }),
        );

        assert_eq!(
            response.headers()["x-sglang-prefill-worker"],
            "http://prefill:31000"
        );
        assert_eq!(
            response.headers()["x-sglang-decode-worker"],
            "http://decode:32000"
        );
        assert_eq!(response.headers()["x-hexgent-router-mode"], "hexgent");
        assert_eq!(response.headers()["x-hexgent-schedule-revision"], "7");
        assert_eq!(response.headers()["x-hexgent-selection-wall-us"], "2500");
        assert_eq!(response.headers()["x-hexgent-snapshot-fetch-us"], "800");
        assert_eq!(response.headers()["x-hexgent-admission-wait-us"], "1600");
    }

    #[test]
    fn test_hexgent_selection_timing_keeps_snapshot_and_wait_separate() {
        let timing = HexgentSelectionTiming::from_admission_phase(
            Duration::from_micros(800),
            Duration::from_micros(1600),
        );

        assert_eq!(timing.snapshot_fetch, Duration::from_micros(800));
        assert_eq!(timing.admission_wait, Duration::from_micros(1600));
    }

    #[test]
    fn test_worker_payload_reads_hexgent_latency_labels() {
        let mut labels = HashMap::new();
        labels.insert("hardware".to_string(), "H100_80G".to_string());
        labels.insert("hexgent_capacity_tokens".to_string(), "4096".to_string());
        labels.insert("hexgent_prefill_base_ms".to_string(), "3.5".to_string());
        labels.insert(
            "hexgent_prefill_ms_per_input_token".to_string(),
            "0.25".to_string(),
        );
        labels.insert("hexgent_decode_base_ms".to_string(), "2.5".to_string());
        labels.insert(
            "hexgent_decode_ms_per_output_token".to_string(),
            "8.25".to_string(),
        );
        labels.insert(
            "hexgent_transfer_ms_per_token".to_string(),
            "0.05".to_string(),
        );
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://prefill".to_string())
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .labels(labels)
                .build(),
        );

        let payload = PDRouter::worker_payload(&[worker], "prefill");
        let item = &payload[0];

        assert_eq!(item.hardware.as_deref(), Some("H100_80G"));
        assert_eq!(item.capacity_tokens, 4096);
        assert_eq!(item.prefill_base_ms, 3.5);
        assert_eq!(item.prefill_ms_per_input_token, 0.25);
        assert_eq!(item.decode_base_ms, 2.5);
        assert_eq!(item.decode_ms_per_output_token, 8.25);
        assert_eq!(item.transfer_ms_per_token, 0.05);
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router.select_pd_pair(None, None, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    #[test]
    fn test_worker_load_metrics() {
        let prefill_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        ));
        let decode_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let _prefill_guard = WorkerLoadGuard::new(prefill_worker.clone(), None);
        let _decode_guard = WorkerLoadGuard::new(decode_worker.clone(), None);

        assert_eq!(prefill_worker.load(), 1);
        assert_eq!(decode_worker.load(), 1);

        drop(_prefill_guard);
        drop(_decode_guard);

        assert_eq!(prefill_worker.load(), 0);
        assert_eq!(decode_worker.load(), 0);
    }

    #[tokio::test]
    async fn test_streaming_load_tracking() {
        use futures_util::StreamExt;
        use tokio::time::{sleep, Duration};

        let router = create_test_pd_router();

        let prefill_worker = create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();

        let prefill_ref = prefill_workers[0].clone();
        let decode_ref = decode_workers[0].clone();

        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);

        {
            let response = router.create_streaming_response(
                stream.map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                None,
                prefill_ref.clone(),
                decode_ref.clone(),
                None,
            );

            // Guards are now attached to response body, so load should be 1
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            tx.send(bytes::Bytes::from("test data")).unwrap();

            sleep(Duration::from_millis(10)).await;

            // Load still 1 while response body exists
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            drop(tx);

            // Response (and its body with guards) dropped here
            drop(response);
        }

        // Guards dropped when response dropped
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);
    }
}
