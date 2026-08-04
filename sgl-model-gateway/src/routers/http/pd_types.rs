//! Types and utilities for the prefill-decode (PD) disaggregated router.

use std::collections::HashMap;

/// Custom error type for PD router operations
#[derive(Debug, thiserror::Error)]
pub enum PDRouterError {
    #[error("Worker already exists: {url}")]
    WorkerAlreadyExists { url: String },

    #[error("Worker not found: {url}")]
    WorkerNotFound { url: String },

    #[error("Lock acquisition failed: {operation}")]
    LockError { operation: String },

    #[error("Health check failed for worker: {url}")]
    HealthCheckFailed { url: String },

    #[error("Invalid worker configuration: {reason}")]
    InvalidConfiguration { reason: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Timeout waiting for worker: {url}")]
    Timeout { url: String },
}

/// Construct a full API URL from a base URL and path.
pub fn api_path(url: &str, api_path: &str) -> String {
    if api_path.starts_with('/') {
        format!("{}{}", url, api_path)
    } else {
        format!("{}/{}", url, api_path)
    }
}

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Optimized bootstrap wrapper for single requests.
#[derive(Serialize)]
pub struct RequestWithBootstrap<'a, T: Serialize> {
    #[serde(flatten)]
    pub original: &'a T,
    pub bootstrap_host: String,
    pub bootstrap_port: Option<u16>,
    pub bootstrap_room: u64,
}

/// Optimized bootstrap wrapper for batch requests.
#[derive(Serialize)]
pub struct BatchRequestWithBootstrap<'a, T: Serialize> {
    #[serde(flatten)]
    pub original: &'a T,
    pub bootstrap_host: Vec<String>,
    pub bootstrap_port: Vec<Option<u16>>,
    pub bootstrap_room: Vec<u64>,
}

/// Generate a random bootstrap room ID.
pub fn generate_room_id() -> u64 {
    // Generate a value in the range [0, 2^63 - 1] to match Python's random.randint(0, 2**63 - 1)
    rand::random::<u64>() & (i64::MAX as u64)
}

#[derive(Clone, Debug, Serialize)]
pub struct HexgentPlannerWorker {
    pub id: String,
    pub url: String,
    pub worker_type: String,
    pub hardware: Option<String>,
    pub available_at_ms: u64,
    pub capacity_tokens: u64,
    pub prefill_base_ms: f64,
    pub prefill_ms_per_input_token: f64,
    pub decode_base_ms: f64,
    pub decode_ms_per_output_token: f64,
    pub transfer_ms_per_token: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HexgentPlannerState {
    pub current_time_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_snapshots: Option<HashMap<String, Value>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HexgentPlannerRequest {
    pub route: &'static str,
    pub model: Option<String>,
    pub request_text: String,
    pub max_tokens: Option<u64>,
    pub hexgent: Value,
    pub prefill_workers: Vec<HexgentPlannerWorker>,
    pub decode_workers: Vec<HexgentPlannerWorker>,
    pub state: HexgentPlannerState,
}

#[derive(Clone, Debug, Serialize)]
pub struct HexgentAdmissionPolicy {
    pub max_prefill_queue_per_worker: u64,
    pub retry_after_ms: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HexgentAdmissionRequest {
    pub request: HexgentPlannerRequest,
    pub policy: HexgentAdmissionPolicy,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HexgentPlannerResponse {
    pub revision: u64,
    #[serde(default)]
    pub workflow_horizon_ms: Option<u64>,
    pub prefill_worker_id: String,
    pub prefill_worker_url: String,
    pub decode_worker_id: String,
    pub decode_worker_url: String,
    pub prefill_priority: i64,
    pub decode_priority: i64,
    pub planned_decode_instance: String,
    pub predicted_output_tokens: u64,
    #[serde(default)]
    pub actions: Vec<HexgentWorkerMutationAction>,
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HexgentAdmissionResponse {
    pub dispatch_ready: bool,
    pub retry_after_ms: u64,
    pub queue_position: u64,
    pub reason: String,
    pub schedule: Option<HexgentPlannerResponse>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HexgentWorkerMutationAction {
    pub worker_url: String,
    pub request_id: String,
    pub target_queue_name: String,
    pub priority: i64,
    pub planned_decode_instance: Option<String>,
    pub schedule_revision: u64,
    #[serde(default)]
    pub expected_revision: Option<u64>,
    #[serde(default)]
    pub target_worker_id: Option<String>,
    #[serde(default)]
    pub target_worker_url: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct HexgentPlannerEvent {
    pub event_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workflow_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_to_end_latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measured_service_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_ms: Option<u64>,
    pub payload: Value,
}

/// PD-specific routing policies.
#[derive(Debug, Clone, PartialEq)]
pub enum PDSelectionPolicy {
    Random,
    PowerOfTwo,
    CacheAware {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
    },
    Bucket {
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        bucket_adjust_interval_secs: usize,
    },
}
