//! Worker mutation outcomes used by asynchronous gateway delivery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MutationOutcome {
    Applied,
    Obsolete,
    Retryable,
}

pub(crate) fn classify_worker_mutation(
    status: u16,
    success: Option<bool>,
    reason: Option<&str>,
) -> MutationOutcome {
    if !(200..300).contains(&status) {
        return MutationOutcome::Retryable;
    }
    if success == Some(true) {
        return MutationOutcome::Applied;
    }
    match reason {
        Some("expected_revision_mismatch" | "stale_revision" | "not_found") => {
            MutationOutcome::Obsolete
        }
        _ => MutationOutcome::Retryable,
    }
}

pub(crate) fn classify_worker_response(status: u16, body: &serde_json::Value) -> MutationOutcome {
    let Some(results) = body.get("results").and_then(serde_json::Value::as_array) else {
        return MutationOutcome::Retryable;
    };
    let mut all_applied = !results.is_empty();
    for result in results {
        match classify_worker_mutation(
            status,
            result.get("success").and_then(serde_json::Value::as_bool),
            result.get("reason_code").and_then(serde_json::Value::as_str),
        ) {
            MutationOutcome::Obsolete => return MutationOutcome::Obsolete,
            MutationOutcome::Applied => {}
            MutationOutcome::Retryable => all_applied = false,
        }
    }
    if all_applied {
        MutationOutcome::Applied
    } else {
        MutationOutcome::Retryable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_worker_cas_is_obsolete_without_rebasing() {
        for reason in ["expected_revision_mismatch", "stale_revision", "not_found"] {
            assert_eq!(
                classify_worker_mutation(200, Some(false), Some(reason)),
                MutationOutcome::Obsolete
            );
        }
    }

    #[test]
    fn network_http_and_unclassified_worker_failures_remain_retryable() {
        assert_eq!(
            classify_worker_mutation(503, Some(false), Some("stale_revision")),
            MutationOutcome::Retryable
        );
        assert_eq!(
            classify_worker_mutation(200, None, None),
            MutationOutcome::Retryable
        );
        assert_eq!(
            classify_worker_mutation(200, Some(false), Some("busy")),
            MutationOutcome::Retryable
        );
        assert_eq!(
            classify_worker_mutation(200, Some(true), None),
            MutationOutcome::Applied
        );
    }
    #[test]
    fn actual_http_all_rejected_revisions_are_terminal_obsolete() {
        for reason in ["expected_revision_mismatch", "stale_revision", "not_found"] {
            let body = serde_json::json!({
                "success": false,
                "results": [{
                    "success": false, "reason_code": reason,
                    "current_revision": 5, "queue_name": "prefill_waiting"
                }]
            });
            assert_eq!(classify_worker_response(200, &body), MutationOutcome::Obsolete);
        }
    }

    #[test]
    fn actual_http_any_applied_cannot_hide_an_obsolete_tensor_parallel_rank() {
        let body = serde_json::json!({
            "success": true,
            "results": [
                {"success": true, "reason_code": "success", "applied_revision": 6},
                {"success": false, "reason_code": "expected_revision_mismatch", "current_revision": 5}
            ]
        });
        assert_eq!(classify_worker_response(200, &body), MutationOutcome::Obsolete);
    }

    #[test]
    fn actual_http_applied_requires_every_tensor_parallel_rank() {
        let body = serde_json::json!({
            "success": true,
            "results": [
                {"success": true, "reason_code": "success", "applied_revision": 6},
                {"success": false, "reason_code": "temporarily_unavailable"}
            ]
        });
        assert_eq!(classify_worker_response(200, &body), MutationOutcome::Retryable);
    }

    #[test]
    fn actual_http_all_applied_is_applied_and_transient_only_retries() {
        let applied = serde_json::json!({
            "success": true,
            "results": [
                {"success": true, "reason_code": "success", "applied_revision": 6},
                {"success": true, "reason_code": "success", "applied_revision": 6}
            ]
        });
        let transient = serde_json::json!({
            "success": false,
            "results": [{"success": false, "reason_code": "temporarily_unavailable"}]
        });
        assert_eq!(classify_worker_response(200, &applied), MutationOutcome::Applied);
        assert_eq!(classify_worker_response(200, &transient), MutationOutcome::Retryable);
        assert_eq!(classify_worker_response(503, &applied), MutationOutcome::Retryable);
    }
}
