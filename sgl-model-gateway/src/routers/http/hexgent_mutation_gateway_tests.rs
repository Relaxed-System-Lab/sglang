#[test]
fn test_obsolete_worker_action_refresh_keeps_only_pending_admissions_owned() {
    let mut queue = HexgentGlobalAdmissionQueue::new(1);
    let running = HexgentAdmissionKey::new("wf", "running");
    let pending = HexgentAdmissionKey::new("wf", "pending");
    assert_eq!(
        queue
            .enqueue(
                running.clone(),
                test_hexgent_decision(1, "http://prefill", 0)
            )
            .len(),
        1
    );
    assert!(queue
        .enqueue(
            pending.clone(),
            test_hexgent_decision(1, "http://prefill", 0)
        )
        .is_empty());
    let event = PDRouter::hexgent_pending_snapshot_event_payload(
        &queue,
        "http://prefill",
        json!({"snapshot": {"timestamp": 7}}),
        7,
    )
    .unwrap();
    assert_eq!(event["event_type"], "worker_snapshot_updated");
    assert_eq!(
        event["payload"]["pending_admissions"],
        json!([{
            "workflow_id": "wf", "call_id": "pending"
        }])
    );
    assert!(queue.is_pending(&pending));
    assert!(!queue.is_pending(&running));
    assert_eq!(event["payload"]["snapshot"]["snapshot"]["timestamp"], 7);
    queue.cancel(&pending);
    assert!(PDRouter::hexgent_pending_snapshot_event_payload(
        &queue,
        "http://prefill",
        json!({}),
        8,
    )
    .is_none());
}
