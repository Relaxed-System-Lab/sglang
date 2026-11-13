#[test]
fn test_hexgent_planner_request_preserves_original_chat_body_and_old_empty_fixture() {
    let body = json!({
        "model": "served-model",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "observed"}
        ],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "chat_template_kwargs": {"enable_thinking": false},
        "max_tokens": 8
    });
    let payload = HexgentPlannerRequest {
        route: "/v1/chat/completions",
        model: Some("served-model".to_string()),
        request_text: "length fallback".to_string(),
        request_body: Some(body.clone()),
        max_tokens: Some(8),
        hexgent: json!({"workflow_id": "wf", "call_id": "c"}),
        prefill_workers: vec![],
        decode_workers: vec![],
        state: HexgentPlannerState { current_time_ms: 0, worker_snapshots: None },
    };
    let serialized = serde_json::to_value(&payload).unwrap();
    assert_eq!(serialized["request_body"], body);
    assert_eq!(serialized["request_text"], "length fallback");
    let mut old_fixture = payload;
    old_fixture.request_body = None;
    assert!(serde_json::to_value(old_fixture).unwrap().get("request_body").is_none());
}
