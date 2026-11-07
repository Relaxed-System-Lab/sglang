import importlib.util
import unittest
from pathlib import Path


def load():
    path = Path(__file__).with_name("aggregate_final.py")
    spec = importlib.util.spec_from_file_location("final_aggregate_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FinalAggregationTests(unittest.TestCase):
    def test_nearest_rank_keeps_sample_size_and_rank(self):
        m = load()
        stats = m.distribution(list(range(1, 101)))
        self.assertEqual(stats["p99"], 99)
        self.assertEqual(stats["p99_rank"], 99)
        self.assertEqual(m.distribution([1, 2, 3, 4])["p95"], 4)
        self.assertIsNone(m.distribution([])["p99"])

    def test_failed_workflow_keeps_diagnostic_latency_but_no_ratio(self):
        m = load()
        wf = {"workflow_id": "w", "calls": [{"call_id": "a", "parents": [], "input_len": 1, "output_len": 2}]}
        rows = [{"workflow_id": "w", "call_id": "a", "workflow_arrival_ms": 100,
                 "response_completed_at_ms": 150, "request_started_at_ms": 101,
                 "error": "HTTP503", "response": None}]
        out = m.workflow_record(wf, rows, {"latency_ms": 25}, "COMPLETE_WITH_FAILURES", "model")
        self.assertEqual(out["observed_latency_ms"], 50)
        self.assertEqual(out["status"], "failed")
        self.assertIsNone(out["slo_ratio"])
        self.assertIsNone(out["successful_latency_ms"])

    def test_failure_inclusive_alpha_and_unobserved_bounds(self):
        m = load()
        rows = [{"status": "successful", "slo_ratio": 1.6},
                *[{"status": "failed", "slo_ratio": None}] * 3]
        alpha = m.alpha_metrics(rows, formal_started=True)
        self.assertEqual([r["violation_rate_lower"] for r in alpha], [1.0, 0.75, 0.75])
        self.assertIsNone(m.alpha_metrics(rows, formal_started=False)[0]["violation_rate_lower"])

    def test_warmup_telemetry_is_excluded_when_formal_never_started(self):
        m = load()
        rows = [{"endpoint": "/plan_pending_admissions", "admission_count": 0,
                 "result_count": 0, "event_count": 1, "wall_ms": 0.1, "cpu_ms": 0.05}] * 10
        out = m.planner_metrics(rows, formal_started=False)
        self.assertEqual(out["formal_invocation_count"], 0)
        self.assertEqual(out["excluded_nonformal_records"], 10)
        self.assertIsNone(out["wall_ms"]["mean"])

    def test_event_actor_rows_do_not_create_canonical_planner_work(self):
        m = load()
        rows = [{"endpoint": "/plan_pending_admissions", "admission_count": 0,
                 "result_count": 0, "event_count": 3, "wall_ms": 0.1, "cpu_ms": 0.05}]
        out = m.planner_metrics(rows, formal_started=True)
        self.assertEqual(out["formal_invocation_count"], 1)
        self.assertEqual(out["nonempty_planning_invocations"], 0)
        self.assertEqual(out["event_count"], 3)

    def test_configured_hexgent_revision_zero_preserves_zero_coverage(self):
        m = load()
        result = m.planner_coverage([{"router_mode": "hexgent", "schedule_revision": 0, "error": None}] * 16)
        self.assertEqual(result["planner_revision_coverage"], 0.0)
        self.assertEqual(result["fallback_revision_calls"], 16)
        self.assertFalse(result["planner_effect_observed"])

    def test_warmup_block_and_budget_are_distinct_case_states(self):
        m = load()
        self.assertEqual(m.case_state(False, [], {}, {}), "NOT_RUN_budget")
        self.assertEqual(m.case_state(True, [], {"successful": False}, {}),
                         "NOT_STARTED_warmup_failure")

    def test_missing_call_is_retained_in_workflow_ledger(self):
        m = load()
        wf = {"workflow_id": "w", "calls": [{"call_id": "a"}, {"call_id": "b"}]}
        out = m.workflow_record(wf, [], {"latency_ms": 25}, "NOT_STARTED_warmup_failure", "model")
        self.assertEqual(out["missing_call_ids"], ["a", "b"])
        self.assertIsNone(out["observed_latency_ms"])
        self.assertIsNone(out["slo_ratio"])
        self.assertEqual(out["status"], "NOT_STARTED_warmup_failure")


if __name__ == "__main__":
    unittest.main()
