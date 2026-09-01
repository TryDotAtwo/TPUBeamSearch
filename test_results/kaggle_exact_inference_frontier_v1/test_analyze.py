from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

try:
    from analyze import summarize_profile_events, validate_frontier_report
except ImportError:
    summarize_profile_events = None
    validate_frontier_report = None


def nested_while_trace() -> list[dict]:
    events = [
        {"ph": "M", "pid": 3, "name": "process_name", "args": {"name": "/device:TPU:0"}},
        {"ph": "M", "pid": 3, "tid": 2, "name": "thread_name", "args": {"name": "XLA Modules"}},
        {"ph": "M", "pid": 3, "tid": 3, "name": "thread_name", "args": {"name": "XLA Ops"}},
    ]
    for index in range(3):
        start = index * 2_000
        events.extend([
            {
                "ph": "X", "pid": 3, "tid": 2, "name": "jit_call",
                "ts": start, "dur": 1_000,
                "args": {"device_duration_ps": "1000000000", "run_id": str(index)},
            },
            {
                "ph": "X", "pid": 3, "tid": 3, "name": "while.1",
                "ts": start + 50, "dur": 900,
            },
            {
                "ph": "X", "pid": 3, "tid": 3,
                "name": "stream1_dense_linear.20",
                "ts": start + 100, "dur": 800,
            },
        ])
    return events


class ProfileEventTests(unittest.TestCase):
    def test_nested_while_container_is_excluded_but_leaf_is_counted(self):
        self.assertIsNotNone(summarize_profile_events, "analysis API is not implemented")
        result = summarize_profile_events(nested_while_trace())
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["excluded_loop_containers"]["event_count"], 3)
        self.assertEqual(result["device_op_count"], 3)
        self.assertEqual(result["categories"]["pallas_dense"]["event_count"], 3)

    def test_non_while_overlap_remains_rejected(self):
        self.assertIsNotNone(summarize_profile_events, "analysis API is not implemented")
        events = copy.deepcopy(nested_while_trace())
        for event in events:
            if event.get("name") == "while.1":
                event["name"] = "fusion"
        result = summarize_profile_events(events)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("overlap", result["error"])

    def test_multidevice_trace_is_scoped_to_tpu_zero(self):
        self.assertIsNotNone(summarize_profile_events, "analysis API is not implemented")
        events = nested_while_trace()
        events.extend([
            {"ph": "M", "pid": 9, "name": "process_name", "args": {"name": "/device:TPU:1"}},
            {"ph": "M", "pid": 9, "tid": 2, "name": "thread_name", "args": {"name": "XLA Modules"}},
            {"ph": "M", "pid": 9, "tid": 3, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {
                "ph": "X", "pid": 9, "tid": 2, "name": "jit_call",
                "ts": 0, "dur": 1_000,
                "args": {"device_duration_ps": "1000000000", "run_id": "other-device"},
            },
            {
                "ph": "X", "pid": 9, "tid": 3,
                "name": "stream1_dense_linear.20", "ts": 100, "dur": 800,
            },
        ])
        result = summarize_profile_events(events)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["device_op_count"], 3)

    def test_two_dispatch_composed_runner_is_collapsed_to_three_forwards(self):
        self.assertIsNotNone(summarize_profile_events, "analysis API is not implemented")
        events = [
            {"ph": "M", "pid": 3, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 3, "tid": 2, "name": "thread_name", "args": {"name": "XLA Modules"}},
            {"ph": "M", "pid": 3, "tid": 3, "name": "thread_name", "args": {"name": "XLA Ops"}},
        ]
        for index in range(3):
            start = index * 2_000
            events.extend([
                {
                    "ph": "X", "pid": 3, "tid": 2, "name": "prefix",
                    "ts": start, "dur": 800,
                    "args": {"device_duration_ps": "800000000", "run_id": f"p{index}"},
                },
                {
                    "ph": "X", "pid": 3, "tid": 2, "name": "head",
                    "ts": start + 810, "dur": 100,
                    "args": {"device_duration_ps": "100000000", "run_id": f"h{index}"},
                },
                {
                    "ph": "X", "pid": 3, "tid": 3,
                    "name": "stream1_dense_prefix", "ts": start + 100, "dur": 600,
                },
                {
                    "ph": "X", "pid": 3, "tid": 3,
                    "name": "stream1_dense_head", "ts": start + 820, "dur": 60,
                },
            ])
        result = summarize_profile_events(events)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["module_count"], 3)
        self.assertEqual(result["compiled_module_count"], 6)
        self.assertEqual(result["dispatches_per_forward"], 2)
        self.assertEqual(result["module_samples_ms"], [0.91, 0.91, 0.91])
        self.assertAlmostEqual(result["inter_dispatch_gap_ms_per_forward"], 0.01)


class FrontierReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parent / "exact_inference_frontier" / "stream1_exact_inference_frontier.json"
        cls.report = json.loads(path.read_text(encoding="utf-8"))

    def test_completed_report_enforces_four_exact_faster_winner_comparisons(self):
        self.assertIsNotNone(validate_frontier_report, "report validator is not implemented")
        result = validate_frontier_report(self.report)
        self.assertEqual(
            result["selected_id"],
            "exact_split_bm4096_pallas_head_bm256_bk1024_bn128_late",
        )
        self.assertEqual(len(result["winner_comparisons"]), 4)
        self.assertEqual(result["compile_rejection_ids"], ["prefix_bm16384", "prefix_bm8192"])

    def test_winner_mismatch_is_rejected(self):
        self.assertIsNotNone(validate_frontier_report, "report validator is not implemented")
        report = copy.deepcopy(self.report)
        winner = next(
            row
            for row in report["full_measurements"]
            if row["section"] == "confirmation"
            and row["corpus"] == "legal_scrambles"
            and row["id"] == report["confirmation_decision"]["selected_id"]
        )
        winner["mismatch_witnesses"]["mismatch_count"] = 1
        with self.assertRaisesRegex(ValueError, "winner is not elementwise exact"):
            validate_frontier_report(report)


if __name__ == "__main__":
    unittest.main()
