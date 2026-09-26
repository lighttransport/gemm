#!/usr/bin/env python3
"""Focused, hardware-independent tests for q38d_capacity.py."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("q38d_capacity.py")
SPEC = importlib.util.spec_from_file_location("q38d_capacity", MODULE_PATH)
assert SPEC and SPEC.loader
capacity = importlib.util.module_from_spec(SPEC)
sys.modules["q38d_capacity"] = capacity
SPEC.loader.exec_module(capacity)


class Q38DCapacityTests(unittest.TestCase):
    def test_source_derived_kv_slopes(self) -> None:
        # TP4 shards one of four KV heads per rank; TP12 replicates all four
        # heads with the replicated-mixer branch.
        self.assertEqual(capacity.local_kv_heads(4), 1)
        self.assertEqual(capacity.local_kv_heads(12), 4)
        self.assertEqual(capacity.kv_bytes_per_token(4, "int8"), 8 * 1024)
        self.assertEqual(capacity.kv_bytes_per_token(4, "fp32"), 32 * 1024)
        self.assertEqual(capacity.kv_bytes_per_token(12, "int8"), 32 * 1024)
        self.assertEqual(capacity.kv_bytes_per_token(12, "fp32"), 128 * 1024)

    def test_ssm_state_is_replicated_only_for_tp12(self) -> None:
        tp4 = capacity.ssm_memory_breakdown(4)
        tp12 = capacity.ssm_memory_breakdown(12)
        self.assertEqual(tp4["ssm_layers"], 48)
        self.assertEqual(tp4["local_ssm_heads"], 12)
        self.assertEqual(tp12["local_ssm_heads"], 48)
        # TP12 has all four times the per-head state, plus the same global
        # conv_state allocation.
        self.assertGreater(tp12["total_bytes"], tp4["total_bytes"] * 3)
        self.assertLess(tp12["total_bytes"], tp4["total_bytes"] * 5)

    def test_balanced_decode_groups(self) -> None:
        self.assertEqual(capacity.balanced_group_sizes(8), [3, 3, 2])
        self.assertEqual(capacity.balanced_group_sizes(16), [6, 5, 5])
        self.assertEqual(capacity.balanced_group_sizes(32), [11, 11, 10])

    def test_missing_tp_model_budget_fails_closed(self) -> None:
        plan = capacity.make_plan()
        for row in plan["rows"]:
            self.assertEqual(row["decode"]["status"], "UNKNOWN")
            self.assertEqual(row["prefill"]["status"], "UNKNOWN")
            self.assertIsNone(row["decode"]["fits"])
            self.assertIsNone(row["prefill"]["fits"])

    def test_capacity_is_separate_from_current_engine_gates(self) -> None:
        config = capacity.PlannerConfig(tp4_model_gib=5.0, tp12_model_gib=5.0)
        plan = capacity.make_plan(config)
        rows = {row["name"]: row for row in plan["rows"]}

        # All decode rows fit this supplied hypothetical TP4 budget, but the
        # current engine still blocks INT8 KV.
        for row in rows.values():
            self.assertTrue(row["decode"]["fits"])
            self.assertEqual(row["decode"]["status"], "BLOCKED")

        # 1M TP12 KV alone is close to the complete 32 GiB HBM budget and
        # rejects before the handoff capability gate is considered.
        self.assertFalse(rows["1m_x8"]["prefill"]["fits"])
        self.assertEqual(rows["1m_x8"]["prefill"]["status"], "REJECT")
        self.assertTrue(rows["256k_x32"]["prefill"]["fits"])
        self.assertEqual(rows["256k_x32"]["prefill"]["status"], "BLOCKED")

    def test_hypothetical_future_gates_can_admit_fitting_rows(self) -> None:
        config = capacity.PlannerConfig(
            tp4_model_gib=5.0,
            tp12_model_gib=5.0,
            engine_int8_kv=True,
            engine_tp12_handoff=True,
        )
        rows = {row["name"]: row for row in capacity.make_plan(config)["rows"]}
        for name in ("512k_x16", "256k_x32"):
            self.assertEqual(rows[name]["decode"]["status"], "ADMITTED")
            self.assertEqual(rows[name]["prefill"]["status"], "ADMITTED")
        self.assertEqual(rows["1m_x8"]["prefill"]["status"], "REJECT")

    def test_output_reserve_is_charged_per_context(self) -> None:
        no_reserve = capacity.PlannerConfig(
            tp4_model_gib=1.0,
            tp12_model_gib=1.0,
            output_reserve_tokens=0,
            engine_int8_kv=True,
            engine_tp12_handoff=True,
        )
        reserve = capacity.replace(no_reserve, output_reserve_tokens=8192)
        a = capacity.make_plan(no_reserve)["rows"][0]["decode"]["kv_bytes_per_node"]
        b = capacity.make_plan(reserve)["rows"][0]["decode"]["kv_bytes_per_node"]
        self.assertEqual(b - a, 3 * 8192 * capacity.kv_bytes_per_token(4, "int8"))


if __name__ == "__main__":
    unittest.main()
