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
        # TP4 shards one of four KV heads per rank.  INT6 packs each row into
        # 192 bytes and keeps one FP32 scale for K and one for V.
        self.assertEqual(capacity.local_kv_heads(4), 1)
        self.assertEqual(capacity.local_kv_heads(12), 4)
        self.assertEqual(capacity.KV_I6_ROW_BYTES, 192)
        self.assertEqual(capacity.kv_bytes_per_token(4, "int6"), 16 * 1 * 2 * (192 + 4))
        self.assertEqual(capacity.kv_bytes_per_token(4, "int8"), 16 * 1 * 2 * (256 + 4))
        self.assertEqual(capacity.kv_bytes_per_token(4, "fp32"), 32 * 1024)
        self.assertEqual(capacity.kv_bytes_per_token(12, "int8"), 16 * 4 * 2 * (256 + 4))
        self.assertEqual(capacity.kv_bytes_per_token(12, "fp32"), 128 * 1024)

    def test_pp12_partition_has_two_attention_layers_at_worst(self) -> None:
        partition = capacity.prefill_stage_partition()
        self.assertEqual(partition["attention_layers_per_stage"], [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2])
        self.assertEqual(partition["worst_attention_layers"], 2)
        self.assertEqual(partition["worst_ssm_layers"], 5)
        self.assertEqual(partition["source"], "q38p_prefill.inc pf_stage_bounds + q38p_partition.h")
        # PP12 owns all four KV heads on a stage, so its worst-stage INT6 KV
        # slope is two attention layers times four heads.
        self.assertEqual(
            capacity.kv_bytes_per_token(12, "int6", attention_layers=2, kv_heads=4),
            2 * 4 * 2 * (192 + 4),
        )

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
        plan = capacity.make_plan(capacity.replace(capacity.PlannerConfig(), tp4_model_gib=None))
        for row in plan["rows"]:
            self.assertEqual(row["decode"]["status"], "UNKNOWN")
            self.assertEqual(row["prefill"]["status"], "ADMITTED")
            self.assertIsNone(row["decode"]["fits"])
            self.assertTrue(row["prefill"]["fits"])

    def test_capacity_is_separate_from_runtime_proof(self) -> None:
        config = capacity.PlannerConfig(tp4_model_gib=1.0, tp12_model_gib=capacity.FULL_MODEL_ALLOCATION_GIB)
        plan = capacity.make_plan(config)
        rows = {row["name"]: row for row in plan["rows"]}

        # All decode rows fit this supplied hypothetical TP4 budget and
        # q38d --kv-i6 is an executable regular-decode gate.
        for row in rows.values():
            self.assertTrue(row["decode"]["fits"])
            self.assertEqual(row["decode"]["status"], "ADMITTED")

        # PP12 charges only its worst two attention layers. Admission is a
        # memory and capability estimate, not proof of full-context quality.
        self.assertTrue(rows["1m_x8"]["prefill"]["fits"])
        self.assertEqual(rows["1m_x8"]["prefill"]["status"], "ADMITTED")
        self.assertFalse(rows["1m_x8"]["prefill"]["current_fp32_fits"])
        self.assertFalse(rows["1m_x8"]["prefill"]["current_alloc_fp32_fits"])
        self.assertFalse(rows["256k_x32"]["prefill"]["current_alloc_fp32_fits"])
        self.assertTrue(rows["256k_x32"]["prefill"]["fits"])
        self.assertEqual(rows["256k_x32"]["prefill"]["status"], "ADMITTED")

    def test_i6_decode_gate_is_independent_of_pp12_handoff_gate(self) -> None:
        config = capacity.PlannerConfig(
            tp4_model_gib=capacity.TP4_POST_PRUNE_RSS_GIB,
            tp12_model_gib=capacity.FULL_MODEL_ALLOCATION_GIB,
            kv_dtype="int6",
            engine_int6_kv=True,
            engine_prefill_compressed_kv=False,
            engine_tp12_handoff=True,
        )
        row = capacity.make_plan(config)["rows"][0]
        self.assertEqual(row["decode"]["status"], "ADMITTED")
        self.assertEqual(row["prefill"]["status"], "BLOCKED")
        self.assertIn("compressed K/V", row["prefill"]["gates"][0])

        disabled = capacity.replace(config, engine_int6_kv=False)
        disabled_row = capacity.make_plan(disabled)["rows"][0]
        self.assertEqual(disabled_row["decode"]["status"], "BLOCKED")
        self.assertIn("--kv-i6", disabled_row["decode"]["gates"][0])

    def test_i6_prefill_gate_does_not_admit_int8_handoff(self) -> None:
        row = capacity.make_plan(
            capacity.PlannerConfig(kv_dtype="int8"),
            [capacity.Workload("small", 1024, 1)],
        )["rows"][0]
        self.assertEqual(row["prefill"]["status"], "BLOCKED")
        self.assertEqual(row["decode"]["status"], "ADMITTED")

    def test_current_engine_gates_can_admit_fitting_rows(self) -> None:
        config = capacity.PlannerConfig(
            tp4_model_gib=1.0,
            tp12_model_gib=capacity.FULL_MODEL_ALLOCATION_GIB,
            engine_int6_kv=True,
            engine_prefill_compressed_kv=True,
            engine_tp12_handoff=True,
        )
        rows = {row["name"]: row for row in capacity.make_plan(config)["rows"]}
        for name in ("512k_x16", "256k_x32"):
            self.assertEqual(rows[name]["decode"]["status"], "ADMITTED")
            self.assertEqual(rows[name]["prefill"]["status"], "ADMITTED")
        self.assertEqual(rows["1m_x8"]["prefill"]["status"], "ADMITTED")

    def test_requested_shapes_are_binary_context_lengths(self) -> None:
        self.assertEqual(
            [(w.context_tokens, w.contexts) for w in capacity.WORKLOADS],
            [(1_048_576, 8), (524_288, 16), (262_144, 32)],
        )

    def test_pruned_mapping_is_reported_without_being_resident_budget(self) -> None:
        plan = capacity.make_plan()
        ref = plan["model_reference"]
        self.assertEqual(ref["tp4_pruned_source_mapping_released_gib"], 13.891)
        self.assertEqual(ref["tp4_rss_after_prune_gib"], 6.02)
        self.assertEqual(plan["rows"][0]["model"]["tp4_resident_gib"], 6.02)

    def test_output_reserve_is_charged_per_context(self) -> None:
        no_reserve = capacity.PlannerConfig(
            tp4_model_gib=1.0,
            tp12_model_gib=1.0,
            output_reserve_tokens=0,
            engine_int6_kv=True,
            engine_prefill_compressed_kv=True,
            engine_tp12_handoff=True,
        )
        reserve = capacity.replace(no_reserve, output_reserve_tokens=8192)
        a = capacity.make_plan(no_reserve)["rows"][0]["decode"]["kv_bytes_per_node"]
        b = capacity.make_plan(reserve)["rows"][0]["decode"]["kv_bytes_per_node"]
        self.assertEqual(b - a, 3 * 8192 * capacity.kv_bytes_per_token(4, "int6"))


if __name__ == "__main__":
    unittest.main()
