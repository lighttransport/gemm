#!/usr/bin/env python3
import importlib.util
import unittest


def load_stage():
    spec = importlib.util.spec_from_file_location(
        "k3_gguf_expert_tp_stage", "k3_gguf_expert_tp_stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ExpertTPStageTest(unittest.TestCase):
    def test_row_and_column_ownership(self):
        stage = load_stage()
        records = []
        for suffix, cols, rows, typ, row_bytes in (
                ("ffn_up_exps.weight", 512, 3072, "IQ1_S", 100),
                ("ffn_down_exps.weight", 3072, 3584, "IQ1_S", 600),
                ("ffn_gate_exps.weight", 512, 3072, "IQ1_S", 100)):
            records.append({
                "name": "blk.1." + suffix, "dims": [cols, rows, 896],
                "type": typ, "row_bytes": row_bytes,
                "data_start": 1000, "source": "/tmp/model.gguf",
            })
        records.append({
            "name": "blk.1.ffn_gate_inp.weight", "dims": [7168, 896],
            "type": "IQ1_S", "row_bytes": 1400,
            "data_start": 1000, "source": "/tmp/model.gguf",
        })
        records.append({
            "name": "blk.1.ffn_routed_down.weight", "dims": [7168, 3584],
            "type": "Q8_0", "row_bytes": 7616,
            "data_start": 1000, "source": "/tmp/model.gguf",
        })
        records.append({
            "name": "blk.1.ffn_routed_norm.weight", "dims": [3584],
            "type": "F32", "row_bytes": 4,
            "data_start": 1000, "source": "/tmp/model.gguf",
        })
        records.append({
            "name": "blk.1.ffn_routed_up.weight", "dims": [3584, 7168],
            "type": "Q8_0", "row_bytes": 7168,
            "data_start": 1000, "source": "/tmp/model.gguf",
        })
        plan, _ = stage.make_plan(records, 1, 3, 12, [0, 15, 895])
        w1 = [x for x in plan if x["role"] == "w1"]
        w2 = [x for x in plan if x["role"] == "w2"]
        router = [x for x in plan if x["role"] == "router"]
        down = [x for x in plan if x["role"] == "routed_down"]
        self.assertEqual(len(router), 1)
        self.assertEqual(router[0]["row_count"], 75)
        self.assertEqual(len(down), 1)
        self.assertEqual(down[0]["row_count"], 299)
        self.assertEqual(len(w1), 3)
        self.assertTrue(all(x["kind"] == "rows" and x["row_count"] == 256
                            for x in w1))
        self.assertEqual(len(w2), 3)
        self.assertTrue(all(x["kind"] == "cols" and x["col_count"] == 256 and
                            x["row_count"] == 3584
                            for x in w2))
        self.assertTrue(all(x["col_start"] == 3 * 256 for x in w2))

    def test_rejects_unaligned_columns(self):
        stage = load_stage()
        records = []
        for suffix, cols, rows in (
                ("ffn_up_exps.weight", 512, 3072),
                ("ffn_down_exps.weight", 3072, 3584),
                ("ffn_gate_exps.weight", 512, 3072)):
            records.append({
                "name": "blk.1." + suffix, "dims": [cols, rows, 896],
                "type": "IQ1_S", "row_bytes": cols // 256 * 50,
                "data_start": 0, "source": "/tmp/model.gguf",
            })
        records.append({
            "name": "blk.1.ffn_gate_inp.weight", "dims": [7168, 896],
            "type": "IQ1_S", "row_bytes": 1400,
            "data_start": 0, "source": "/tmp/model.gguf",
        })
        records.append({
            "name": "blk.1.ffn_routed_down.weight", "dims": [7168, 3584],
            "type": "Q8_0", "row_bytes": 7616,
            "data_start": 0, "source": "/tmp/model.gguf",
        })
        records.extend([
            {"name": "blk.1.ffn_routed_norm.weight", "dims": [3584],
             "type": "F32", "row_bytes": 4, "data_start": 0,
             "source": "/tmp/model.gguf"},
            {"name": "blk.1.ffn_routed_up.weight", "dims": [3584, 7168],
             "type": "Q8_0", "row_bytes": 7168, "data_start": 0,
             "source": "/tmp/model.gguf"},
        ])
        with self.assertRaises(ValueError):
            stage.make_plan(records, 1, 0, 7, [0])


if __name__ == "__main__":
    unittest.main()
