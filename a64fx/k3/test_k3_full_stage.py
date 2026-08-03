import os
import tempfile
import unittest

import k3_full_stage as stage


class ExpertTpPlanTest(unittest.TestCase):
    def test_group_split_keeps_q8_rows_aligned(self):
        self.assertEqual(stage.split_group_dim(7168, 0, 96, 8), (0, 80))
        self.assertEqual(stage.split_group_dim(7168, 95, 96, 8), (7096, 72))
        self.assertEqual(stage.split_group_dim(3584, 95, 96, 8), (3552, 32))

    def test_partition_is_aligned_and_slices_packed_w2_by_columns(self):
        prefix = "language_model.model.layers.1."
        ep = prefix + "block_sparse_moe.experts.7."
        headers = [{
            ep + "w1.weight_packed": {
                "source": "s0", "source_offset": 1000, "nbytes": 3072 * 1536,
                "dtype": "U8", "shape": [3072, 1536]},
            ep + "w1.weight_scale": {
                "source": "s0", "source_offset": 2000, "nbytes": 3072 * 96,
                "dtype": "U8", "shape": [3072, 96]},
            ep + "w2.weight_packed": {
                "source": "s0", "source_offset": 3000, "nbytes": 3584 * 1536,
                "dtype": "U8", "shape": [3584, 1536]},
            ep + "w2.weight_scale": {
                "source": "s0", "source_offset": 4000, "nbytes": 3584 * 96,
                "dtype": "U8", "shape": [3584, 96]},
            ep + "w3.weight_packed": {
                "source": "s0", "source_offset": 5000, "nbytes": 3072 * 1536,
                "dtype": "U8", "shape": [3072, 1536]},
            ep + "w3.weight_scale": {
                "source": "s0", "source_offset": 6000, "nbytes": 3072 * 96,
                "dtype": "U8", "shape": [3072, 96]},
        }]
        records = stage.expert_tp_records(headers, prefix, rank=5, size=12, expert=7)
        self.assertEqual(len(records), 6)
        by_name = {r["name"]: r for r in records}
        w1 = by_name[ep + "w1.weight_packed"]
        self.assertEqual(w1["shape"], [256, 1536])
        self.assertEqual(w1["segments"], [(1000 + 1280 * 1536, 256 * 1536)])
        w2 = by_name[ep + "w2.weight_packed"]
        self.assertEqual(w2["shape"], [3584, 128])
        self.assertEqual(w2["nbytes"], 3584 * 128)
        self.assertEqual(w2["segments"], ("rows", 3000, 3584, 1536, 640, 128))

    def test_expert_tp_rejects_unaligned_partition(self):
        with self.assertRaises(ValueError):
            stage.expert_tp_records([], "", rank=0, size=7, expert=0)

    def test_rows_group_record_is_kernel_aligned(self):
        name = "block_sparse_moe.routed_expert_up_proj.weight"
        rec = {"name": name, "source": "s0", "source_offset": 1024,
               "nbytes": 7168 * 3584 * 2, "dtype": "BF16",
               "shape": [7168, 3584]}
        out = stage.copy_record(rec, "rows-group", rank=95, size=96)[0]
        self.assertEqual(out["shape"], [72, 3584])
        self.assertEqual(out["segments"], [(1024 + 7096 * 3584 * 2,
                                              72 * 3584 * 2)])

    def test_row_descriptor_copies_compacted_columns_in_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source")
            output = os.path.join(tmp, "output")
            rows, row_bytes, first, count = 700, 32, 7, 11
            payload = b"".join(bytes([r & 255]) * row_bytes for r in range(rows))
            with open(source, "wb") as f:
                f.write(payload)
            rec = {"source": source, "name": "w2", "segments":
                   ("rows", 0, rows, row_bytes, first, count)}
            with open(output, "wb", buffering=0) as f:
                stage.copy_ranges(f.fileno(), rec, 64)
            with open(output, "rb") as f:
                self.assertEqual(f.read(), b"".join(
                    bytes([r & 255]) * count for r in range(rows)))


if __name__ == "__main__":
    unittest.main()
