#!/usr/bin/env python3
"""Check all TP rank/row mappings without reading model payloads."""
import unittest
from pathlib import Path
import stage_tp


def tensor(name, shape, dtype='F8_E4M3', kind='dense'):
    unit = 4 if dtype == 'F32' else 2 if dtype == 'BF16' else 1
    n = 1
    for dim in shape:
        n *= dim
    return dict(name=name, shape=shape, source_shape=shape, dtype=dtype,
                source='checkpoint.safetensors', source_offset=1024,
                bytes=n*unit, first_row=0, kind=kind)


class LayoutTest(unittest.TestCase):
    def setUp(self):
        self.old = stage_tp.inventory
        self.dense = [tensor('head.weight', [129280, 5120], 'BF16'),
                      tensor('embed.weight', [129280, 5120], 'BF16'),
                      tensor('norm.weight', [5120], 'BF16')]
        for layer in range(40):
            self.dense.append(tensor('layers.%d.attn.attn_sink' % layer, [64], 'F32'))
            self.dense.append(tensor('layers.%d.hc_attn_fn' % layer, [24, 20480], 'F32'))
            for name, rows, cols in [('attn.wq_b', 32768, 1280), ('attn.wo_a', 8192, 4096),
                                     ('attn.wo_b', 5120, 8192), ('ffn.shared_experts.w1', 2304, 5120),
                                     ('ffn.shared_experts.w3', 2304, 5120), ('ffn.shared_experts.w2', 5120, 2304)]:
                base = 'layers.%d.%s' % (layer, name)
                self.dense += [tensor(base+'.weight', [rows, cols]),
                               tensor(base+'.scale', [rows//32, cols//32], 'F8_E8M0')]
        stage_tp.inventory = lambda model, rank: iter(self.dense if rank == 0 else [])

    def tearDown(self):
        stage_tp.inventory = self.old

    def test_all_rank_ranges(self):
        for tp in (2, 4):
            layouts = [stage_tp.layout(Path('.'), rank, tp) for rank in range(12)]
            copies = {}
            for rank, items in enumerate(layouts):
                self.assertEqual(len(items), len({x['name'] for x in items}))
                for item in items:
                    copies.setdefault(item['name'], []).append((rank, item))
            for original in self.dense:
                name = original['name']; shards = copies[name]
                if 'global_shape' in shards[0][1]:
                    self.assertEqual(len(shards), 12 if name == 'head.weight' else tp)
                    expected = 0
                    for rank, item in shards:
                        self.assertEqual(item['first_row'], expected)
                        self.assertEqual(item['source_offset'], 1024+expected*original['bytes']//original['shape'][0])
                        if name != 'head.weight':
                            self.assertEqual(rank//tp, stage_tp.dense_owner(name)//tp)
                        expected += item['shape'][0]
                    self.assertEqual(expected, original['shape'][0])
                    self.assertEqual(sum(x['bytes'] for _, x in shards), original['bytes'])
                else:
                    self.assertEqual(len(shards), tp if name.endswith('attn_sink') else 1)
            self.assertEqual(sum(x['bytes'] for items in layouts for x in items),
                             sum(x['bytes'] for x in self.dense)+(tp-1)*40*64*4)

    def test_invalid_ranges(self):
        item = tensor('weight', [128, 32])
        for first, rows in [(-1, 1), (0, 0), (127, 2), (128, 1)]:
            with self.assertRaises(ValueError):
                stage_tp.row_shard(item, first, rows)
        with self.assertRaises(ValueError):
            stage_tp.layout(Path('.'), 12, 2)


if __name__ == '__main__':
    unittest.main()
