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

    def test_shared_tp12_uses_all_ranks_and_uneven_output_rows(self):
        layouts = [stage_tp.layout(Path('.'), rank, 4, shared_tp=12)
                   for rank in range(12)]
        for name in ('layers.0.ffn.shared_experts.w1.weight',
                     'layers.0.ffn.shared_experts.w2.weight'):
            shards = []
            for rank, items in enumerate(layouts):
                found = [x for x in items if x['name'] == name]
                self.assertEqual(len(found), 1)
                shards.append((rank, found[0]))
            self.assertEqual(len(shards), 12)
            expected = 0
            original = next(x for x in self.dense if x['name'] == name)
            for rank, item in shards:
                self.assertEqual(item['first_row'], expected)
                alignment = 32 if name.endswith('.weight') else 1
                end = (original['shape'][0] // alignment) * (rank + 1) // 12 * alignment
                self.assertEqual(item['shape'][0], end - expected)
                expected += item['shape'][0]
            self.assertEqual(expected, original['shape'][0])
            scale_name = name[:-7] + '.scale'
            scale_shards = []
            for rank, items in enumerate(layouts):
                found = [x for x in items if x['name'] == scale_name]
                self.assertEqual(len(found), 1)
                scale_shards.append((rank, found[0]))
            scale_original = next(x for x in self.dense if x['name'] == scale_name)
            self.assertEqual(sum(x['shape'][0] for _, x in scale_shards),
                             scale_original['shape'][0])
            for (_, weight), (_, scale) in zip(shards, scale_shards):
                self.assertEqual(scale['first_row'], weight['first_row'] // 32)
                self.assertEqual(scale['shape'][0], weight['shape'][0] // 32)

    def test_attention_tp12_uses_eight_head_and_twelve_output_shards(self):
        layouts = [stage_tp.layout(Path('.'), rank, 4, shared_tp=12,
                                   attention_tp12=True) for rank in range(12)]
        for name, degree, alignment in (
                ('layers.0.attn.wq_b.weight', 8, 1),
                ('layers.0.attn.wo_a.weight', 8, 1),
                ('layers.0.attn.wo_b.weight', 12, 32)):
            original = next(x for x in self.dense if x['name'] == name)
            shards = [(rank, next((x for x in items if x['name'] == name), None))
                      for rank, items in enumerate(layouts)]
            present = [(rank, item) for rank, item in shards if item is not None]
            self.assertEqual(len(present), degree)
            expected = 0
            for shard_index, (rank, item) in enumerate(present):
                self.assertEqual(rank, shard_index)
                self.assertEqual(item['first_row'], expected)
                blocks = original['shape'][0] // alignment
                end = blocks * (shard_index + 1) // degree * alignment
                self.assertEqual(item['shape'][0], end - expected)
                expected = end
            self.assertEqual(expected, original['shape'][0])
            scale_name = name[:-7] + '.scale'
            scale_shards = [(rank, next((x for x in items if x['name'] == scale_name), None))
                            for rank, items in enumerate(layouts)]
            scale_present = [(rank, item) for rank, item in scale_shards if item is not None]
            self.assertEqual(len(scale_present), degree)
            for (_, weight), (_, scale) in zip(present, scale_present):
                self.assertEqual(scale['first_row'], weight['first_row'] // 32)
                self.assertEqual(scale['shape'][0], weight['shape'][0] // 32)
        sinks = [sum(x['name'] == 'layers.0.attn.attn_sink' for x in items)
                 for items in layouts]
        self.assertEqual(sinks, [1] * 8 + [0] * 4)


if __name__ == '__main__':
    unittest.main()
