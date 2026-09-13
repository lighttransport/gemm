#!/usr/bin/env python3
"""Small synthetic header/partition test; never reads checkpoint payloads."""
import json
import struct
import tempfile
import unittest
from pathlib import Path
from stage_backbone import inventory


class PartitionTest(unittest.TestCase):
    def test_ownership_and_coverage(self):
        root = Path(__file__).resolve().parents[2] / 'tmp'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(root)) as temporary:
            model = Path(temporary)
            descriptors = {}
            offset = 0
            tensors = [('embed.weight', [2, 8], 32),
                       ('layers.1.engram.embed.weight', [25, 256], 25*256),
                       ('layers.1.engram.embed.scale', [25, 8], 25*8),
                       ('vision.foo', [8], 8), ('mtp.foo', [8], 8)]
            tensors += [('layers.0.ffn.experts.%d.w1.weight' % i, [2, 16], 32)
                        for i in range(24)]
            for name, shape, size in tensors:
                descriptors[name] = dict(dtype='I8', shape=shape,
                                         data_offsets=[offset, offset+size])
                offset += size
            header = json.dumps(descriptors).encode()
            with (model/'model-00001-of-00001.safetensors').open('wb') as f:
                f.write(struct.pack('<Q', len(header)))
                f.write(header)
                f.write(bytes(offset))
            all_items = [list(inventory(model,r)) for r in range(12)]
            self.assertEqual(sum(x['kind']=='dense' for items in all_items for x in items),1)
            for rank, items in enumerate(all_items):
                experts = [x for x in items if x['kind']=='expert']
                self.assertEqual(len(experts),2)
                for x in experts:
                    self.assertEqual(int(x['name'].split('.')[4])%12,rank)
                self.assertFalse(any(x['name'].startswith(('vision','mtp')) for x in items))
            for suffix, stride in [('weight',256),('scale',8)]:
                spans = [x for items in all_items for x in items
                         if x['name']=='layers.1.engram.embed.'+suffix]
                self.assertEqual(sum(x['bytes'] for x in spans),25*stride)
                rows = [row for x in spans for row in range(x['first_row'],x['first_row']+x['shape'][0])]
                self.assertEqual(rows,list(range(25)))
                start=descriptors['layers.1.engram.embed.'+suffix]['data_offsets'][0]
                for x in spans:
                    self.assertEqual(x['source_offset'],8+len(header)+start+x['first_row']*stride)


if __name__ == '__main__':
    unittest.main()
