"""Lossless checkpoint layout checks, including reversed Q/K source order."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from .export_flux2_weights import export


class ExportTests(unittest.TestCase):
    def test_qkv_and_final_scale_shift(self):
        root = Path(__file__).resolve().parents[2] / 'tmp'
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as work:
            source, output = Path(work)/'source.st', Path(work)/'native.st'
            values = {
                'transformer_blocks.0.attn.to_k.weight': [3, 4],
                'transformer_blocks.0.attn.to_q.weight': [1, 2],
                'transformer_blocks.0.attn.to_v.weight': [5, 6],
                'norm_out.linear.weight': [7, 8, 9, 10],
                'single_transformer_blocks.0.attn.norm_q.weight': [11, 12],
            }
            header, data = {}, b''
            for name, value in values.items():
                raw = struct.pack('<'+'H'*len(value), *value)
                shape = [len(value)] if 'norm_q' in name else [len(value), 1]
                header[name] = dict(dtype='BF16', shape=shape, data_offsets=[len(data), len(data)+len(raw)])
                data += raw
            encoded = json.dumps(header).encode()
            source.write_bytes(struct.pack('<Q', len(encoded))+encoded+data)
            receipt = export(source, output)
            self.assertEqual(receipt['tensors'], 3)
            raw = output.read_bytes(); size = struct.unpack_from('<Q', raw)[0]
            header = json.loads(raw[8:8+size]); data = raw[8+size:]
            expected = {'double_blocks.0.img_attn.qkv.weight': [1, 2, 3, 4, 5, 6],
                        'final_layer.adaLN_modulation.1.weight': [9, 10, 7, 8],
                        'single_blocks.0.norm.query_norm.scale': [11, 12]}
            for name, value in expected.items():
                lo, hi = header[name]['data_offsets']
                self.assertEqual(list(struct.unpack('<'+'H'*len(value), data[lo:hi])), value)
            self.assertEqual(header['double_blocks.0.img_attn.qkv.weight']['shape'], [6, 1])
            source.write_bytes(source.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'ranges'):
                export(source, Path(work)/'truncated.st')


if __name__ == '__main__':
    unittest.main()
