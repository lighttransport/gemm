#!/usr/bin/env python3
import argparse
from pathlib import Path
import tempfile
import unittest
import numpy as np
from compare_glm53f_fields import compare, read_capture

ROOT = None


def capture(prefix, pipeline):
    for rank in range(12):
        stage, tp = (rank // 4, rank % 4) if pipeline else (0, rank)
        first, end = ([0, 15, 30][stage], [15, 30, 45][stage]) if pipeline else (0, 45)
        ranks = 4 if pipeline else 12
        h0, h1 = 64 * tp // ranks, 64 * (tp + 1) // ranks
        hn = h1 - h0
        base = f'{prefix}.rank{rank:02d}'
        lines = [f'GLM53F_FIELDS_V1 {rank} {first} {end}']
        for layer in range(first, end):
            if layer % 4 != 3:
                data = np.concatenate((np.ones(hn * 128 * 128, dtype='<f4'), np.zeros(hn * 3 * 128 * 4, dtype='<f4')))
                data.tofile(f'{base}.layer{layer:02d}.kda')
                lines.append(f'KDA {layer} {h0} {hn}')
            else:
                with open(f'{base}.layer{layer:02d}.sparse', 'wb') as f:
                    np.array([1, 0, 0, 0, 0], dtype='<i4').tofile(f)
                    np.ones(512 + 128 + 128, dtype='<f4').tofile(f)
                    np.array([0], dtype='<i4').tofile(f)
                lines.append(f'SPARSE {layer}')
        if rank == (8 if pipeline else 0):
            np.ones(16384, dtype='<f4').tofile(base + '.streams')
            lines.append('STREAMS')
            np.ones(16384, dtype='<f4').tofile(str(prefix) + '.hidden')
            lines.append('HIDDEN 1')
        Path(base + '.fields').write_text('\n'.join(lines) + '\n')
    for layer in range(3, 45):
        with open(f'{prefix}.layer{layer:02d}.routes', 'wb') as f:
            np.arange(8, dtype='<i4').tofile(f)
            np.full(8, 1 / 8, dtype='<f4').tofile(f)


class FieldsTest(unittest.TestCase):
    def test_cross_layout_and_rejections(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            a, b = str(Path(directory) / 'tp12'), str(Path(directory) / 'pp')
            capture(a, False); capture(b, True)
            self.assertTrue(compare(a, b, 0)['pass'])
            path = f'{b}.rank00.layer00.kda'
            data = np.memmap(path, dtype='<f4', mode='r+')
            data[:128 * 128] += 1e-4; data.flush()
            self.assertTrue(compare(a, b, 1e-3)['pass'])
            self.assertFalse(compare(a, b, 0)['pass'])
            data[:128 * 128] = 1.01; data.flush()
            self.assertFalse(compare(a, b, 1e-3)['pass'])
            data[:128 * 128] = 1.0
            data[16 * 128 * 128] = 1e-9; data.flush()
            result = compare(a, b, 1e-3)
            self.assertTrue(any(x.get('reason') == 'zero_norm_requires_exact' for x in result['failures']))
            data[16 * 128 * 128] = 0; data[0] = np.nan; data.flush()
            self.assertFalse(compare(a, b, 1e-3)['pass'])
            data[0] = 1; data.flush(); del data
            route = np.memmap(f'{b}.layer03.routes', dtype='<i4', mode='r+')
            route[0] = 9; route.flush(); del route
            result = compare(a, b, 1e-3)
            self.assertTrue(result['pass']); self.assertEqual(result['expert_route_changes']['layer3'], 1)
            self.assertFalse(compare(a, b, 0, bit_exact=True)['pass'])
            route = np.memmap(f'{b}.layer03.routes', dtype='<i4', mode='r+')
            route[0] = 1; route.flush()
            self.assertFalse(compare(a, b, 1e-3)['pass'])
            route[0] = 9; route.flush(); del route
            selection = np.memmap(f'{b}.rank00.layer03.sparse', dtype='<i4', mode='r+')
            selection[-1] = 1; selection.flush()
            with self.assertRaises(ValueError):
                read_capture(b)
            selection[-1] = 0; selection.flush(); del selection
            manifest = Path(b + '.rank00.fields')
            original = manifest.read_text()
            manifest.write_text(original.replace('KDA 0 0 16', 'KDA 0 1 16'))
            with self.assertRaises(ValueError):
                read_capture(b)
            manifest.write_text(original)
            with self.assertRaises(ValueError):
                compare(a, b, 1.1e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('temporary_root')
    args = parser.parse_args(); ROOT = args.temporary_root
    unittest.main(argv=['test_compare_glm53f_fields'], verbosity=2)
