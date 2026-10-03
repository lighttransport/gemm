import argparse
from pathlib import Path
import struct
import tempfile
import unittest
from glm53f_capture_counts import generated_ids_match, validate_counts

ROOT = None


class CountsTest(unittest.TestCase):
    def test_complete_run_and_truncation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            prefix = str(Path(directory) / 'run')
            for phase, length in (('', 1), ('.decode', 129)):
                for rank in range(12):
                    base = '{}{}.rank{:02d}'.format(prefix, phase, rank)
                    Path(base + '.fields').write_text('GLM53F_FIELDS_V1 {} 0 45\n{}SPARSE 3\n'.format(
                        rank, 'HIDDEN 1\n' if not phase and rank == 0 else ''))
                    Path(base + '.layer03.sparse').write_bytes(struct.pack('<i', length))
            for layer in range(3, 45):
                Path('{}.layer{:02d}.routes'.format(prefix, layer)).write_bytes(bytes(129 * 64))
            a, b = Path(directory) / 'a.ids', Path(directory) / 'b.ids'
            a.write_text(' '.join(['42'] * 129)); b.write_text(a.read_text())
            self.assertTrue(generated_ids_match(a, b, 129))
            for phase in ('prefill', 'decode'):
                validate_counts(prefix, 1, 129, phase)
            b.write_text('42')
            self.assertFalse(generated_ids_match(a, b, 129))
            a.write_text('42')
            self.assertFalse(generated_ids_match(a, b, 129))
            Path(prefix + '.decode.rank00.layer03.sparse').write_bytes(struct.pack('<i', 1))
            with self.assertRaises(ValueError):
                validate_counts(prefix, 1, 129, 'decode')
            Path(prefix + '.decode.rank00.layer03.sparse').write_bytes(struct.pack('<i', 129))
            Path(prefix + '.layer03.routes').write_bytes(bytes(64))
            with self.assertRaises(ValueError):
                validate_counts(prefix, 1, 129, 'prefill')
            with self.assertRaises(ValueError):
                validate_counts(prefix, 0, 129, 'prefill')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('temporary_root')
    ROOT = parser.parse_args().temporary_root
    unittest.main(argv=['capture_counts'], verbosity=2)
