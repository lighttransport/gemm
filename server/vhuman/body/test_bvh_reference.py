"""BVH channel order, clock and bounded-input checks without model assets."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ..service import ROOT
from .bvh_reference import read_bvh


HEADER = '''HIERARCHY
ROOT Hips
{
 OFFSET 0 100 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT Head
 {
  OFFSET 0 50 0
  CHANNELS 3 Zrotation Xrotation Yrotation
  End Site
  {
   OFFSET 0 10 0
  }
 }
}
MOTION
Frames: 5
Frame Time: 0.033333
'''


class BVHReferenceTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=ROOT/'tmp')
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)/'motion.bvh'

    def write(self, rows, header=HEADER):
        self.path.write_text(header+'\n'.join(' '.join(map(str, row)) for row in rows)+'\n')

    def test_root_translation_hierarchy_and_declared_euler_order(self):
        self.write([[i, 100, 0, 90, 0, 0, 0, 0, 0] for i in range(5)])
        data = read_bvh(self.path)
        self.assertEqual(data['names'], ['Hips', 'Head'])
        np.testing.assert_allclose(data['positions'][:, 0, 0], np.arange(5)*.01)
        np.testing.assert_allclose(data['positions'][0, 1], [-.5, 1, 0], atol=1e-12)
        np.testing.assert_allclose(data['rest_positions'][1], [0, 1.5, 0])
        np.testing.assert_allclose(np.diff(data['times']), .033333)

    def test_intrinsic_rotations_are_composed_in_channel_order(self):
        self.write([[0, 100, 0, 90, 90, 0, 0, 0, 0]]*5)
        # Rz(90) Rx(90) maps +Z to +X; reversing the order maps it to -Y.
        data = read_bvh(self.path)
        np.testing.assert_allclose(data['matrices'][0, 0]@[0, 0, 1], [1, 0, 0], atol=1e-12)

    def test_truncated_nonfinite_and_oversized_clock_rejected(self):
        for rows, header in (
            ([[0]*9]*4, HEADER),
            ([[float('nan')]*9]*5, HEADER),
            ([[0]*9]*5, HEADER.replace('0.033333', '3.0')),
            ([[0]*9]*5, HEADER.replace('Zrotation Xrotation Yrotation', 'Zrotation Xrotation Xrotation')),
        ):
            with self.subTest(header=header, rows=len(rows)):
                self.write(rows, header)
                with self.assertRaises(ValueError): read_bvh(self.path)

    def test_trailing_non_numeric_data_rejected(self):
        self.write([[0]*9]*5)
        self.path.write_text(self.path.read_text()+'unexpected\n')
        with self.assertRaises(ValueError):
            read_bvh(self.path)


if __name__ == '__main__':
    unittest.main()
