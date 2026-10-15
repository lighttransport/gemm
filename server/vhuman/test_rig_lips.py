"""Anatomical lip transfer across a closed or slightly crossed neutral seam."""
from types import SimpleNamespace
from importlib.util import find_spec
import unittest

import numpy as np

from .rig.face_models import _map_values, _surface_map, native_mouth_curves, remap_contacts


class NativeLipTests(unittest.TestCase):
    @unittest.skipUnless(find_spec('scipy'), 'optional scipy unavailable')
    def test_nearby_opposite_lip_cannot_take_jaw_motion(self):
        surface = np.array([[-1, .001, 0], [1, .001, 0], [0, .001, 1],
                            [-1, -.001, 0], [1, -.001, 0], [0, -.001, 1]])
        triangles = np.array([[0, 1, 2], [3, 4, 5]])
        points = np.array([[0, -.0005, .2], [0, .0005, .2]])
        sides = np.array([1, 1, 1, -1, -1, -1])
        # Neutral reconstruction can put the lips closer to the opposite half.
        delta = np.zeros_like(surface)
        delta[3:, 1] = -.02
        ids, bary, _ = _surface_map(points, surface, triangles, np.array([1, -1]), sides)
        moved = _map_values(delta, ids, bary)
        np.testing.assert_allclose(moved[:, 1], [0, -.02])

    def test_native_lip_curves_ignore_the_photographed_mouth(self):
        x = np.arange(5.)
        y = np.array([-3., -2., -1., 1., 2., 3.])
        positions = np.array([[xx, yy, .1*xx] for yy in y for xx in x])
        triangles = []
        for row in range(5):
            for col in range(4):
                if row == 2 and col in (1, 2):
                    continue  # native mouth opening
                a = row*5+col
                triangles.extend([[a, a+1, a+5], [a+1, a+6, a+5]])
        sides = np.zeros(30, np.int8)
        for row in (3, 4):
            sides[row*5+1:row*5+4] = 1
        for row in (1, 2):
            sides[row*5+1:row*5+4] = -1
        source = SimpleNamespace(triangles=np.array(triangles), lip_sides=sides)
        seam, upper, lower = native_mouth_curves(source, positions, samples=9)
        np.testing.assert_allclose(seam[:, 1], 0)
        self.assertEqual(seam.shape, (9, 3))
        self.assertAlmostEqual(upper[4, 1], 2)
        self.assertAlmostEqual(lower[4, 1], -2)
        np.testing.assert_allclose(seam[[0, -1], 0], [1, 3])

    @unittest.skipUnless(find_spec('scipy'), 'optional scipy unavailable')
    def test_contact_pairs_keep_anatomical_order_when_neutral_lips_cross(self):
        procedural = np.array([[0,.001,0], [0,-.001,0]])
        imported = procedural[::-1].copy()
        viz = dict(eye=[], lip_ids=[0,1], pairs_upper=[0], pairs_lower=[1])
        out = remap_contacts(viz, procedural, imported, np.empty((0,3),np.int32),
                             proc_lip_sides=np.array([1,-1]), model_lip_sides=np.array([1,-1]))
        self.assertEqual(out['pairs_upper'], [0])
        self.assertEqual(out['pairs_lower'], [1])

    def test_lod_keeps_nearby_upper_and_lower_sheets_separate(self):
        from .rig.source_lod import simplify
        plane = np.array([[x*.01, y*.01, 0] for y in range(7) for x in range(7)])
        points = np.concatenate([plane, plane+[0,0,.0001]])
        triangles = []
        for offset in (0,49):
            for y in range(6):
                for x in range(6):
                    a = offset+y*7+x
                    triangles.extend([[a,a+1,a+7],[a+1,a+8,a+7]])
        sides = np.r_[np.ones(49),-np.ones(49)]
        _,_,groups,_,_ = simplify(points,np.array(triangles),.003,vertex_regions=sides)
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(sides[groups==group])),1)


if __name__ == '__main__':
    unittest.main()
