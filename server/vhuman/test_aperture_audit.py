"""Geometric mask metrics with known synthetic outlines."""
import unittest
import numpy as np
from .metahuman.aperture_audit import aperture_mask,compare_masks


class ApertureAuditTests(unittest.TestCase):
    def test_identity_and_known_translation(self):
        a=np.zeros((40,50),bool);a[10:25,12:32]=True
        same=compare_masks(a,a)
        self.assertEqual(same['iou'],1)
        self.assertEqual(same['boundary_mean_px'],0)
        b=np.zeros_like(a);b[12:27,15:35]=True
        moved=compare_masks(a,b)
        np.testing.assert_allclose(moved['centroid_offset_px'],[3,2])
        self.assertAlmostEqual(moved['iou'],221/379)
        self.assertGreater(moved['boundary_mean_px'],0)

    def test_aperture_ignores_speckles_and_fills_holes(self):
        image=np.zeros((30,40,3),np.float32)
        image[5:20,8:30]=1;image[10:13,15:18]=0;image[1,1]=1
        result=aperture_mask(image)
        self.assertEqual(int(result.sum()),330)
        self.assertFalse(result[1,1])
        self.assertTrue(result[11,16])

    def test_empty_and_dimension_mismatch_rejected(self):
        with self.assertRaises(ValueError):aperture_mask(np.zeros((5,5,3)))
        with self.assertRaises(ValueError):compare_masks(np.zeros((5,5),bool),np.zeros((6,5),bool))
        with self.assertRaises(ValueError):compare_masks(np.zeros((5,5),bool),np.zeros((5,5),bool))


if __name__=='__main__':unittest.main()
