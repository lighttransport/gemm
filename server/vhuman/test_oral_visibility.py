import unittest
import numpy as np

from .reconstruction.oral_visibility import source_tooth_mask, overlap_metrics


class OralVisibilityTests(unittest.TestCase):
    def test_mask_rejects_dark_saturated_and_outside_pixels(self):
        image = np.full((20, 20, 3), 220, np.uint8)
        polygon = np.array([[5, 5], [8, 5], [10, 5], [12, 5],
                            [15, 5], [15, 15], [10, 15], [5, 15]])
        image[8, 8] = [70, 70, 70]
        image[9, 9] = [255, 140, 140]
        mask, interior = source_tooth_mask(image, polygon)
        self.assertTrue(mask[10, 10])
        self.assertFalse(mask[8, 8])
        self.assertFalse(mask[9, 9])
        self.assertFalse(mask[1, 1])
        self.assertTrue(interior[8, 8])

    def test_overlap_separates_coverage_and_spill(self):
        target = np.array([[True, True], [False, False]])
        pred = np.array([[True, False], [True, False]])
        report = overlap_metrics(pred, target, target)
        self.assertAlmostEqual(report['iou'], 1/3)
        self.assertEqual(report['target_recall'], .5)
        self.assertEqual(report['teeth_outside_mouth_pixels'], 1)
        empty = np.zeros((2, 2), bool)
        self.assertIsNone(overlap_metrics(empty, empty, empty)['iou'])

    def test_invalid_source_and_landmarks(self):
        with self.assertRaises(ValueError):
            source_tooth_mask(np.zeros((20, 20, 3)), np.zeros((8, 2)))
        with self.assertRaises(ValueError):
            source_tooth_mask(np.zeros((20, 20, 3), np.uint8), np.full((8, 2), np.nan))


if __name__ == '__main__':
    unittest.main()
