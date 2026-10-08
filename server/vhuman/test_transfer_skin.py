"""Photographic protection and eligibility for changed-geometry prior transfer."""
import unittest
import numpy as np
from .reconstruction.transfer_skin import blend_prior


class PriorTransferTests(unittest.TestCase):
    def test_only_eligible_unobserved_pixels_change(self):
        base = np.full((2, 3, 3), 40, np.uint8)
        prior = np.full_like(base, 200)
        observed = np.array([[True, False, False], [False, False, True]])
        eligible = np.array([[True, True, False], [True, True, True]])
        weight = np.array([[1., 1., 1.], [0., .5, 1.]])
        result = blend_prior(base, prior, observed, eligible, weight)
        np.testing.assert_array_equal(result[observed], base[observed])
        np.testing.assert_array_equal(result[0, 1], prior[0, 1])
        np.testing.assert_array_equal(result[0, 2], base[0, 2])
        np.testing.assert_array_equal(result[1, 0], base[1, 0])
        self.assertTrue(np.all((result[1, 1] > 40) & (result[1, 1] < 200)))
        np.testing.assert_array_equal(base, np.full_like(base, 40))

    def test_invalid_weight_rejected(self):
        base = np.zeros((2, 2, 3), np.uint8)
        mask = np.zeros((2, 2), bool)
        for value in (np.nan, -1., 1.1):
            with self.assertRaises(ValueError):
                blend_prior(base, base, mask, ~mask, np.full((2, 2), value))

class PriorTransferIntegrationTests(unittest.TestCase):
    def fixture(self, root, name, shift, completed):
        import json
        from PIL import Image
        from .reconstruction.observations import sha256
        path = root/name
        path.mkdir()
        vertices = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.]])
        vertices[:, 2] += shift
        np.savez_compressed(path/'geometry.npz', captured=vertices[None],
            triangles=np.array([[0, 1, 2]]), triangle_uvs=np.array([[[0., 0.], [1., 0.], [0., 1.]]]))
        Image.new('RGB', (8, 8), (90, 80, 70)).save(path/'portrait.png')
        Image.new('RGB', (8, 8), (160, 150, 140) if completed else (70, 60, 50)).save(path/'skin_basecolor.png')
        coverage = np.zeros((8, 8), np.uint8)
        coverage[:, :2 if completed else 3] = 255
        Image.fromarray(coverage).save(path/'skin_coverage.png')
        manifest = dict(format='vhuman.reconstruction.v1', face_model='gnm_v3',
            geometry_sha256=sha256(path/'geometry.npz'), portrait_sha256=sha256(path/'portrait.png'), material={})
        if completed:
            Image.new('L', (8, 8), 64).save(path/'skin_generated_support.png')
            report = dict(schema='vhuman.synthetic_skin_completion.v1',
                source_geometry_sha256=manifest['geometry_sha256'],
                basecolor_sha256=sha256(path/'skin_basecolor.png'), license='fixture', unseen_covered=.99)
            manifest['material']['synthetic_completion'] = report
            (path/'generated_skin.json').write_text(json.dumps(report))
        (path/'manifest.json').write_text(json.dumps(manifest))
        return path

    def test_transfer_binds_new_geometry_and_drops_old_coverage_claim(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from .reconstruction.transfer_skin import transfer
        from .reconstruction.provenance import validate_candidate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = self.fixture(root, 'target', .001, False)
            prior = self.fixture(root, 'prior', 0., True)
            out = root/'out'
            report = transfer(target, prior, out, feather_mm=1.)
            self.assertGreater(report['generated_texels'], 0)
            self.assertEqual(report['photographed_texels_changed'], 0)
            self.assertNotIn('unseen_covered', report)
            self.assertEqual(report['source_geometry_sha256'], validate_candidate(target)['geometry_sha256'])
            self.assertNotEqual(report['source_geometry_sha256'], report['prior_transfer']['source_geometry_sha256'])
            observed = np.asarray(Image.open(target/'skin_coverage.png')) > 0
            np.testing.assert_array_equal(np.asarray(Image.open(out/'skin_basecolor.png'))[observed],
                                          np.asarray(Image.open(target/'skin_basecolor.png'))[observed])
            validate_candidate(out)
            Image.new('L', (8, 8), 0).save(out/'skin_prior_transfer.png')
            with self.assertRaisesRegex(ValueError, 'transfer asset hash'):
                validate_candidate(out)

    def test_large_geometry_change_rejected_before_writing(self):
        import tempfile
        from pathlib import Path
        from .reconstruction.transfer_skin import transfer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = self.fixture(root, 'target', .01, False)
            prior = self.fixture(root, 'prior', 0., True)
            with self.assertRaisesRegex(ValueError, 'displacement'):
                transfer(target, prior, root/'out')
            self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
