"""Asset preparation works without importing training/inference frameworks."""
import builtins
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


def fixture():
    from .rig import template as T
    rng = np.random.default_rng(19)
    n = T.MOUTH_N * 4 + 10
    tmpl = SimpleNamespace(group=np.arange(n) % 3, ring=np.zeros(n, int),
                           ring_ids=lambda name, k: np.arange(T.MOUTH_N) + (1-k)*T.MOUTH_N)
    feat = SimpleNamespace(eyes=[dict(side=side, center=[x, .06, .04], radius=.012)
                                for side, x in [('right', -.032), ('left', .032)]])
    skel = {'joints': [{'name': n} for n in ('head', 'jaw', 'eye_R', 'eye_L', 'teeth_upper', 'teeth_lower')]}
    teeth = [(SimpleNamespace(positions=rng.normal(size=(362, 3))*.01), joint)
             for joint in ('teeth_upper', 'teeth_lower')]
    return tmpl, feat, skel, teeth, rng.normal(size=(n, 3))*.03


class AssetDependencyTests(unittest.TestCase):
    def test_contact_assets_and_sampling_do_not_import_torch(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'onnx', 'onnxruntime', 'mediapipe'):
                raise AssertionError('unexpected model runtime: ' + name)
            return original(name, *args, **kwargs)
        with patch.object(builtins, '__import__', side_effect=guarded):
            from .rig import mldeformer
            result = mldeformer.export_contacts(*fixture())
            from .rig.rigdef import CONTROLS
            controls = mldeformer.sample_controls(list(CONTROLS), 8)
        self.assertEqual(len(result['eye']), 2)
        self.assertEqual(len(result['spheres']['teeth']['centers']), 4)
        self.assertTrue(np.isfinite(controls).all())

    def test_smoothing_matches_scalar_jacobi_and_preserves_fixed_vertices(self):
        from .rig.expressions import smooth_shapes
        triangles = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
        fields = SimpleNamespace(tmpl=SimpleNamespace(n=4, tris=triangles), eye_k=np.ones(4),
                                 eye_side=np.full(4, -1), lip_k=np.array([0, 1, 1, 1]))
        initial = np.arange(12, dtype=np.float64).reshape(4, 3) / 1000
        expected = initial.copy()
        neighbors = [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]
        for _ in range(4):
            old = expected.copy()
            for i in range(1, 4):
                expected[i] = .5 * (old[i] + sum(old[j] for j in neighbors[i]) / len(neighbors[i]))
        actual = smooth_shapes(fields, {'test': initial})['test']
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(actual[0], initial[0])

    def test_export_matches_training_reference_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest('optional training oracle unavailable')
        from .rig.contact_setup import export_contacts
        from .rig.mldeformer_training import Contacts
        tmpl, feat, skel, teeth, rest = fixture()
        expected = Contacts(tmpl, feat, skel, teeth, 'cpu', rest).export()
        self.assertEqual(export_contacts(tmpl, feat, skel, teeth, rest), expected)


if __name__ == '__main__':
    unittest.main()
