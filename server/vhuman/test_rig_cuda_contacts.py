"""Opt-in CUDA regression: VHUMAN_TEST_CUDA=1 python -m unittest ..."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .rig import native, rigdef
from .test_rig import _toy_rig


@unittest.skipUnless(os.environ.get('VHUMAN_TEST_CUDA') == '1', 'explicit CUDA test required')
class SharedLipContactTests(unittest.TestCase):
    def test_shared_upper_vertex_matches_ordered_cpu_projection(self):
        # The moving upper point crosses two stationary lower points. These
        # pairs cannot run concurrently: both update the same upper point.
        rest = np.array([[0, .005, .1], [-.01, 0, .1], [.01, 0, .1]])
        joints = np.zeros((3, 4), np.int32)
        joints[0, 0] = 1
        weights = np.zeros((3, 4))
        weights[:, 0] = 1
        viz = dict(eye=[dict(ids=[2], joint=0, center=[0, 0, -10], radius_mm=0)],
                   margins_mm=dict(eye=0, sphere=0), lip_ids=[0, 1, 2],
                   spheres=dict(dummy=dict(centers=[[0, 0, -10]], joints=[[0, 0]],
                                           weights=[[1, 0]], radius_mm=[0])),
                   pairs_upper=[0, 0], pairs_lower=[1, 2], head=0, jaw=1,
                   verts=[], nbr_ptr=[0], nbr_idx=[])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'tmp') as tmp:
            package = Path(tmp)/'rig.safetensors'
            native.write_package(package, _toy_rig(), rest,
                                 dict(mouthSmileLeft=np.zeros_like(rest), c=np.zeros_like(rest)),
                                 joints, weights, contacts_viz=viz)
            lib = native.build_gpu_library(Path(tmp), backend='cuda')
            gpu = native.NativeGPU(lib, package)
            try:
                controls = np.zeros((19, gpu.C), np.float32)
                controls[:, rigdef.Rig(_toy_rig()).cidx['jawOpen']] = np.linspace(.3, 1, 19)
                gpu.set_contact_iterations(0)
                raw = gpu.eval_batch(controls)
                self.assertLess(float(raw[-1, 0, 1]-raw[-1, 1, 1]), -.02)
                gpu.set_contact_iterations(4)
                expected = gpu.eval_batch(controls)
                self.assertGreater(float(expected[-1, 0, 1]-raw[-1, 0, 1]), .01)
                for _ in range(8):
                    actual, _ = gpu.eval_gpu(controls)
                    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
            finally:
                gpu.close()


if __name__ == '__main__':
    unittest.main()
