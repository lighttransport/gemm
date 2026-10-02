"""Explicit GPU checks; no synthetic workload is reported as avatar quality."""
from pathlib import Path
import unittest
import numpy as np
import torch
from .src.avatar.native_gpu import NativeSharedGPU
from .src.avatar.bundle import bind, deform

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "tmp/vhuman-independent/heads/291dfa911553/rig/rig_deformer.safetensors"


@unittest.skipUnless(torch.cuda.is_available() and PACKAGE.exists(), "CUDA and diagnostic rig fixture required")
class NativeGPUChecks(unittest.TestCase):
    def test_shared_stream_contacts_match_cpu(self):
        stream = torch.cuda.Stream()
        rig = NativeSharedGPU(PACKAGE, ROOT / "tmp/vhuman-realtime/native", stream=stream)
        try:
            rng = np.random.default_rng(7)
            for _ in range(4):
                controls = rng.random(rig.C).astype(np.float32) * .35
                expected = rig.eval(controls)
                with torch.cuda.stream(stream):
                    vertices = rig.submit(controls)
                    saved = torch.from_dlpack(vertices).clone()
                stream.synchronize()
                np.testing.assert_allclose(saved.cpu().numpy(), expected, atol=2e-6, rtol=2e-5)
                with self.assertRaises(RuntimeError): rig.close()
                del vertices, saved
        finally: rig.close()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GaussianGPUChecks(unittest.TestCase):
    def test_deform_and_rasterize(self):
        from .src.renderer.gsplat_reference import GaussianRenderer
        vertices = np.array([[-.1, -.1, 1], [.1, -.1, 1], [0, .1, 1]], np.float32)
        triangles = np.array([[0, 1, 2]], np.int32)
        avatar = bind(vertices, triangles, ["jawOpen"], count=128)
        renderer = GaussianRenderer(avatar, triangles)
        self.addCleanup(renderer.close)
        tensor = torch.tensor(vertices, device="cuda")
        actual = renderer.deform(tensor)
        expected = deform(avatar, vertices, triangles)
        for gpu, cpu in zip(actual, expected):
            np.testing.assert_allclose(gpu.cpu().numpy(), cpu, atol=2e-7, rtol=1e-4)
        # New fitted bundles use differentiable trace bounds. Exercise a large
        # covariance and saturated expression so both limits actually engage.
        avatar.metadata["covariance_policy"] = "trace-v1"
        avatar.arrays["covariance_local"] *= 100000
        avatar.arrays["color_basis"][:] = 2
        avatar.arrays["expression_matrix"][:] = 1
        renderer = GaussianRenderer(avatar, triangles)
        self.addCleanup(renderer.close)
        controls = np.ones(1, np.float32)
        actual = renderer.deform(tensor, torch.tensor(controls, device="cuda"))
        expected = deform(avatar, vertices, triangles, controls)
        for gpu, cpu in zip(actual, expected):
            np.testing.assert_allclose(gpu.cpu().numpy(), cpu, atol=2e-7, rtol=1e-4)
        view = torch.eye(4, device="cuda")
        intrinsics = torch.tensor([[200., 0, 64], [0, 200, 64], [0, 0, 1]], device="cuda")
        handle = renderer.render(tensor, view, intrinsics, (128, 128))
        self.addCleanup(handle.ready.close)
        handle.ready.synchronize()
        self.assertTrue(torch.isfinite(handle.rgba).all())
        self.assertGreater(float(handle.rgba[..., 3].max()), 0)
        # Presentation goes through native CUDA, with a separately evaluated CPU oracle.
        from .src.avatar.cuda_runtime import library
        rgba = handle.rgba.cpu().numpy()
        for alpha in (False, True):
            expected = np.empty((128, 128, 4 if alpha else 3), np.uint8)
            self.assertEqual(library().vh_pixels_cpu(rgba.ctypes.data, 128*128, .18,
                                                     int(alpha), expected.ctypes.data), 0)
            actual = handle.pixels(straight_alpha=alpha)
            self.assertLessEqual(abs(actual.astype(int)-expected.astype(int)).max(), 1)


if __name__ == "__main__": unittest.main()
