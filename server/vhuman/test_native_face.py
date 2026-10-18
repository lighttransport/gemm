"""CPU-only native face math and optional development-oracle comparisons."""
import importlib.util
import builtins
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from . import native_face as nf
from . import native_face_fit as fit

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless((ROOT / 'cpu/vhuman/libvhuman_face.so').is_file(), 'native build required')
class NativeFaceTests(unittest.TestCase):
    def test_gemm_transposes_and_rotation_derivatives(self):
        rng = np.random.default_rng(8)
        a = rng.normal(size=(7, 11)).astype(np.float32)
        b = rng.normal(size=(11, 5)).astype(np.float32)
        for ta in (False, True):
            for tb in (False, True):
                np.testing.assert_allclose(nf.matmul(a.T if ta else a, b.T if tb else b,
                    transpose_a=ta, transpose_b=tb, threads=2), a @ b, atol=2e-6, rtol=2e-6)
        for r in (np.zeros(3), np.array([.2, -.3, .1])):
            _, jac = nf.rodrigues(r, derivatives=True)
            for k in range(3):
                d = np.eye(3)[k] * 1e-6
                numeric = (nf.rodrigues(r + d) - nf.rodrigues(r - d)) / 2e-6
                np.testing.assert_allclose(jac[k], numeric, atol=1e-9)

    def test_registration_finite_difference(self):
        rng = np.random.default_rng(71)
        x = rng.normal(0, .02, (11, 3))
        ids = np.array([1, 3, 4, 7, 9])
        edges = np.array([[i, (i + 1) % 11] for i in range(11)])
        target = rng.normal(0, .02, (5, 3))
        normal = rng.normal(size=(5, 3))
        length = rng.uniform(.001, .03, 11)
        def objective(p):
            return fit.registration(p, ids, target, normal, edges, length)
        _, gradient, _ = objective(x)
        for i in range(len(x)):
            for c in range(3):
                d = np.zeros_like(x); d[i, c] = 1e-7
                numeric = (objective(x + d)[0] - objective(x - d)[0]) / 2e-7
                self.assertAlmostEqual(numeric, gradient[i, c], delta=2e-5)

    def test_area_and_temporal_gradients(self):
        rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        posed = rest[None].copy(); posed[0, 2, 1] = -.2
        tri = np.array([[0, 1, 2]])
        loss, gradient = fit.area(rest, posed, tri)
        self.assertGreater(loss, 0)
        for c in range(9):
            d = np.zeros_like(posed); d.flat[c] = 1e-3
            numeric = (fit.area(rest, posed + d, tri)[0] - fit.area(rest, posed - d, tri)[0]) / .002
            self.assertAlmostEqual(numeric, gradient.flat[c], delta=2e-4)
        rng = np.random.default_rng(3)
        expr = rng.normal(size=(4, 5)).astype(np.float32)
        rot = rng.normal(size=(4, 4, 3)).astype(np.float32)
        times = np.array([0., .03, .065, .12])
        _, ge, gr = fit.temporal(expr, rot, times)
        for a, g in ((expr, ge), (rot, gr)):
            for k in range(a.size):
                old = a.flat[k]; a.flat[k] = old + .001
                hi = fit.temporal(expr, rot, times)[0]
                a.flat[k] = old - .001
                lo = fit.temporal(expr, rot, times)[0]; a.flat[k] = old
                self.assertAlmostEqual((hi-lo)/.002, g.flat[k], delta=2e-5)

    def test_lbfgs_quadratic_and_callback_error(self):
        x = np.array([5., -3.], np.float32)
        def objective(p):
            delta = p - np.array([1., 2.])
            return float((delta**2).sum()), 2*delta
        fit.lbfgs(x, objective, 50, 100)
        np.testing.assert_allclose(x, [1, 2], atol=3e-5)
        before = x.copy()
        def fail(_):
            raise InterruptedError('cancelled')
        with self.assertRaisesRegex(InterruptedError, 'cancelled'):
            fit.lbfgs(x, fail, 50, 100)
        np.testing.assert_array_equal(x, before)

    def test_native_model_without_framework_imports(self):
        from .face_assets import asset_path
        if not asset_path('gnm').is_file():
            self.skipTest('pinned GNM asset unavailable')
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'cv2', 'onnx', 'onnxruntime', 'mediapipe', 'tensorflow', 'threadpoolctl'):
                raise AssertionError('unexpected runtime import: '+name)
            return original(name, *args, **kwargs)
        with patch.object(builtins, '__import__', side_effect=guarded):
            from .rig.gnm_model import GNMModel
            model = GNMModel()
            rng = np.random.default_rng(19)
            beta = rng.normal(0, .1, model.identity_dim).astype(np.float32)
            expr = rng.normal(0, .1, (1, model.expression_dim)).astype(np.float32)
            rot = rng.normal(0, .1, (1, 4, 3)).astype(np.float32)
            trans = rng.normal(0, .01, (1, 3)).astype(np.float32)
            expected, joints = model.evaluate(beta, expr[0], rot[0], trans[0])
            ids = np.arange(0, len(expected), 97)
            with nf.Gnm(model, ids, threads=2) as native:
                actual, actual_joints = native.evaluate(beta, expr, rot, trans)
            np.testing.assert_allclose(actual[0], expected[ids], atol=2e-5)
            np.testing.assert_allclose(actual_joints[0], joints, atol=2e-5)
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                native.evaluate(beta, expr, rot, trans)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional CPU development oracle')
    def test_gnm_reverse_reference(self):
        import torch
        from .face_assets import asset_path
        from .rig.gnm_model import GNMModel
        if not asset_path('gnm').is_file():
            self.skipTest('pinned GNM asset unavailable')
        model = GNMModel(device='cpu')
        rng = np.random.default_rng(11)
        ids = np.arange(0, len(model.data['template_vertex_positions']), 97)
        values = [rng.normal(0, .1, shape).astype(np.float32) for shape in
                  ((model.identity_dim,), (2, model.expression_dim), (2, 4, 3), (2, 3))]
        residual = rng.normal(0, .001, model.data['template_vertex_positions'].shape).astype(np.float32)
        upstream = rng.normal(size=(2, len(ids), 3)).astype(np.float32)
        joint_upstream = rng.normal(size=(2, 4, 3)).astype(np.float32)
        tensors = [torch.tensor(v, device='cpu', requires_grad=True) for v in values]
        evaluate = model.frame_evaluator(values[0], ids, bind_residual=residual, fit_identity=True)
        p, j = evaluate(*tensors[1:], identity=tensors[0])
        ((p * torch.tensor(upstream)).sum() + (j * torch.tensor(joint_upstream)).sum()).backward()
        with nf.Gnm(model, ids, threads=2) as native:
            actual, joints, gradients = native.evaluate(*values, bind_residual=residual[ids],
                upstream=upstream, joint_upstream=joint_upstream)
        np.testing.assert_allclose(actual, p.detach().numpy(), atol=2e-5)
        np.testing.assert_allclose(joints, j.detach().numpy(), atol=2e-5)
        for gradient, tensor in zip(gradients, tensors):
            np.testing.assert_allclose(gradient, tensor.grad.numpy(), atol=2e-5, rtol=2e-4)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional CPU development oracle')
    def test_adam_reference(self):
        import torch
        rng = np.random.default_rng(5)
        for dtype in (np.float32, np.float64):
            x = rng.normal(size=(6, 3)).astype(dtype)
            reference = torch.tensor(x, device='cpu', requires_grad=True)
            optimizer = torch.optim.Adam([reference], lr=.003)
            native = fit.Adam(x, .003)
            for _ in range(20):
                g = rng.normal(size=x.shape).astype(dtype)
                reference.grad = torch.tensor(g, device='cpu')
                optimizer.step(); native.step(g)
            np.testing.assert_allclose(x, reference.detach().numpy(), atol=1e-6)

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'optional image development oracle')
    def test_resize_reference(self):
        import cv2
        x = np.random.default_rng(15).integers(0, 256, (379, 511, 3), dtype=np.uint8)
        expected = cv2.resize(x, (512, 512))
        actual = nf.resize(x, (512, 512), quantize=True)
        self.assertLessEqual(np.max(abs(expected-actual)), 1)
        self.assertGreater(np.mean(expected == actual), .999)
        with self.assertRaisesRegex(RuntimeError, 'byte values'):
            nf.resize(np.full((2, 2, 3), 1e20), (3, 3), quantize=True)


if __name__ == '__main__':
    unittest.main()
