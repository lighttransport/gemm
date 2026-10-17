"""CPU-only native boundary tests; no driver initialization or GPU execution."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from .rig import contacts, native, safetensors

ROOT = Path(__file__).resolve().parents[2]


def tensors():
    t = {'controls.range': np.array([[0., 1.]]),
         'joints.matrix': np.zeros((6, 1)), 'joints.parent': np.array([-1]),
         'joints.rest_t': np.zeros((1, 3)), 'joints.rest_R': np.eye(3)[None],
         'joints.inv_bind': np.eye(4)[None], 'shapes.src': np.array([0]),
         'morph': np.zeros((1, 3, 3)), 'rest': np.array([[0., -.03, 0], [0, .03, 0], [0, .03, 0]]),
         'skin.joints': np.zeros((3, 4), np.int32),
         'skin.weights': np.tile([1., 0, 0, 0], (3, 1))}
    for name, value in dict(eye_ids=np.zeros(0, np.int32), eye_joint=np.zeros(0, np.int32),
                            eye_center=np.zeros((0, 3)), eye_thr=np.zeros(0),
                            lip_ids=np.zeros(0, np.int32), sph_center=np.zeros((0, 3)),
                            sph_joint=np.zeros((0, 2), np.int32), sph_weight=np.zeros((0, 2)),
                            sph_thr=np.zeros((0, 0)), pair_u=np.array([0, 0]), pair_l=np.array([1, 2]),
                            pair_floor=np.zeros(2), up_joints=np.array([0, 0]),
                            verts=np.arange(3), nbr_ptr=np.array([0, 1, 2, 2]), nbr_idx=np.array([1, 0])).items():
        t['contact.'+name] = value
    return t


class NativeSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = ROOT/'tmp/test-native-safety'
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=scratch)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        # Use the production fast-math build: NaN rejection must survive it.
        cls.lib = native.build_library(cls.root)

    def load(self, t):
        path = self.root/'rig.safetensors'
        safetensors.save(path, t)
        return native.Native(self.lib, path)

    def test_rejects_invalid_shapes_indices_and_nonfinite_data(self):
        changes = {
            'short rest': ('rest', np.zeros((1, 3))),
            'joint index': ('skin.joints', np.ones((3, 4), np.int32)),
            'parent cycle': ('joints.parent', np.array([0])),
            'parent outside': ('joints.parent', np.array([64])),
            'bad weight': ('skin.weights', np.zeros((3, 4))),
            'NaN': ('rest', np.full((3, 3), np.nan)),
            'infinity': ('morph', np.full((1, 3, 3), np.inf)),
            'bad source': ('shapes.src', np.array([1])),
            'bad pair': ('contact.pair_u', np.array([0, 3])),
            'duplicate eye': ('contact.eye_ids', np.array([0, 0])),
            'bad CSR end': ('contact.nbr_ptr', np.array([0, 1, 2, 9])),
            'bad CSR order': ('contact.nbr_ptr', np.array([0, 2, 1, 2])),
            'bad neighbor': ('contact.nbr_idx', np.array([3, 0])),
            'bad sphere shape': ('contact.sph_center', np.zeros((1, 2))),
            'partial ML': ('ml.fc1.weight', np.ones((1, 1))),
        }
        for label, (key, value) in changes.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                t = tensors(); t[key] = value
                self.load(t)

    def test_tensor_byte_count_must_match_shape(self):
        path = self.root/'bad-bytes.safetensors'
        safetensors.save(path, tensors())
        raw = path.read_bytes()
        n = struct.unpack('<Q', raw[:8])[0]
        header = json.loads(raw[8:8+n])
        header['rest']['data_offsets'][1] -= 4
        h = json.dumps(header).encode(); h += b' ' * (-len(h) % 8)
        path.write_bytes(struct.pack('<Q', len(h))+h+raw[8+n:])
        with self.assertRaises(ValueError):
            native.Native(self.lib, path)

    def test_ml_and_corrective_shapes_and_indices(self):
        t = tensors()
        t.update({'joints.matrix': np.zeros((6, 2)), 'corr.inputs': np.array([[0]]),
                  'corr.weight': np.ones(1), 'morph': np.zeros((3, 3, 3)),
                  'ml.fc1.weight': np.zeros((2, 2)), 'ml.fc1.bias': np.zeros(2),
                  'ml.fc2.weight': np.zeros((1, 2)), 'ml.fc2.bias': np.zeros(1),
                  'ml.input.mean': np.zeros(2), 'ml.input.scale': np.ones(2),
                  'ml.output.mean': np.zeros(1), 'ml.output.scale': np.ones(1)})
        n = self.load(t)
        try:
            self.assertTrue(np.isfinite(n.eval_batch(np.zeros((2, 1)))).all())
        finally:
            n.close()
        for key, value in (('corr.inputs', np.array([[1]])), ('ml.output.scale', np.zeros(1)),
                           ('ml.fc2.weight', np.zeros((1, 3)))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(dict(t, **{key: value}))

    def test_shared_pairs_match_cpu_projection(self):
        t = tensors()
        # No smoothing graph: isolate ordered shared-vertex projection.
        for key in ('verts', 'nbr_ptr', 'nbr_idx'):
            del t['contact.'+key]
        n = self.load(t)
        try:
            expected = contacts.project(t['rest'], np.eye(4)[None], t, smooth=0)
            actual = n.eval([0])
            self.assertGreater(actual[0, 1]-t['rest'][0, 1], .03)
            np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=0)
        finally:
            n.close()

    def test_control_validation_and_closed_handles_all_backends(self):
        for cls in (native.Native, native.NativeGPU, native.NativeVK):
            with self.subTest(backend=cls.__name__):
                n = cls.__new__(cls)
                n.h, n.g, n.C, n.V = 1, 1, 2, 3
                n.lib = Mock()
                for x in ([], [0], [0, 0, 0], 0, [np.nan, 0], [[0, 0]]):
                    with self.assertRaises(ValueError):
                        n.eval(x)
                for x in ([0, 0], [[0]], np.zeros((1, 1, 2)), [[0, np.inf]]):
                    with self.assertRaises(ValueError):
                        n.eval_batch(x)
                    if cls is not native.Native:
                        with self.assertRaises(ValueError):
                            n.eval_gpu(x)
                self.assertEqual(n.lib.mock_calls, [])
                n.close()
                for call in (lambda: n.eval([0, 0]), lambda: n.eval_batch([[0, 0]]),
                             lambda: n.set_contact_iterations(1)):
                    with self.assertRaises(RuntimeError):
                        call()
                if cls is not native.Native:
                    with self.assertRaises(RuntimeError):
                        n.eval_gpu([[0, 0]])

    def test_gpu_host_failures_with_cpu_driver_doubles(self):
        exe = self.root/'driver-test'
        cmd = ['gcc', '-O1', '-g', '-Wall', '-Wextra', '-mavx2', '-mfma',
               '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-no-pie',
               '-o', str(exe), str(ROOT/'cuda/vhuman/test_vhuman_driver.c'),
               str(ROOT/'cuda/cuew.c'), *[str(ROOT/s) for s in native.SOURCES], '-ldl', '-lm', '-lpthread']
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        package = self.root/'driver.safetensors'; safetensors.save(package, tensors())
        # LeakSanitizer cannot run under this workspace's ptrace sandbox.
        env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0')
        result = subprocess.run([str(exe), str(package)], capture_output=True, text=True, timeout=30, env=env)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        for key, value in (('skin.joints', np.ones((3, 4), np.int32)),
                           ('joints.parent', np.array([64])), ('rest', np.zeros((1, 3))),
                           ('contact.nbr_idx', np.array([3, 0]))):
            with self.subTest(key=key):
                t = tensors(); t[key] = value; safetensors.save(package, t)
                result = subprocess.run([str(exe), str(package), 'reject'], capture_output=True,
                                        text=True, timeout=30, env=env)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_vulkan_resize_failures_with_cpu_runner_double(self):
        source = (ROOT/'vulkan/vhuman/vhuman_deformer_vk.cc').read_text()
        # Compile the exact production allocator against a CPU runner double.
        # This avoids loading Vulkan or requiring compiled shader artifacts.
        start = source.index('static bool ensure(')
        end = source.index('\nint vh_vk_eval_batch(', start)
        (self.root/'vhuman_ensure_under_test.inc').write_text(source[start:end])
        exe = self.root/'vulkan-resize-test'
        subprocess.run(['g++', '-std=c++17', '-O1', '-Wall', '-Wextra',
                        '-fsanitize=address,undefined', '-no-pie', '-I', str(self.root),
                        '-o', str(exe), str(ROOT/'vulkan/vhuman/test_buffer_resize.cc')],
                       check=True, capture_output=True, text=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30,
                                env=dict(os.environ, ASAN_OPTIONS='detect_leaks=0'))
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
