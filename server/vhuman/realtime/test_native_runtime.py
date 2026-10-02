"""Native CUDA ownership and presentation contracts; CPU tests need no Torch."""
import ctypes as C
import gc
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from .src.avatar.cuda_runtime import CudaRuntime, library, ROOT
from .src.avatar.native_gpu import DeviceView, Managed, _views


class OwnershipTests(unittest.TestCase):
    def test_runtime_modules_import_without_model_frameworks(self):
        # Fresh interpreter: already-imported training modules cannot mask a dependency.
        script = '''
import importlib, importlib.abc, sys
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'onnx', 'onnxruntime', 'tensorflow', 'mediapipe', 'diffusers'}:
            raise AssertionError('unexpected inference dependency: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
for name in ('server.vhuman.cli', 'server.vhuman.video', 'server.vhuman.rig.mldeformer',
             'server.vhuman.rig.expressions', 'server.vhuman.realtime.__main__',
             'server.vhuman.realtime.src.pipeline.live',
             'server.vhuman.realtime.src.avatar.native_identity',
             'server.vhuman.realtime.src.benchmark.motion',
             'server.vhuman.realtime.src.benchmark.appearance',
             'server.vhuman.realtime.src.animation.train',
             'server.vhuman.rig.soft_deformer',
             'server.vhuman.rig.mldeformer_training',
             'server.vhuman.rig.native_corrective',
             'server.vhuman.reconstruction.learned_cues',
             'server.vhuman.realtime.src.renderer.gaussian'):
    importlib.import_module(name)
from server.vhuman.video_backend import select
assert select().frames == (81,)
'''
        subprocess.run([sys.executable, '-c', script], cwd=ROOT, check=True,
                       capture_output=True, text=True, timeout=30)

    def test_runtime_refuses_live_events_and_borrowers(self):
        lib = Mock()
        lib.vh_cuda_open.return_value = 123
        lib.vh_cuda_stream.return_value = 456
        lib.vh_cuda_record.return_value = 789
        lib.vh_cuda_event_sync.return_value = 0
        runtime = CudaRuntime(lib=lib)
        event = runtime.record()
        with self.assertRaisesRegex(RuntimeError, 'events'):
            runtime.close()
        event.synchronize()
        event.close(); event.close()
        runtime.borrowers = 1
        with self.assertRaises(RuntimeError):
            runtime.close()
        runtime.borrowers = 0
        runtime.close(); runtime.close()
        lib.vh_cuda_close.assert_called_once_with(123)
        lib.vh_cuda_event_free.assert_called_once_with(123, 789)
        with self.assertRaises(RuntimeError):
            runtime.record()

    def test_capsule_retains_owner_and_unconsumed_capsule_is_released(self):
        runtime = SimpleNamespace(device=0, cuda_stream=37, check_open=lambda: None)
        owner = SimpleNamespace(runtime=runtime, live_views=0)
        view = DeviceView(owner, 4096, (3, 3))
        before = len(_views)
        capsule = view.__dlpack__(stream=37)
        del view
        self.assertEqual(owner.live_views, 1)
        del capsule
        gc.collect()
        self.assertEqual(owner.live_views, 0)
        self.assertEqual(len(_views), before)

    def test_consumed_capsule_deleter_and_stream_contract(self):
        runtime = SimpleNamespace(device=0, cuda_stream=37, check_open=lambda: None)
        owner = SimpleNamespace(runtime=runtime, live_views=0)
        view = DeviceView(owner, 4096, (3, 3))
        with self.assertRaisesRegex(ValueError, 'stream'):
            view.__dlpack__(stream=99)
        capsule = view.__dlpack__(stream=37)
        get_pointer = C.pythonapi.PyCapsule_GetPointer
        pointer = get_pointer(C.c_void_p(id(capsule)), b'dltensor')
        managed = C.cast(pointer, C.POINTER(Managed))
        rename = C.pythonapi.PyCapsule_SetName
        rename.argtypes = [C.py_object, C.c_char_p]
        rename.restype = C.c_int
        self.assertEqual(rename(capsule, b'used_dltensor'), 0)
        del view, capsule
        self.assertEqual(owner.live_views, 1)
        managed.contents.deleter(managed)
        self.assertEqual(owner.live_views, 0)


@unittest.skipUnless((ROOT / 'cuda/vhuman/libvhuman_runtime.so').is_file(), 'build native presentation library')
class PresentationTests(unittest.TestCase):
    def test_native_srgb_and_straight_alpha_match_independent_formula(self):
        lib = library()
        rng = np.random.default_rng(104)
        rgba = rng.uniform(-.1, 1.2, (41, 67, 4)).astype(np.float32)
        rgba[0, :4] = [[0, 0, 0, 0], [1, 1, 1, 1], [.0031308, .001, .18, 1], [.25, .125, 0, .5]]
        for alpha in (False, True):
            got = np.empty((*rgba.shape[:2], 4 if alpha else 3), np.uint8)
            self.assertEqual(lib.vh_pixels_cpu(rgba.ctypes.data, 41*67, .18, int(alpha), got.ctypes.data), 0)
            x = rgba.astype(np.float64)
            a = np.clip(x[..., 3:4], 0, 1)
            linear = (np.clip(x[..., :3], 0, 1) / np.maximum(a, 1e-8) if alpha
                      else x[..., :3] + (1-x[..., 3:4])*.18)
            linear = np.clip(linear, 0, 1)
            srgb = np.where(linear <= .0031308, 12.92*linear, 1.055*np.power(linear, 1/2.4)-.055)
            expected = np.rint(np.concatenate((srgb, a), -1)*255 if alpha else srgb*255).astype(np.uint8)
            self.assertLessEqual(abs(got.astype(int)-expected).max(), 1)
        rgba[0, 0, 0] = np.nan
        self.assertNotEqual(lib.vh_pixels_cpu(rgba.ctypes.data, 41*67, .18, 0, got.ctypes.data), 0)

    def test_native_doctor_does_not_import_frameworks(self):
        if importlib.util.find_spec('torch') is not None:
            self.skipTest('framework-free interpreter check')
        from .__main__ import doctor
        result = doctor()
        self.assertFalse(result['torch_installed'])


if __name__ == '__main__':
    unittest.main()
