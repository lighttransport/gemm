"""Backend selection and subprocess propagation without loading model weights."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace
from . import gpu, runtime

class RuntimeTest(unittest.TestCase):
    def test_rocm_library_set_includes_the_smi_abi_required_by_torch(self):
        from .rocm_runtime import library_path
        with patch.object(Path, 'is_file', lambda p: str(p).startswith('/opt/rocm/core-10.0/')):
            self.assertEqual(library_path(), '/opt/rocm/core-10.0/lib')

    def test_torch_rocm_uses_configured_ordinal_and_rejects_missing_device(self):
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2, set_device=Mock())
        torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip="7.14"))
        with gpu.execution('rocm', 1):
            self.assertEqual(runtime.torch_device(torch), 'cuda:1')
        cuda.set_device.assert_called_once_with(1)
        cuda.set_device.reset_mock()
        with gpu.execution('rocm', 2), self.assertRaisesRegex(RuntimeError, 'rocm device 2 unavailable'):
            runtime.torch_device(torch)
        cuda.set_device.assert_not_called()

    def test_torch_wrong_build_fails_before_device_selection(self):
        cuda = SimpleNamespace(is_available=lambda: True, set_device=Mock())
        torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip=None))
        with gpu.execution('rocm'), self.assertRaisesRegex(RuntimeError, 'wrong PyTorch build for rocm'):
            runtime.torch_device(torch)
        cuda.set_device.assert_not_called()

    def test_rocm_interpreter_defaults_and_explicit_override(self):
        from argparse import Namespace
        args = Namespace(inference_backend='rocm', device=1, models_root='/mnt/disk1/models',
                         qwen_python=None, rig_python='/explicit/python')
        with gpu.execution('cpu'), patch.object(Path, 'exists', return_value=True):
            runtime.configure_args(args)
            self.assertEqual(Path(args.qwen_python).resolve(),
                             (gpu.ROOT / 'tmp/vhuman-rocm-venv/bin/python').resolve())
            self.assertEqual(args.rig_python, '/explicit/python')
            self.assertEqual(args.sam3d_body_model, str(gpu.model_path('sam3d-body')))

    def test_auto_amd_and_context_restoration(self):
        with patch.object(gpu, 'gpu_status', return_value={'backend':'rocm'}):
            with gpu.execution('auto', 2, '/mnt/disk1/models'):
                self.assertEqual(gpu.backend(), 'rocm')
                self.assertEqual(gpu.device_index(), 2)
                command=runtime.python_command(['python','-m','server.vhuman.rig.build','head'])
                self.assertIn('rocm',command)
                self.assertEqual(command[command.index('--device')+1], '2')
                self.assertEqual(command[-1], 'head')
        self.assertEqual(gpu.device_index(),0)

    def test_explicit_backend_and_validation(self):
        with gpu.execution('cpu', 0, '/configured/models'):
            self.assertIsNone(gpu.gpu_status())
            self.assertEqual(gpu.model_path('qimg-21'),Path('/configured/models/qimg-21'))
        for backend,device in [('vulkan',0),('rocm',-1),('rocm',True)]:
            with self.assertRaises(ValueError): gpu.configure(backend,device)

    def test_amd_device_lock_is_shared(self):
        with tempfile.TemporaryDirectory(dir=gpu.ROOT/'tmp') as td:
            default=Path(td)/'cuda-0.lock'
            with patch.object(gpu,'LOCK_PATH',default), patch.object(gpu,'gpu_status',return_value={'backend':'rocm','free_mib':16000}):
                with gpu.execution('rocm',3):
                    with gpu.device_session(1024, lock_path=default): pass
            self.assertTrue((Path(td)/'rocm-3.lock').is_file())
            self.assertFalse(default.exists())

if __name__=='__main__':unittest.main()
