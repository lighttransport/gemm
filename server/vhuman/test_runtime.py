"""Backend selection and subprocess propagation without loading model weights."""
import tempfile
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace
from . import gpu, runtime

class RuntimeTest(unittest.TestCase):
    def test_face_assets_follow_models_root_across_contexts_and_subprocess(self):
        from .face_assets import asset_path, data_root
        from .rig import face_models
        from .reconstruction.upload import server_request
        import subprocess
        import json
        import sys
        with tempfile.TemporaryDirectory(dir=gpu.ROOT/'tmp') as td, \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop('VHUMAN_DATA_ROOT', None)
            for name in ('first', 'second'):
                root = Path(td)/name
                with gpu.execution('cpu', models_root=str(root)):
                    expected = root/'vhuman-rig-models'
                    self.assertEqual(data_root(), expected)
                    for asset in ('gnm', 'mediapipe', 'parsing'):
                        self.assertTrue(asset_path(asset).is_relative_to(expected))
                    with patch.object(face_models, '_gnm_path', side_effect=RuntimeError('stop before loading')) as lookup:
                        with self.assertRaisesRegex(RuntimeError, 'stop before loading'):
                            face_models.load('gnm_v3')
                        lookup.assert_called_once_with(expected)
                    req = server_request(None, {'depth': True})
                    self.assertEqual(req['depth_installation'], str(expected/'depth-anything-v2-small'))
            with gpu.execution('cpu', models_root=str(root)):
                probe = Path(td)/'probe.py'
                probe.write_text('import json\nfrom server.vhuman.face_assets import asset_path\n'
                                 'print(json.dumps(str(asset_path("gnm"))))\n')
                command = runtime.python_command([sys.executable, '-m', 'server.vhuman.face_assets'])
                command[command.index('--module')+1] = f'tmp.{Path(td).name}.probe'
                result = subprocess.run(command, cwd=gpu.ROOT, check=True, capture_output=True, text=True)
                self.assertEqual(Path(json.loads(result.stdout)), expected/'gnm-v3/gnm_head.npz')
            with patch.dict(os.environ, VHUMAN_DATA_ROOT=str(Path(td)/'explicit-face-root')):
                self.assertEqual(asset_path('gnm').parent.parent, Path(td)/'explicit-face-root')

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
