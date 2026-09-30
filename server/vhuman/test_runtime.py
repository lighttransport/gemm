"""Backend selection and subprocess propagation without loading model weights."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import gpu, runtime

class RuntimeTest(unittest.TestCase):
    def test_rocm_interpreter_defaults_and_explicit_override(self):
        from argparse import Namespace
        args = Namespace(inference_backend='rocm', device=1, models_root='/mnt/disk1/models',
                         qwen_python=None, rig_python='/explicit/python')
        with gpu.execution('cpu'), patch.object(Path, 'exists', return_value=True):
            runtime.configure_args(args)
            self.assertEqual(args.qwen_python, str(gpu.ROOT / 'tmp/vhuman-rocm-venv/bin/python'))
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
        with gpu.execution('cpu', 0, '/mnt/disk1/models'):
            self.assertIsNone(gpu.gpu_status())
            self.assertEqual(gpu.model_path('qimg-21'),Path('/mnt/disk1/models/qimg-21').resolve())
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
