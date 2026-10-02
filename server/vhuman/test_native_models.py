"""Runtime contracts and optional saved-oracle checks, with no ML imports."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
from PIL import Image
from . import native_models as native


class ImageContracts(unittest.TestCase):
    def test_framework_free(self):
        if os.environ.get('VHUMAN_REQUIRE_FRAMEWORK_FREE')=='1':
            for name in ('torch','onnx','onnxruntime'):self.assertIsNone(importlib.util.find_spec(name))
        from .realtime.src.animation.causal import MotionAdapter
        self.assertTrue(callable(MotionAdapter))

    def test_invalid_inputs(self):
        for x in (np.zeros((2,4,4)),np.zeros((3,0,4)),np.full((3,4,4),np.nan)):
            with self.assertRaises(ValueError):native.run_image_model('rmbg','missing',x)
        with self.assertRaises(ValueError):native.run_image_model('unknown','missing',np.zeros((3,4,4)))
        with self.assertRaises(RuntimeError):native.run_image_model('rmbg','missing',np.zeros((3,4,4)),runner='missing-native-runner')

    def test_rmbg_pixels_and_mask_contract(self):
        rgb=np.arange(19*27*3,dtype=np.uint8).reshape(19,27,3);image=Image.fromarray(rgb)
        with mock.patch.object(native,'run_image_model',return_value=np.zeros((1,1024,1024),np.float32)) as run:
            mask=native.rmbg_alpha(image,'weights',backend='cpu')
        self.assertEqual(mask.size,image.size);self.assertTrue((np.asarray(mask)==127).all())
        args=run.call_args.args;self.assertEqual(args[0],'rmbg')
        self.assertEqual(args[2].shape,(3,1024,1024));self.assertEqual(args[2].dtype,np.float32)
        np.testing.assert_array_equal(np.asarray(image),rgb)

    def test_camera_fit(self):
        h,w=79,93;focal,shift=2.3,.6;diagonal=np.hypot(h,w)
        u,v=np.meshgrid((2*np.arange(w)+1-w)/diagonal,(2*np.arange(h)+1-h)/diagonal)
        z=1+.4*np.sin(u*3)+.3*np.cos(v*4)
        points=np.stack((u*(z+shift)/focal,v*(z+shift)/focal,z),-1).astype(np.float32)
        fitted=native.recover_camera(points,np.ones((h,w),np.float32))
        self.assertAlmostEqual(fitted['focal'],focal,places=4);self.assertAlmostEqual(fitted['shift'],shift,places=4)
        fallback=native.recover_camera(points,np.zeros((h,w),np.float32))
        self.assertEqual(fallback['focal'],1);self.assertEqual(fallback['shift'],0)

    def test_moge_manifest_integrity(self):
        (native.ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=native.ROOT/'tmp') as d:
            p=Path(d);(p/'native.json').write_text(json.dumps({'format':'vhuman.moge2_camera.v1','files':{'dinov2.safetensors':'bad'}}))
            (p/'dinov2.safetensors').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'checksum'):
                native.moge_camera(Image.new('RGB',(32,32)),p)

    def test_native_bundle_readiness_and_qimg_camera_launcher(self):
        sys.path.insert(0, str(native.ROOT / 'cuda/qimg21'))
        from qimg21_i23d import reconstruct
        (native.ROOT / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=native.ROOT / 'tmp') as d:
            root = Path(d)
            bundle = root / 'native'
            bundle.mkdir()
            (root / 'model.pt').write_bytes(b'offline checkpoint')
            self.assertFalse(native.moge_ready(root / 'model.pt'))
            (bundle / 'native.json').write_text(json.dumps({'format': 'vhuman.moge2_camera.v1'}))
            for name in ('dinov2.safetensors', 'heads.safetensors'):
                (bundle / name).write_bytes(b'stub; no inference')
            self.assertTrue(native.moge_ready(root / 'model.pt'))
            self.assertTrue(native.moge_ready(bundle))
            cancelled = object()
            def run(command, log, timeout, cancel):
                self.assertEqual(command[:2], [sys.executable, str(reconstruct.PIXAL3D / 'prepare_input.py')])
                self.assertEqual(command[command.index('--device') + 1], 'cpu')
                self.assertIs(cancel, cancelled)
                Path(command[command.index('--metadata') + 1]).write_text('{"fov": 0.7}')
            with mock.patch.object(reconstruct, '_run', side_effect=run):
                self.assertEqual(reconstruct.estimate_camera(root / 'image.png', root,
                                 backend='rocm', moge=bundle, cancel=cancelled)['fov'], .7)

    def test_qimg_native_matting_preserves_pixels_and_device(self):
        sys.path.insert(0, str(native.ROOT / 'cuda/qimg21'))
        from qimg21_i23d import rmbg
        pixels = np.full((13, 17, 4), 200, np.uint8)
        with mock.patch.object(native, 'rmbg_alpha', return_value=Image.new('L', (17, 13), 42)) as run:
            result = rmbg.remove_background(pixels, 'cuda:2')
        self.assertEqual(result.shape, (13, 17))
        self.assertTrue((result == 42).all())
        self.assertTrue((pixels == 200).all())
        self.assertEqual(run.call_args.kwargs, {'backend': 'cuda', 'device': 2})


@unittest.skipUnless(os.environ.get('VHUMAN_NATIVE_FIXTURES'),'set VHUMAN_NATIVE_FIXTURES for real native replay')
class NativeReplay(unittest.TestCase):
    def test_motion_stream(self):
        from .realtime.src.animation.causal import MotionAdapter
        root=Path(os.environ['VHUMAN_NATIVE_FIXTURES']);bundle=root/'motion.native'
        spec=json.loads((bundle/'native.json').read_text());revision=spec['tts_revision']
        with self.assertRaises(ValueError):MotionAdapter(bundle,revision)
        adapter=MotionAdapter(bundle,revision,allow_diagnostic=True);self.addCleanup(adapter.close)
        with np.load(root/'small-models/motion.npz') as f:h,c,ref=f['hidden'],f['codes'],f['reference']
        def frame(i,epoch=0,start=None):
            return SimpleNamespace(epoch=epoch,sample_start=i*1920 if start is None else start,
                                   hidden=h[i],codes=c[i],model_revision=revision)
        first=None
        for i in range(len(h)):
            values=np.stack([v.controls for v in adapter.push(frame(i))])
            np.testing.assert_allclose(values,ref[i],atol=2e-5,rtol=0)
            if first is None:first=values.copy()
        adapter.reset(2);self.assertEqual(adapter.push(frame(0,1)),[])
        invalid=frame(0,2);invalid.codes=np.full(16,2048,np.int32)
        with self.assertRaises(ValueError):adapter.push(invalid)
        np.testing.assert_array_equal(np.stack([v.controls for v in adapter.push(frame(0,2))]),first)
        with self.assertRaises(ValueError):adapter.push(frame(2,2))
        adapter.close()
        with self.assertRaises(RuntimeError):adapter.push(frame(1,2))

    def test_cue_replay(self):
        root=Path(os.environ['VHUMAN_NATIVE_FIXTURES'])
        for shape in ('64-64','57-83'):
            with np.load(root/f'small-models/cues-{shape}.npz') as f:inputs,ref=f['input'],f['reference']
            got=native.run_image_model('cues',root/'cues/normal_cue.safetensors',inputs)
            np.testing.assert_allclose(got,ref,atol=2e-5,rtol=0)

    def test_cue_quality_gate_preserved(self):
        from .reconstruction.learned_cues import infer
        root=Path(os.environ['VHUMAN_NATIVE_FIXTURES'])
        with self.assertRaisesRegex(ValueError,'synthetic gate'):
            infer('unused.png',root/'cues',root/'unused.npz')


if __name__=='__main__':unittest.main()
