"""Native landmark model integration, ownership and dependency boundaries."""
import builtins
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .native_landmarks import Graph, FaceLandmarker, ROOT, detections
from .landmark_assets import export


class NativeLandmarkTests(unittest.TestCase):
    def test_empty_detections_and_weighted_suppression(self):
        raw=np.zeros(896*17,np.float32);raw[896*16:]=-100
        self.assertEqual(detections(raw),[])
        boxes=raw[:896*16].reshape(896,16)
        boxes[:2,2:4]=64
        raw[896*16:896*16+2]=[2,3]
        got=detections(raw)
        self.assertEqual(len(got),1)
        np.testing.assert_allclose(got[0][0][:4],[.5/16,.5/16,.5,.5])

    @unittest.skipUnless((ROOT/'cpu/vhuman/libvhuman_landmarks.so').is_file(),'native build required')
    def test_truncated_graph_rejected_and_input_size_checked(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            path=Path(tmp)/'bad.bin';path.write_bytes(b'VHFACE1\0'+struct.pack('<4i',1,1,1,1))
            with self.assertRaisesRegex(RuntimeError,'truncated'):
                Graph(path)

    @unittest.skipUnless((ROOT/'tmp/vhuman-rig/models/face_landmarker.task').is_file() and
                         (ROOT/'cpu/vhuman/libvhuman_landmarks.so').is_file(),'pinned model/native build required')
    def test_observation_and_video_paths_without_inference_runtime(self):
        from PIL import Image
        from .reconstruction.observations import observe
        from .rig.video_fit import _observe
        task=ROOT/'tmp/vhuman-rig/models/face_landmarker.task'
        portrait=ROOT/'tmp/vhuman-realtime/clean-identity-001/neutral.png'
        if not portrait.is_file():self.skipTest('local portrait fixture absent')
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','onnx','onnxruntime','mediapipe','tensorflow','tflite_runtime','ai_edge_litert'):
                raise AssertionError('unexpected inference runtime '+name)
            return original(name,*args,**kwargs)
        with patch.object(builtins,'__import__',side_effect=guarded):
            result=observe(portrait,task=task)
            self.assertEqual(result['tracker']['backend'],'native_gemm')
            controls,landmarks,valid=_observe([portrait]*3,task)
            self.assertTrue(valid.all());self.assertTrue(np.isfinite(controls).all())
            np.testing.assert_array_equal(landmarks[0],landmarks[2])
            with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
                assets=Path(tmp);receipt=export(task,assets)
                self.assertEqual(len(receipt['models']),3)
                with FaceLandmarker(task,assets=assets) as model:
                    self.assertEqual(model.detect(np.zeros((256,384,3),np.uint8)),[])
                    with self.assertRaises(ValueError):model.detect(np.zeros((10,10),np.uint8))
                    graph=model.graphs[0]
                    with self.assertRaises(ValueError):graph(np.zeros(1))
                with self.assertRaisesRegex(RuntimeError,'closed'):model.detect(np.zeros((10,10,3),np.uint8))
                (assets/'face_detector.bin').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checksum'):FaceLandmarker(task,assets=assets)


if __name__=='__main__':unittest.main()
