"""DA2 native adapter contracts and optional real-weight parity."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
from PIL import Image

from . import gpu
from .reconstruction import depth
from .service import ROOT


class NativeDepthTest(unittest.TestCase):
    def test_preprocessing_layout_and_size_guards(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV preprocessing dependency absent")
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            path=Path(folder)/"image.png"
            Image.new("RGB",(73,101),(128,80,30)).save(path)
            chw,h,w=depth._preprocess(path,56)
            self.assertEqual((h,w),(101,73));self.assertEqual(chw.shape,(3,84,56))
            self.assertEqual(chw.dtype,np.float32);self.assertTrue(chw.flags.c_contiguous)
            expected=(np.array([128,80,30])/255-[.485,.456,.406])/[.229,.224,.225]
            np.testing.assert_allclose(chw[:,0,0],expected,atol=1e-6)
            with self.assertRaises(ValueError):depth._preprocess(path,0)
            Image.new("RGB",(1,1000)).save(path)
            with self.assertRaisesRegex(ValueError,"4096-patch"):depth._preprocess(path)

    def test_missing_native_export_never_loads_pth(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            root=Path(folder)
            (root/"installation.json").write_text(json.dumps({"model":"Depth-Anything-V2-Small"}))
            (root/"depth_anything_v2_vits.pth").write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(ValueError,"offline export"):
                depth._native_manifest(root)

    def test_checksum_and_model_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            root=Path(folder); (root/"native").mkdir()
            source={"model":"Depth-Anything-V2-Small"}
            (root/"installation.json").write_text(json.dumps(source))
            record={"version":1,"model":source["model"],"source":source,"dtype":"F32","files":{}}
            (root/"native/native.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,"checksum"):
                depth._native_manifest(root)
            record["model"]="Depth-Anything-V2-Large"
            (root/"native/native.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,"unverified Small"):
                depth._native_manifest(root)

    def test_runner_rejects_bad_dimensions_and_nan(self):
        subprocess.run(["make","-s","-C",str(ROOT/"cpu/da2")],check=True,capture_output=True)
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            root=Path(folder)
            np.full((3,14,14),np.nan,np.float32).tofile(root/"input.f32")
            cmd=[str(ROOT/"cpu/da2/da2_depth"),"--backbone","missing","--head","missing",
                 "--input",str(root/"input.f32"),"--output",str(root/"out.f32"),
                 "--width","14","--height","14","--output-width","1","--output-height","1"]
            proc=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(proc.returncode,2);self.assertIn("finite F32",proc.stderr)
            cmd[cmd.index("--width")+1]="15"
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,2)
            self.assertFalse((root/"out.f32").exists())


@unittest.skipUnless(os.environ.get("DA2_TEST_INSTALLATION") and os.environ.get("DA2_TEST_FIXTURE"),
                     "set native installation and exported oracle fixture")
class NativeDepthModelTest(unittest.TestCase):
    def test_framework_free_adapter_and_alignment(self):
        fixture=Path(os.environ["DA2_TEST_FIXTURE"])
        spec=json.loads((fixture/"fixture.json").read_text())
        if os.environ.get("DA2_REQUIRE_FRAMEWORK_FREE"):
            for name in ("torch","onnx","onnxruntime"):
                self.assertIsNone(importlib.util.find_spec(name))
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder, gpu.execution("cpu"):
            out=Path(folder)/"depth.npy"
            report=depth.infer(spec["image"],os.environ["DA2_TEST_INSTALLATION"],out,input_size=spec["input_size"])
            got=np.load(out)
        ref=np.fromfile(fixture/"reference.f32",dtype="<f4").reshape(got.shape)
        delta=np.abs(got-ref)
        self.assertLess(float(delta.max()),2e-4);self.assertLess(float(delta.mean()),2e-5)
        self.assertEqual(report["runner"]["execution"],"native_cpu")
        # Downstream metric alignment must still accept the same cue.
        r=ref[::8,::8].astype(np.float64);g=got[::8,::8].astype(np.float64)
        metric=1/(2+.2*r)
        aligned,_=depth.align(g,metric,np.ones_like(metric))
        np.testing.assert_allclose(aligned,metric,atol=1e-5,rtol=0)


if __name__=="__main__":
    unittest.main()
