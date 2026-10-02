"""Native boundary checks; optional real-model cases use MHR_TEST_ASSETS/REFS."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from ..service import ROOT
from . import mhr, job


class NativeBoundaryTest(unittest.TestCase):
    def test_invalid_coefficients_rejected_before_loading(self):
        with patch.object(mhr, "metadata", side_effect=AssertionError("must validate first")):
            for pose in (np.zeros(204), np.zeros((0,204)), np.zeros((1,203)), np.full((1,204),np.nan)):
                with self.subTest(shape=pose.shape), self.assertRaises(ValueError):
                    mhr.decode("missing",pose,np.zeros(45),backend="cpu")
            with self.assertRaises(ValueError):
                mhr.decode("missing",np.zeros((1,204)),np.zeros(44),backend="cpu")
            with self.assertRaises(ValueError):
                mhr.decode("missing",np.zeros((1,204)),np.zeros(45),np.zeros((1,71)),backend="cpu")

    def test_legacy_locator_never_opens_torchscript(self):
        self.assertEqual(mhr.resolve_assets(model=Path("/missing/model/dinov3/assets/mhr_model.pt")),
                         Path("/missing/model/safetensors"))
        with self.assertRaisesRegex(ValueError,"--mhr-assets"):
            mhr.resolve_assets(model="unknown.pt")
        with self.assertRaisesRegex(ValueError,"--rig-only"):
            mhr.metadata(ROOT / "tmp/nonexistent-mhr-assets")

    def test_cli_rejects_bad_arrays(self):
        runner=mhr.binary("cpu")
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            root=Path(folder)
            np.save(root/"shape.npy",np.zeros((1,45),np.float32))
            for pose in (np.zeros((1,203),np.float32), np.zeros((1,204),np.float64),
                         np.full((1,204),np.inf,np.float32), np.zeros((204,),np.float32),
                         np.asfortranarray(np.zeros((2,204),np.float32))):
                np.save(root/"params.npy",pose)
                proc=subprocess.run([str(runner),"--params",str(root/"params.npy"),
                                     "--shape",str(root/"shape.npy"),"--mhr-assets",str(root),
                                     "--output-dir",str(root)],capture_output=True,text=True)
                self.assertNotEqual(proc.returncode,0)
                self.assertIn("float32 matrices",proc.stderr)

    def test_cancellation_terminates_native_descendants(self):
        # Parent and child share job._run's process group, like Python + native MHR.
        import sys
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            marker=Path(folder)/"child.pid"
            code=("import subprocess,sys,time; "
                  "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                  "open(sys.argv[1],'w').write(str(p.pid)); p.wait()")
            cancel=threading.Event()
            def stop():
                deadline=time.monotonic()+10
                while not marker.exists() and time.monotonic()<deadline:
                    time.sleep(.02)
                cancel.set()
            watcher=threading.Thread(target=stop)
            watcher.start()
            with self.assertRaises(job.gpu.Cancelled):
                job._run([sys.executable,"-c",code,str(marker)],cancel,timeout=15)
            watcher.join()
            pid=int(marker.read_text())
            for _ in range(100):
                state=Path(f"/proc/{pid}/stat")
                if not state.exists() or state.read_text().split()[2]=="Z":
                    break
                time.sleep(.02)
            else:
                self.fail("native descendant survived cancellation")


@unittest.skipUnless(os.environ.get("MHR_TEST_ASSETS") and os.environ.get("MHR_TEST_REFS"),
                     "set exported native assets and offline oracle directory")
class NativeModelTest(unittest.TestCase):
    def test_native_rejects_out_of_range_skin_index(self):
        # Sparse fixture: copy headers and index buffers only, never 661 MiB of weights.
        assets=Path(os.environ["MHR_TEST_ASSETS"])
        refs=Path(os.environ["MHR_TEST_REFS"])
        source=assets/(mhr.STEM+".safetensors")
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            root=Path(folder)
            with source.open("rb") as src, (root/source.name).open("w+b") as dst:
                raw=src.read(8); size,=struct.unpack("<Q",raw)
                header=src.read(size); specs=json.loads(header)
                dst.write(raw+header); dst.truncate(source.stat().st_size)
                for key in ("skeleton.pmi","skeleton.joint_parents","pose_correctives.sparse_indices",
                            "lbs.skin_indices_flattened","lbs.vert_indices_flattened"):
                    start,end=specs[key]["data_offsets"]
                    src.seek(8+size+start); data=src.read(end-start)
                    if key=="lbs.skin_indices_flattened":
                        data=struct.pack("<i",127)+data[4:]
                    dst.seek(8+size+start); dst.write(data)
            (root/(mhr.STEM+".json")).write_bytes((assets/(mhr.STEM+".json")).read_bytes())
            proc=subprocess.run([str(mhr.binary("cpu")),"--params",str(refs/"params.npy"),
                                 "--shape",str(refs/"shape.npy"),"--mhr-assets",str(root),
                                 "--output-dir",str(root)],capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)
            self.assertIn("Invalid MHR assets/shapes/indices",proc.stderr)

    def test_native_coefficients_and_metadata(self):
        assets=Path(os.environ["MHR_TEST_ASSETS"])
        refs=Path(os.environ["MHR_TEST_REFS"])
        params=np.load(refs/"params.npy"); shape=np.load(refs/"shape.npy"); face=np.load(refs/"face.npy")
        vertices,state,rig,report=mhr.decode(assets,params,shape,face,backend="cpu")
        self.assertIn(report["math"],("avx2_gemm","portable_c"))
        np.testing.assert_allclose(vertices,np.load(refs/"reference_vertices.npy"),atol=5e-3,rtol=0)
        np.testing.assert_allclose(state,np.load(refs/"reference_skeleton.npy"),atol=1e-3,rtol=0)
        self.assertEqual(len(rig["names"]),127)
        _,skeleton,_,report=mhr.decode(assets,params,shape,skeleton_only=True,backend="rocm")
        np.testing.assert_array_equal(skeleton,state)
        self.assertEqual(report["backend"],"cpu")
        zero,_,_,_=mhr.decode(assets,params,shape,backend="cpu")
        np.testing.assert_allclose(zero,np.load(refs/"reference_zero_face_vertices.npy"),atol=5e-3,rtol=0)
        self.assertGreater(float(np.max(np.abs(zero[2:]-vertices[2:]))),.01)

    def test_runtime_environment_has_no_frameworks(self):
        if os.environ.get("MHR_REQUIRE_FRAMEWORK_FREE"):
            for name in ("torch","onnx","onnxruntime"):
                self.assertIsNone(importlib.util.find_spec(name))


if __name__ == "__main__":
    unittest.main()
