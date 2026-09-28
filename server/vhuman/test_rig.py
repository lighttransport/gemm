"""Facial rig: evaluator, chart/loop geometry, file guards (numpy only), and an
end-to-end mock build in the rig interpreter (scipy + torch; skipped when
tmp/vhuman-rig-venv or VHUMAN_RIG_PYTHON is unavailable)."""
import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .rig import rigdef, template as T
from .rig.usd import _f

ROOT = Path(__file__).resolve().parents[2]


def _toy_rig():
    joints = [{"name": "root", "parent": None, "bind": np.eye(4).tolist(), "rest_translation": [0, 0, 0],
               "rest_rotation": np.eye(3).tolist()},
              {"name": "jaw", "parent": "root", "bind": np.eye(4).tolist(), "rest_translation": [0, 0, 0],
               "rest_rotation": np.eye(3).tolist()}]
    return {"controls": rigdef.control_table(), "joints": joints,
            "correctives": [{"name": "c", "inputs": ["jawOpen", "mouthSmileLeft"], "weight": 2.0}],
            "joint_matrix": [{"input": "jawOpen", "joint": "jaw", "attr": "rx", "value": math.radians(20)},
                             {"input": "jawForward", "joint": "jaw", "attr": "tz", "value": 0.004}],
            "blendshapes": [{"name": "mouthSmileLeft", "input": "mouthSmileLeft"}, {"name": "c", "input": "c"}]}


class RigEvaluatorTests(unittest.TestCase):
    def test_controls_cover_lightrig_namespace(self):
        names = [c["name"] for c in rigdef.control_table()]
        self.assertEqual(len(rigdef.LR_FACE_V1), 51)
        self.assertEqual(names[:51], list(rigdef.LR_FACE_V1))
        self.assertEqual(len(set(names)), len(names))

    def test_joint_deltas_and_correctives(self):
        r = rigdef.Rig(_toy_rig())
        ev = r.evaluate({"jawOpen": 1.0, "jawForward": 0.5, "mouthSmileLeft": 0.25})
        jaw = ev["local"][1]
        np.testing.assert_allclose(jaw[:3, :3], rigdef.euler_matrix(math.radians(20), 0, 0), atol=1e-12)
        self.assertAlmostEqual(jaw[2, 3], 0.002)
        # corrective = min(1, 2 * 1.0 * 0.25)
        self.assertAlmostEqual(dict(zip(r.shape_names, ev["weights"]))["c"], 0.5)
        # clamping to the control range
        self.assertAlmostEqual(r.input_vector({"jawOpen": 3.0})[r.cidx["jawOpen"]], 1.0)
        # a point in front of the hinge moves down when the jaw opens
        p = rigdef.deform(np.array([[0, 0, 0.1]]), np.array([[1, 0, 0, 0]]), np.array([[1.0, 0, 0, 0]]),
                          r.evaluate({"jawOpen": 1})["skin"])
        self.assertLess(p[0, 1], -0.02)

    def test_rest_pose_is_identity(self):
        r = rigdef.Rig(_toy_rig())
        rest = np.random.default_rng(0).normal(size=(20, 3))
        J = np.zeros((20, 4), np.int64)
        W = np.zeros((20, 4))
        W[:, 0] = 1
        np.testing.assert_allclose(rigdef.deform(rest, J, W, r.evaluate({})["skin"]), rest, atol=1e-12)

    def test_read_track(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.txt"
            row = [0.0] * 53
            row[1 + 1 + rigdef.LR_FACE_V1.index("jawOpen")] = 0.7
            p.write_text(" ".join(map(str, row)) + "\n" + " ".join(map(str, [0.1] + row[1:])) + "\n")
            times, frames = rigdef.read_track(p)
            self.assertEqual(len(frames), 2)
            self.assertAlmostEqual(frames[0]["jawOpen"], 0.7)
            p.write_text("0 1 2\n")
            with self.assertRaises(ValueError):
                rigdef.read_track(p)

    def test_usd_numbers_have_no_subnormals(self):
        self.assertEqual(_f(1e-223), "0")
        self.assertEqual(_f(-2.5e-3), "-0.0025")


class ChartTests(unittest.TestCase):
    def test_chart_round_trip(self):
        d = np.random.default_rng(1).normal(size=(500, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        d = d[d[:, 2] > -0.99]
        np.testing.assert_allclose(T.from_chart(T.to_chart(d)), d, atol=1e-9)

    def test_closed_mouth_loop_offsets(self):
        # a closed mouth (both lips on the seam) still yields separated rings
        x = np.linspace(-30, 30, T.MOUTH_HALF + 1)
        seam = np.stack([x, -50 + 0.002 * x ** 2], 1)
        loop = T.mouth_loop(seam, seam)
        self.assertEqual(len(loop), T.MOUTH_N)
        ring = T.offset_loop(loop, 3.0, -1.0)
        upper = ring[1:T.MOUTH_HALF]
        lower = ring[T.MOUTH_HALF + 1:]
        self.assertTrue((upper[:, 1] > np.interp(upper[:, 0], seam[:, 0], seam[:, 1])).all())
        self.assertTrue((lower[:, 1] < np.interp(lower[:, 0], seam[:, 0], seam[:, 1])).all())
        # corners stay the ring's samples 0 and N/2, pushed outwards
        self.assertLess(ring[0, 0], seam[0, 0])
        self.assertGreater(ring[T.MOUTH_HALF, 0], seam[-1, 0])


class SafetensorsTests(unittest.TestCase):
    def test_round_trip(self):
        from .rig import safetensors as st
        with tempfile.TemporaryDirectory() as d:
            t = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.array([1, -2], np.int32)}
            st.save(Path(d) / "x.safetensors", t, {"k": "v"})
            back, meta = st.load(Path(d) / "x.safetensors")
            np.testing.assert_array_equal(back["a"], t["a"])
            np.testing.assert_array_equal(back["b"], t["b"])
            self.assertEqual(meta, {"k": "v"})
            raw = (Path(d) / "x.safetensors").read_bytes()
            self.assertEqual((8 + int.from_bytes(raw[:8], "little")) % 8, 0)     # aligned tensor data


class RigFileTests(unittest.TestCase):
    def test_rig_file_guard(self):
        from .service import EyeService, ServiceError
        with tempfile.TemporaryDirectory() as d:
            svc = EyeService(Path(d))
            rig = Path(d) / "heads" / "abc123" / "rig"
            (rig / "textures").mkdir(parents=True)
            (rig / "rig.glb").write_bytes(b"x")
            (rig / "textures" / "skin_baseColorTexture.png").write_bytes(b"x")
            self.assertEqual(svc.rig_file("abc123", "rig.glb").name, "rig.glb")
            self.assertTrue(svc.rig_file("abc123", "textures/skin_baseColorTexture.png").is_file())
            for bad in ("../fit.json", "textures/../../x.png", "fit_cache.pkl", "textures/a/b.png"):
                with self.assertRaises(ServiceError):
                    svc.rig_file("abc123", bad)
            with self.assertRaises(ServiceError):
                svc.rig_file("../x", "rig.glb")


def _rig_python():
    env = os.environ.get("VHUMAN_RIG_PYTHON")
    p = Path(env) if env else ROOT / "tmp/vhuman-rig-venv/bin/python"
    return p if p.exists() else None


@unittest.skipIf(_rig_python() is None, "no rig interpreter (server/vhuman/requirements-rig.txt)")
class RigEndToEndTests(unittest.TestCase):
    """A mock head (synthetic portrait, ellipsoid) through the whole builder."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        out = subprocess.run([str(_rig_python()), "-m", "server.vhuman.rig.selftest", cls.tmp.name], cwd=ROOT,
                             capture_output=True, text=True, timeout=1800)
        if out.returncode != 0:
            raise AssertionError(out.stderr[-3000:])
        cls.summary = json.loads(out.stdout.strip().splitlines()[-1])
        cls.rig_dir = Path(cls.tmp.name) / "head" / "rig"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_template_topology(self):
        t = self.summary["template"]
        self.assertEqual(t["nonmanifold_edges"], 0)
        self.assertEqual(t["duplicate_directed"], 0)       # consistently oriented
        self.assertEqual(t["unused_vertices"], 0)
        self.assertEqual(t["boundary_edges"], t["expected_boundary"])   # lid linings + the neck cut

    def test_outputs(self):
        for name in ("rig.glb", "rig.json", "rig.usda", "rig_usd.zip", "rig_report.json", "rig_basecolor.png",
                     "deformer.lrm", "deformer_basis.safetensors", "rig_deformer.safetensors", "viz.json"):
            self.assertIn(name, self.summary["files"])
        self.assertEqual(self.summary["report"]["controls"], len(rigdef.CONTROLS))
        self.assertGreaterEqual(self.summary["report"]["shapes"], 50)
        for j in ("jaw", "eye_L", "eye_R", "teeth_upper", "teeth_lower", "tongue_04"):
            self.assertIn(j, self.summary["report"]["joints"])
        self.assertLess(self.summary["register"]["final_mean_dist_mm"], 3.0)

    def test_glb_skin_and_morphs(self):
        from .eye.glb import GLB
        g = GLB.load(self.rig_dir / "rig.glb")
        self.assertEqual(len(g.doc["skins"]), 1)
        skin = next(m for m in g.doc["meshes"] if m["name"] == "head_skin")
        prim = skin["primitives"][0]
        self.assertIn("JOINTS_0", prim["attributes"])
        self.assertEqual(len(prim["targets"]), len(skin["extras"]["targetNames"]))
        self.assertIn("eyeBlinkLeft", skin["extras"]["targetNames"])
        # glTF: all targets of a primitive carry the same attributes
        self.assertEqual(len({tuple(sorted(t)) for t in prim["targets"]}), 1)
        w = g.accessor(prim["attributes"]["WEIGHTS_0"])
        np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-5)

    def test_ml_deformer(self):
        d = self.summary["deformer"]
        self.assertEqual(self.summary["ml_targets"], d["components"] + 1)
        self.assertLess(d["val_error_mean_mm"], d["residual_mean_mm"] + 1e-6)
        rig = json.loads((self.rig_dir / "rig.json").read_text())
        r = rigdef.Rig(rig, folder=self.rig_dir)                  # numpy runtime, no torch
        ev = r.evaluate({"jawOpen": 1.0, "mouthSmileLeft": 1.0})
        w = dict(zip(r.shape_names, ev["weights"]))
        self.assertEqual(w["ml_mean"], 1.0)
        self.assertTrue(all(np.isfinite(v) for v in w.values()))

    @unittest.skipIf(shutil.which("gcc") is None, "no gcc")
    def test_native_deformer_parity(self):
        from .rig import native, safetensors as st
        lib = native.build_library(Path(self.tmp.name) / "native")
        N = native.Native(lib, self.rig_dir / "rig_deformer.safetensors")
        N.set_contact_iterations(0)                     # the rig alone; the projection is tested below
        try:
            rig = json.loads((self.rig_dir / "rig.json").read_text())
            r = rigdef.Rig(rig, folder=self.rig_dir)
            pk, meta = st.load(self.rig_dir / "rig_deformer.safetensors")
            names = json.loads(meta["morphs"])
            rng = np.random.default_rng(0)
            X = np.zeros((6, N.C), np.float32)
            X[:, :51] = (rng.random((6, 51)) < 0.15) * rng.random((6, 51))
            X[:, r.cidx["headYaw"]] = rng.uniform(-1, 1, 6)
            for x in X:
                ev = r.evaluate(x)
                w = dict(zip(r.shape_names, ev["weights"]))
                p = pk["rest"] + np.tensordot(np.array([w.get(n, 0.0) for n in names]), pk["morph"], 1)
                ref = rigdef.deform(p.astype(np.float64), pk["skin.joints"].astype(int),
                                    pk["skin.weights"].astype(np.float64), ev["skin"])
                np.testing.assert_allclose(N.eval(x), ref, atol=2e-6)
            B = N.eval_batch(X)
            np.testing.assert_allclose(B[2], N.eval(X[2]), atol=2e-6)
        finally:
            N.close()

    @unittest.skipIf(shutil.which("gcc") is None, "no gcc")
    def test_contact_projection(self):
        """Exact contacts: C equals the numpy reference; nothing penetrates after it."""
        from .rig import contacts, native, safetensors as st
        lib = native.build_library(Path(self.tmp.name) / "native_ct")
        N = native.Native(lib, self.rig_dir / "rig_deformer.safetensors")
        try:
            self.assertTrue(N.has_contacts)
            rig = rigdef.Rig(json.loads((self.rig_dir / "rig.json").read_text()), folder=self.rig_dir)
            t, _ = st.load(self.rig_dir / "rig_deformer.safetensors")
            rng = np.random.default_rng(3)
            X = np.zeros((12, N.C), np.float32)
            X[:, :51] = (rng.random((12, 51)) < 0.2) * rng.random((12, 51))
            X[::2, rig.cidx["jawOpen"]] = 0.7
            X[::3, rig.cidx["tongueOut"]] = 1.0
            for x in X:
                N.set_contact_iterations(0)
                raw = N.eval(x)
                N.set_contact_iterations(contacts.ITERATIONS)
                got = N.eval(x)
                skin = rig.evaluate(x)["skin"]
                np.testing.assert_allclose(got, contacts.project(raw, skin, t), atol=1e-4)
                self.assertEqual(contacts.depths(got, skin, t), {"eye": 0, "spheres": 0, "lips": 0})
                untouched = np.setdiff1d(np.arange(N.V), t["contact.verts"])
                np.testing.assert_array_equal(got[untouched], raw[untouched])
        finally:
            N.close()

    @unittest.skipIf(shutil.which("gcc") is None, "no gcc")
    def test_gpu_deformer_parity(self):
        from .rig import native
        try:
            lib = native.build_gpu_library(Path(self.tmp.name) / "gpu")
            G = native.NativeGPU(lib, self.rig_dir / "rig_deformer.safetensors")
        except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
            self.skipTest(f"CUDA unavailable: {exc}")
        try:
            rng = np.random.default_rng(1)
            X = np.zeros((19, G.C), np.float32)                  # not a multiple of the 8-frame tile
            X[:, :51] = (rng.random((19, 51)) < 0.15) * rng.random((19, 51))
            out, ms = G.eval_gpu(X)
            for f in (0, 7, 8, 18):
                np.testing.assert_allclose(out[f], G.eval(X[f]), atol=2e-6)
            self.assertGreaterEqual(ms[2], 0.0)
        finally:
            G.close()

    @unittest.skipIf(shutil.which("glslc") is None or shutil.which("g++") is None, "no glslc/g++")
    def test_vulkan_deformer_parity(self):
        from .rig import native
        try:
            lib = native.build_vk_library(Path(self.tmp.name) / "vk")
            G = native.NativeVK(lib, self.rig_dir / "rig_deformer.safetensors")
        except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
            self.skipTest(f"Vulkan unavailable: {exc}")
        try:
            rng = np.random.default_rng(2)
            X = np.zeros((13, G.C), np.float32)
            X[:, :51] = (rng.random((13, 51)) < 0.15) * rng.random((13, 51))
            out, _ = G.eval_gpu(X)
            for f in (0, 8, 12):
                np.testing.assert_allclose(out[f], G.eval(X[f]), atol=2e-6)
        finally:
            G.close()

    def test_viz_export(self):
        viz = json.loads((self.rig_dir / "viz.json").read_text())
        self.assertEqual(set(viz["parts"]), {"head_skin", "head_mouth"})
        c = viz["contacts"]
        self.assertEqual(set(c["spheres"]), {"teeth", "tongue"})
        self.assertEqual(len(c["nbr_ptr"]), len(c["verts"]) + 1)
        self.assertEqual(len(c["eye"]), 2)
        self.assertEqual(len(c["pairs_upper"]), len(c["pairs_lower"]))
        vmax = max(max(p["vmap"]) for p in viz["parts"].values())
        self.assertLess(vmax, viz["welded_vertices"])

    def test_usd_layer(self):
        text = (self.rig_dir / "rig.usda").read_text()
        self.assertTrue(text.startswith("#usda 1.0"))
        for token in ('def SkelRoot "Character"', 'def Skeleton "Skeleton"', 'def SkelAnimation "ROM"',
                      "primvars:skel:jointIndices", 'def BlendShape "eyeBlinkLeft"', "dictionary vchar"):
            self.assertIn(token, text)
        self.assertNotRegex(text, r"e-[0-9]{3}")          # no subnormals
        try:
            import lightusd
        except ImportError:
            return
        lightusd.load(str(self.rig_dir / "rig.usda"))


if __name__ == "__main__":
    unittest.main()
