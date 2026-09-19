import argparse
import base64
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import time
import unittest
from unittest import mock

from server.pixal3d import app


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0],
            [0, 0, 1, 0], [0, 0, 0, 1]]


class PixalServerTest(unittest.TestCase):
    def make_server(self, work_dir: Path) -> app.PixalServer:
        return app.PixalServer(argparse.Namespace(
            binary="cpu/pixal3d/pixal3d", model_dir="models/pixal3d",
            dinov3="models/dinov3.safetensors", naf="models/naf.safetensors",
            work_dir=str(work_dir), backend="cuda", gpu_execution="resident",
            gpu_kernels="auto", gpu_flow_precision="mixed", threads=0,
            timeout=30, reference_timeout=30))

    def test_multiview_manifest_and_command(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        captured = {}
        with tempfile.TemporaryDirectory(prefix="web-", dir=scratch) as td:
            server = self.make_server(Path(td))

            def native_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                if str(server.prepare_script) in command:
                    output.write_bytes(b"prepared-rgba")
                    Path(command[command.index("--metadata") + 1]).write_text(json.dumps(
                        {"fov": 0.8, "distance": 1.2, "mesh_scale": 1.25,
                         "mask_source": "alpha", "camera_source": "manual"}))
                    return subprocess.CompletedProcess(command, 0, "{}\n", "")
                ply_output = Path(command[command.index("--ply-output") + 1])
                views_dir = Path(command[command.index("--views-dir") + 1])
                captured["command"] = command
                captured["manifest"] = json.loads(
                    (views_dir / "transforms.json").read_text())
                output.write_bytes(b"glTF-test")
                ply_output.write_bytes(b"ply-test")
                Path(command[command.index("--profile-json") + 1]).write_text(
                    '{"peak_vram_mib": 1024}')
                return subprocess.CompletedProcess(command, 0, '{"views":2}\n', "")

            encoded = base64.b64encode(b"image-data").decode()
            request = {
                "backend": "cuda", "gpu_execution": "resident",
                "gpu_kernels": "mma", "gpu_flow_precision": "mixed",
                "seed": 7, "device": 0, "vram_budget_mib": 12288,
                "texture_size": 2048, "triangle_target": 500000,
                "include_ply": True,
                "auto_mask": True,
                "mesh_scale": 1.25, "fov": 0.8,
                "views": [
                    {"image_b64": encoded, "transform_matrix": IDENTITY},
                    {"image_b64": encoded, "transform_matrix": IDENTITY,
                     "fov": 0.9},
                ],
            }
            with mock.patch.object(app.subprocess, "run", side_effect=native_run):
                result = server.infer(request)

        self.assertEqual(base64.b64decode(result["glb_b64"]), b"glTF-test")
        self.assertEqual(base64.b64decode(result["ply_b64"]), b"ply-test")
        self.assertEqual(len(result["preparation"]["views"]), 2)
        self.assertEqual(result["preparation"]["views"][1]["view"], 1)
        self.assertEqual(result["stats"], {"views": 2})
        self.assertEqual(result["profile"]["peak_vram_mib"], 1024)
        self.assertIn("--views-dir", captured["command"])
        self.assertNotIn("--input", captured["command"])
        self.assertEqual(captured["command"][captured["command"].index("--texture-size") + 1], "2048")
        self.assertEqual(captured["command"][captured["command"].index("--triangle-target") + 1], "500000")
        self.assertIn("--ply-output", captured["command"])
        self.assertEqual(captured["manifest"]["mesh_scale"], 1.25)
        self.assertEqual(len(captured["manifest"]["frames"]), 2)
        self.assertEqual(captured["manifest"]["frames"][1]["camera_angle_x"], 0.9)

    def test_rejects_invalid_camera_and_integer_controls(self):
        with self.assertRaisesRegex(ValueError, "4x4"):
            app.camera_matrix([[1]], "camera")
        with self.assertRaisesRegex(ValueError, "integer"):
            app.bounded_integer(1.5, "device", 0, 255)
        with self.assertRaisesRegex(ValueError, "finite"):
            app.camera_matrix([[float("nan")] * 4 for _ in range(4)], "camera")
        with self.assertRaisesRegex(ValueError, "boolean"):
            app.boolean("yes", "auto_camera")
        resolved = app.reference_request(
            {"fov": 0.8, "auto_camera": True},
            {"preparation": {"fov": 0.6, "distance": 1.5}})
        self.assertEqual((resolved["fov"], resolved["distance"], resolved["auto_camera"]),
                         (0.6, 1.5, False))
        self.assertFalse(app.rmbg_ready(Path("/missing/rmbg")))
        scratch_file = app.ROOT / "tmp/pixal3d/tests/empty-output"
        scratch_file.parent.mkdir(parents=True, exist_ok=True)
        scratch_file.write_bytes(b"")
        self.assertFalse(app.valid_output(scratch_file))
        scratch_file.write_bytes(b"mesh")
        self.assertTrue(app.valid_output(scratch_file))
        scratch_file.unlink()
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="web-", dir=scratch) as td:
            server = self.make_server(Path(td))
            encoded = base64.b64encode(b"image-data").decode()
            with self.assertRaisesRegex(ValueError, "include_ply"):
                server.infer({"image_b64": encoded, "include_ply": "yes"})

    def test_glb_mesh_comparison_summary(self):
        scratch = app.ROOT / "tmp/pixal3d/tests/summary.glb"
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scene = {"meshes": [{"primitives": [{"attributes": {"POSITION": 0},
                                                "indices": 1}]}],
                 "accessors": [{"count": 12, "min": [-0.5, -0.4, -0.3],
                                 "max": [0.5, 0.4, 0.3]},
                                {"count": 30}]}
        encoded = json.dumps(scene, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 4)
        raw = (struct.pack("<III", 0x46546c67, 2, 20 + len(encoded)) +
               struct.pack("<II", len(encoded), 0x4e4f534a) + encoded)
        scratch.write_bytes(raw)
        summary = app.glb_mesh_summary(scratch)
        scratch.unlink()
        self.assertEqual((summary["vertices"], summary["triangles"]), (12, 10))
        comparison = app.mesh_comparison(summary, {
            "bytes": 1, "vertices": 10, "triangles": 8,
            "bounds": [[-0.4, -0.4, -0.3], [0.5, 0.4, 0.2]]})
        self.assertAlmostEqual(comparison["vertices_relative_delta"], 0.2)
        self.assertAlmostEqual(comparison["bounds_max_abs_delta"], 0.1)

    def test_automatic_camera_prepares_native_input(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        commands = []
        with tempfile.TemporaryDirectory(prefix="prepare-", dir=scratch) as td:
            server = self.make_server(Path(td))
            server.moge = Path("/mnt/disk2/models/moge-2-vitl/model.pt")

            def run(command, **kwargs):
                commands.append(command)
                output = Path(command[command.index("--output") + 1])
                if str(server.prepare_script) in command:
                    output.write_bytes(b"prepared-rgba")
                    metadata = Path(command[command.index("--metadata") + 1])
                    metadata.write_text(json.dumps({"fov": 0.6, "distance": 1.5,
                                                    "mesh_scale": 1.0,
                                                    "mask_source": "alpha",
                                                    "camera_source": "moge-2"}))
                    return subprocess.CompletedProcess(command, 0, "{}\n", "")
                output.write_bytes(b"glTF-auto")
                Path(command[command.index("--profile-json") + 1]).write_text("{}")
                return subprocess.CompletedProcess(command, 0, "{}\n", "")

            request = {"backend": "cuda", "image_b64": base64.b64encode(b"rgba").decode(),
                       "auto_camera": True}
            with mock.patch.object(app.subprocess, "run", side_effect=run):
                result = server.infer(request)

        self.assertEqual(result["preparation"]["camera_source"], "moge-2")
        native = commands[1]
        self.assertEqual(native[native.index("--input") + 1].split("/")[-1], "prepared.png")
        self.assertEqual(native[native.index("--fov") + 1], "0.6")
        self.assertEqual(native[native.index("--distance") + 1], "1.5")

    def test_multiview_reference_uses_pinned_entry_point(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        captured = {}
        with tempfile.TemporaryDirectory(prefix="reference-", dir=scratch) as td:
            server = self.make_server(Path(td))

            def reference_run(command, **kwargs):
                views_dir = Path(command[command.index("--views_dir") + 1])
                output = Path(command[command.index("--output") + 1])
                captured["command"] = command
                captured["manifest"] = json.loads(
                    (views_dir / "transforms.json").read_text())
                output.write_bytes(b"glTF-reference")
                return subprocess.CompletedProcess(command, 0, "reference complete\n", "")

            encoded = base64.b64encode(b"rgba-image").decode()
            request = {
                "backend": "cuda", "seed": 9, "fov": 0.75,
                "mesh_scale": 1.5,
                "views": [{"image_b64": encoded,
                           "transform_matrix": IDENTITY, "fov": 0.85}],
            }
            with mock.patch.object(app.subprocess, "run", side_effect=reference_run):
                result = server.reference(request)

        self.assertEqual(base64.b64decode(result["glb_b64"]), b"glTF-reference")
        self.assertEqual(Path(captured["command"][2]), server.reference_mv_script)
        self.assertIn("--low_vram", captured["command"])
        self.assertEqual(captured["manifest"]["mesh_scale"], 1.5)
        self.assertEqual(captured["manifest"]["frames"][0]["camera_angle_x"], 0.85)

    def test_job_queue_reports_phase_and_result(self):
        class FakePixal:
            def infer(self, request, cancel=None, progress=None):
                progress("Pixal3D shape1024: step 6/12")
                return {"ok": True, "value": request["value"]}
            def reference(self, request, cancel=None):
                return {"value": "reference"}

        jobs = app.JobQueue(FakePixal(), retained=2)
        submitted = jobs.submit({"value": 7, "reference": True})
        self.assertIn(submitted["state"], ("queued", "running", "complete"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = jobs.status(submitted["id"], include_result=True)
            if status["state"] == "complete":
                break
            time.sleep(0.01)
        self.assertEqual(status["phase"], "complete")
        self.assertEqual(status["progress"], 100)
        self.assertEqual(status["result"]["value"], 7)
        self.assertEqual(status["result"]["reference"]["value"], "reference")

    def test_job_failure_has_stable_error_code(self):
        class InvalidPixal:
            def infer(self, request, cancel=None, progress=None):
                raise ValueError("bad camera")
        jobs = app.JobQueue(InvalidPixal(), retained=1)
        job = jobs.submit({})
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = jobs.status(job["id"])
            if status["state"] == "failed":
                break
            time.sleep(0.01)
        self.assertEqual(status["error_code"], "invalid_request")
        self.assertEqual(app.error_payload("not_found", "missing")["error_code"], "not_found")

    def test_queued_job_can_be_cancelled(self):
        gate = __import__("threading").Event()
        class BlockingPixal:
            def infer(self, request, cancel=None, progress=None):
                gate.wait(2)
                return {"ok": True}
            def reference(self, request, cancel=None):
                raise AssertionError("reference should not run")

        jobs = app.JobQueue(BlockingPixal(), retained=2)
        first = jobs.submit({})
        deadline = time.monotonic() + 2
        while jobs.status(first["id"])["state"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.01)
        second = jobs.submit({})
        cancelled = jobs.cancel(second["id"])
        gate.set()
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertEqual(jobs.status(second["id"])["phase"], "cancelled")

    def test_cancellable_command_terminates_child(self):
        cancel = __import__("threading").Event()
        timer = __import__("threading").Timer(0.05, cancel.set)
        timer.start()
        started = time.monotonic()
        with self.assertRaises(app.JobCancelled):
            app.run_command([sys.executable, "-c", "import time; time.sleep(10)"],
                            timeout=20, cancel=cancel)
        timer.cancel()
        self.assertLess(time.monotonic() - started, 2)

    def test_native_progress_streaming(self):
        updates = []
        command = [sys.executable, "-c",
                   "import sys; print('Pixal3D structure: conditioning 512x512, 4096 tokens', file=sys.stderr, flush=True); print('Pixal3D structure: step 6/12', file=sys.stderr, flush=True)"]
        result = app.run_command(command, timeout=5, progress=updates.append)
        self.assertEqual(result.returncode, 0)
        self.assertEqual([app.native_progress(line) for line in updates],
                         [("structure conditioning", 2), ("structure diffusion 6/12", 11)])
        self.assertEqual(app.native_progress("unrelated diagnostic"), None)

    def test_raw_upload_is_consumed_and_cleaned(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="uploads-", dir=scratch) as td:
            uploads = app.UploadStore(Path(td), retained=2)
            upload_id = uploads.put(b"raw-image")

            class UploadPixal:
                def infer(self, request, cancel=None, progress=None):
                    self.path = request["image_b64"]
                    return {"ok": True, "bytes": app.decode_b64(
                        self.path, "image_b64", app.MAX_IMAGE_BYTES).decode()}
                def reference(self, request, cancel=None):
                    raise AssertionError("reference should not run")

            pixal = UploadPixal()
            jobs = app.JobQueue(pixal, retained=1, uploads=uploads)
            job = jobs.submit({"image_upload": upload_id})
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                status = jobs.status(job["id"], include_result=True)
                if status["state"] == "complete":
                    break
                time.sleep(0.01)
            self.assertEqual(status["result"]["bytes"], "raw-image")
            while pixal.path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(pixal.path.exists())
            with self.assertRaisesRegex(ValueError, "unknown or expired"):
                uploads.claim({"image_upload": upload_id})

    def test_upload_expiry_delete_and_startup_cleanup(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="upload-life-", dir=scratch) as td:
            root = Path(td)
            stale = root / "left-by-old-server"
            stale.write_bytes(b"stale")
            uploads = app.UploadStore(root, retained=2, ttl=0.01)
            self.assertFalse(stale.exists())

            first = uploads.put(b"first")
            self.assertTrue(uploads.delete(first))
            self.assertFalse(uploads.delete(first))
            second = uploads.put(b"second")
            time.sleep(0.02)
            with self.assertRaisesRegex(ValueError, "unknown or expired"):
                uploads.claim({"image_upload": second})
            self.assertFalse(any(root.iterdir()))

    def test_terminal_job_expiry_and_explicit_delete(self):
        class ImmediatePixal:
            def infer(self, request, cancel=None, progress=None):
                return {"ok": True}

        jobs = app.JobQueue(ImmediatePixal(), retained=2, ttl=0.02)
        first = jobs.submit({})
        deadline = time.monotonic() + 2
        while jobs.status(first["id"])["state"] != "complete" and time.monotonic() < deadline:
            time.sleep(0.01)
        deleted = jobs.delete(first["id"])
        self.assertTrue(deleted["deleted"])
        with self.assertRaises(KeyError):
            jobs.status(first["id"])

        second = jobs.submit({})
        while jobs.status(second["id"])["state"] != "complete" and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.03)
        with self.assertRaises(KeyError):
            jobs.status(second["id"])

    def test_queued_results_are_file_backed(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="artifacts-", dir=scratch) as td:
            class ArtifactPixal:
                work_dir = Path(td)
                def infer(self, request, cancel=None, progress=None):
                    return {"ok": True,
                            "glb_b64": base64.b64encode(b"native-glb").decode(),
                            "ply_b64": base64.b64encode(b"native-ply").decode()}
                def reference(self, request, cancel=None):
                    return {"glb_b64": base64.b64encode(b"reference-glb").decode()}

            jobs = app.JobQueue(ArtifactPixal(), retained=1)
            submitted = jobs.submit({"reference": True})
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                status = jobs.status(submitted["id"], include_result=True)
                if status["state"] == "complete":
                    break
                time.sleep(0.01)
            result = status["result"]
            self.assertNotIn("glb_b64", result)
            self.assertEqual(jobs.artifact(submitted["id"], "native.glb").read_bytes(),
                             b"native-glb")
            self.assertEqual(jobs.artifact(submitted["id"], "reference.glb").read_bytes(),
                             b"reference-glb")
            artifact_dir = jobs.artifact(submitted["id"], "native.ply").parent
            jobs.delete(submitted["id"])
            self.assertFalse(artifact_dir.exists())


if __name__ == "__main__":
    unittest.main()
