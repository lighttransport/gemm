import argparse
import base64
import json
from pathlib import Path
import subprocess
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
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="web-", dir=scratch) as td:
            server = self.make_server(Path(td))
            encoded = base64.b64encode(b"image-data").decode()
            with self.assertRaisesRegex(ValueError, "include_ply"):
                server.infer({"image_b64": encoded, "include_ply": "yes"})

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
            def infer(self, request, cancel=None):
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
        self.assertEqual(status["result"]["value"], 7)
        self.assertEqual(status["result"]["reference"]["value"], "reference")

    def test_job_failure_has_stable_error_code(self):
        class InvalidPixal:
            def infer(self, request, cancel=None):
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
            def infer(self, request, cancel=None):
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

    def test_raw_upload_is_consumed_and_cleaned(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="uploads-", dir=scratch) as td:
            uploads = app.UploadStore(Path(td), retained=2)
            upload_id = uploads.put(b"raw-image")

            class UploadPixal:
                def infer(self, request, cancel=None):
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


if __name__ == "__main__":
    unittest.main()
