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

    def test_batch_runs_qwen_and_pixal_jobs(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        class FakePixal:
            def __init__(self, work_dir): self.work_dir = Path(work_dir)
            def infer(self, request, cancel=None, progress=None):
                return {"ok": True, "glb_b64": base64.b64encode(b"glTF").decode()}
            def qwen_generate(self, request, cancel=None, progress=None):
                if progress: progress("Qwen test", 50)
                return {"ok": True, "cuda": {"image": "data:image/png;base64,AA=="}}
        with tempfile.TemporaryDirectory(prefix="batch-", dir=scratch) as td:
            jobs = app.JobQueue(FakePixal(td), retained=4, min_free_disk_mib=0)
            try:
                batch = jobs.submit_batch([
                    {"kind": "qwen-image", "request": {"prompt": "a red apple"}},
                    {"kind": "pixal3d", "request": {}},
                ])
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    status = jobs.batch_status(batch["batch_id"], include_results=True)
                    if status["state"] == "complete":
                        break
                    time.sleep(0.01)
                self.assertEqual((status["completed"], status["failed"]), (2, 0))
                self.assertTrue(status["jobs"][0]["result"]["cuda"]["image"].startswith("data:"))
            finally:
                jobs.shutdown()
            recovered = app.JobQueue(FakePixal(td), retained=4, min_free_disk_mib=0)
            try:
                restored = recovered.batch_status(batch["batch_id"])
                self.assertEqual(restored["state"], "complete")
                self.assertEqual(restored["completed"], 2)
            finally:
                recovered.shutdown()

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
        ready, missing = app.reference_environment_ready("missing-backend")
        self.assertFalse(ready)
        self.assertIn("python", missing)
        self.assertIn("o_voxel", missing)
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

    def test_explicit_mask_prepares_shared_reference_rgba(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        commands = []
        reference_input = {}
        with tempfile.TemporaryDirectory(prefix="mask-reference-", dir=scratch) as td:
            server = self.make_server(Path(td))

            def run(command, **kwargs):
                commands.append(command)
                output = Path(command[command.index("--output") + 1])
                if str(server.prepare_script) in command:
                    output.write_bytes(b"prepared-rgba")
                    Path(command[command.index("--metadata") + 1]).write_text(json.dumps({
                        "fov": 0.8, "distance": 1.2, "mesh_scale": 1.0,
                        "mask_source": "mask", "camera_source": "manual"}))
                    return subprocess.CompletedProcess(command, 0, "{}\n", "")
                if str(server.reference_script) in command:
                    reference_input["bytes"] = Path(command[command.index("--image") + 1]).read_bytes()
                    output.write_bytes(b"reference-glb")
                    return subprocess.CompletedProcess(command, 0, "reference\n", "")
                output.write_bytes(b"native-glb")
                Path(command[command.index("--profile-json") + 1]).write_text("{}")
                return subprocess.CompletedProcess(command, 0, "{}\n", "")

            request = {
                "backend": "cuda", "reference": True, "fov": 0.8,
                "image_b64": base64.b64encode(b"rgb").decode(),
                "mask_b64": base64.b64encode(b"mask").decode(),
            }
            with mock.patch.object(app.subprocess, "run", side_effect=run):
                native = server.infer(request)
                reference = server.reference(app.reference_request(request, native))

        preparation = [c for c in commands if str(server.prepare_script) in c]
        self.assertEqual(len(preparation), 1)
        self.assertIn("--mask", preparation[0])
        native_command = next(c for c in commands if "--profile-json" in c)
        self.assertNotIn("--mask", native_command)
        self.assertEqual(reference_input["bytes"], b"prepared-rgba")
        self.assertEqual(base64.b64decode(reference["glb_b64"]), b"reference-glb")

    def test_surface_comparison_is_bounded_and_attached(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="comparison-", dir=scratch) as td:
            server = self.make_server(Path(td))
            measured = {"samples": 50000, "seed": 17, "geometry": {
                "symmetric_chamfer_rms": 0.01,
                "native_to_reference": {"p95": 0.02, "normal_abs_cosine_mean": 0.98},
                "reference_to_native": {"p95": 0.03, "normal_abs_cosine_mean": 0.97}}}
            captured = {}

            def run(command, **kwargs):
                captured["command"] = command
                return subprocess.CompletedProcess(command, 0, json.dumps(measured), "")

            native = {"glb_b64": base64.b64encode(b"native").decode(),
                      "mesh_summary": {"vertices": 10, "triangles": 8,
                                       "bounds": [[0, 0, 0], [1, 1, 1]]}}
            reference = {"glb_b64": base64.b64encode(b"reference").decode(),
                         "mesh_summary": {"vertices": 9, "triangles": 7,
                                          "bounds": [[0, 0, 0], [1, 1, 1]]}}
            with mock.patch.object(app.subprocess, "run", side_effect=run):
                result = dict(native, reference=reference)
                app.attach_comparison(server, result)

        self.assertEqual(result["comparison"]["surface"]["samples"], 50000)
        self.assertEqual(result["comparison"]["surface"]["symmetric_chamfer_rms"], 0.01)
        self.assertEqual(captured["command"][captured["command"].index("--samples") + 1], "50000")

    def test_rendered_comparison_publishes_metrics_and_previews(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="render-comparison-", dir=scratch) as td:
            server = self.make_server(Path(td))
            server.preview_renderer = Path(td) / "preview-render"
            server.preview_renderer.write_bytes(b"renderer")
            artifact_dir = Path(td) / "results" / ("f" * 32)
            artifact_dir.mkdir(parents=True)
            native_path = artifact_dir / "native.glb"
            reference_path = artifact_dir / "reference.glb"
            native_path.write_bytes(b"native")
            reference_path.write_bytes(b"reference")
            measured = {"samples": 50000, "seed": 17, "geometry": {
                "symmetric_chamfer_rms": 0.01,
                "native_to_reference": {"p95": 0.02},
                "reference_to_native": {"p95": 0.03}},
                "renders": [{"name": "view-0.png", "rgb_mae": 0.01,
                             "rgb_rmse": 0.02, "rgb_psnr": 33.98,
                             "silhouette_iou": 0.99}]}

            def run(command, **kwargs):
                if str(server.preview_script) in command:
                    output_dir = Path(command[command.index("--output-dir") + 1])
                    output_dir.mkdir(parents=True)
                    (output_dir / "views.png").write_bytes(b"preview")
                    return subprocess.CompletedProcess(command, 0, "views.png\n", "")
                return subprocess.CompletedProcess(command, 0, json.dumps(measured), "")

            result = {
                "mesh_summary": {"vertices": 10, "triangles": 8,
                                 "bounds": [[0, 0, 0], [1, 1, 1]]},
                "_artifact_files": {"native.glb": native_path},
                "reference": {
                    "mesh_summary": {"vertices": 9, "triangles": 7,
                                     "bounds": [[0, 0, 0], [1, 1, 1]]},
                    "_artifact_files": {"reference.glb": reference_path},
                }}
            with mock.patch.object(app.subprocess, "run", side_effect=run):
                app.attach_comparison(server, result, render_comparison=True)

            self.assertEqual(result["comparison"]["renders"][0]["rgb_psnr"], 33.98)
            previews = result["_comparison_artifact_files"]
            self.assertEqual(previews["native-preview.png"].read_bytes(), b"preview")
            self.assertEqual(previews["reference-preview.png"].read_bytes(), b"preview")

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

    def test_multiview_masks_share_exact_prepared_reference_views(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        captured = {"preparations": [], "reference_inputs": []}
        with tempfile.TemporaryDirectory(prefix="masked-multiview-", dir=scratch) as td:
            server = self.make_server(Path(td))
            artifact_dir = Path(td) / "results" / ("e" * 32)
            artifact_dir.mkdir(parents=True)

            def run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                if str(server.prepare_script) in command:
                    input_path = Path(command[command.index("--input") + 1])
                    prepared = b"prepared-" + input_path.read_bytes()
                    output.write_bytes(prepared)
                    Path(command[command.index("--metadata") + 1]).write_text(json.dumps({
                        "fov": 0.8, "distance": 0.0, "mesh_scale": 1.0,
                        "mask_source": "mask" if "--mask" in command else "rmbg-2.0",
                        "camera_source": "manual"}))
                    captured["preparations"].append(command)
                    return subprocess.CompletedProcess(command, 0, "{}", "")
                if str(server.reference_mv_script) in command:
                    views_dir = Path(command[command.index("--views_dir") + 1])
                    manifest = json.loads((views_dir / "transforms.json").read_text())
                    captured["reference_inputs"] = [
                        (views_dir / frame["file_path"]).read_bytes()
                        for frame in manifest["frames"]]
                    output.write_bytes(b"reference-glb")
                    return subprocess.CompletedProcess(command, 0, "reference", "")
                output.write_bytes(b"native-glb")
                Path(command[command.index("--profile-json") + 1]).write_text("{}")
                return subprocess.CompletedProcess(command, 0, "{}", "")

            request = {
                "backend": "cuda", "reference": True, "auto_mask": True,
                "_artifact_dir": artifact_dir,
                "views": [
                    {"image_b64": base64.b64encode(b"first").decode(),
                     "mask_b64": base64.b64encode(b"mask").decode(),
                     "transform_matrix": IDENTITY},
                    {"image_b64": base64.b64encode(b"second").decode(),
                     "transform_matrix": IDENTITY},
                ],
            }
            with mock.patch.object(app.subprocess, "run", side_effect=run):
                native = server.infer(request)
                reference = server.reference(app.reference_request(request, native))
            captured["native_artifact"] = native["_artifact_files"]["native.glb"].read_bytes()
            captured["reference_artifact"] = (
                reference["_artifact_files"]["reference.glb"].read_bytes())

        self.assertNotIn("glb_b64", native)
        self.assertNotIn("glb_b64", reference)
        self.assertEqual(captured["native_artifact"], b"native-glb")
        self.assertEqual(captured["reference_artifact"], b"reference-glb")
        self.assertIn("--mask", captured["preparations"][0])
        self.assertNotIn("--rembg-model", captured["preparations"][0])
        self.assertIn("--rembg-model", captured["preparations"][1])
        self.assertEqual(captured["reference_inputs"],
                         [b"prepared-first", b"prepared-second"])

    def test_view_mask_upload_is_claimed_once(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="view-mask-upload-", dir=scratch) as td:
            uploads = app.UploadStore(Path(td), retained=2)
            image_id = uploads.put(b"image")
            mask_id = uploads.put(b"mask")
            request, paths = uploads.claim({"views": [{
                "image_upload": image_id, "mask_upload": mask_id,
                "transform_matrix": IDENTITY}]})
            self.assertEqual(app.decode_b64(request["views"][0]["image_b64"],
                                           "image", 100), b"image")
            self.assertEqual(app.decode_b64(request["views"][0]["mask_b64"],
                                           "mask", 100), b"mask")
            self.assertEqual(len(paths), 2)

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

    def test_cuda_file_lock_wait_is_cancellable_and_bounded(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="device-lock-", dir=scratch) as td:
            path = Path(td) / "cuda-0.lock"
            cancel = __import__("threading").Event()
            timer = __import__("threading").Timer(0.05, cancel.set)
            with app.cancellable_file_lock(path, 1):
                timer.start()
                with self.assertRaises(app.JobCancelled):
                    with app.cancellable_file_lock(path, 1, cancel):
                        pass
                timer.cancel()
                with self.assertRaises(app.DeviceBusy):
                    with app.cancellable_file_lock(path, 0.01):
                        pass

    def test_disk_admission_and_bounded_job_log(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="job-ops-", dir=scratch) as td:
            class LoggingPixal:
                work_dir = Path(td)
                def infer(self, request, cancel=None, progress=None):
                    progress("x" * 80)
                    progress("tail-marker")
                    return {"ok": True}

            jobs = app.JobQueue(LoggingPixal(), retained=1,
                                min_free_disk_mib=1, job_log_bytes=64)
            with mock.patch.object(app.shutil, "disk_usage",
                                   return_value=mock.Mock(free=0)):
                with self.assertRaises(app.StorageFull):
                    jobs.submit({})
            submitted = jobs.submit({})
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                status = jobs.status(submitted["id"])
                if status["state"] == "complete":
                    break
                time.sleep(0.01)
            log = jobs.log(submitted["id"]).read_text()
            self.assertLessEqual(len(log.encode()), 64)
            self.assertIn("tail-marker", log)
            self.assertIn("complete", log)

    def test_graceful_shutdown_stops_admission_and_fails_active_job(self):
        started = __import__("threading").Event()
        class BlockingPixal:
            def infer(self, request, cancel=None, progress=None):
                started.set()
                cancel.wait(2)
                raise app.JobCancelled("stopped")

        jobs = app.JobQueue(BlockingPixal(), retained=1)
        submitted = jobs.submit({})
        self.assertTrue(started.wait(1))
        self.assertTrue(jobs.shutdown(1))
        status = jobs.status(submitted["id"])
        self.assertEqual(status["error_code"], "server_shutdown")
        with self.assertRaises(app.ServerShuttingDown):
            jobs.submit({})

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
                    directory = request["_artifact_dir"]
                    native_preview = directory / "native-preview.png"
                    reference_preview = directory / "reference-preview.png"
                    native_preview.write_bytes(b"native-preview")
                    reference_preview.write_bytes(b"reference-preview")
                    return {"ok": True,
                            "glb_b64": base64.b64encode(b"native-glb").decode(),
                            "ply_b64": base64.b64encode(b"native-ply").decode(),
                            "comparison": {"renders": []},
                            "_comparison_artifact_files": {
                                "native-preview.png": native_preview,
                                "reference-preview.png": reference_preview}}
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
            self.assertEqual(result["comparison"]["artifacts"]["native_preview"],
                             f"/v1/jobs/{submitted['id']}/artifacts/native-preview.png")
            self.assertEqual(
                jobs.artifact(submitted["id"], "reference-preview.png").read_bytes(),
                b"reference-preview")
            artifact_dir = jobs.artifact(submitted["id"], "native.ply").parent
            jobs.delete(submitted["id"])
            self.assertFalse(artifact_dir.exists())

    def test_completed_job_survives_restart(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="job-recovery-", dir=scratch) as td:
            class ArtifactPixal:
                work_dir = Path(td)
                def infer(self, request, cancel=None, progress=None):
                    return {"ok": True,
                            "glb_b64": base64.b64encode(b"native-glb").decode(),
                            "ply_b64": base64.b64encode(b"native-ply").decode()}
                def reference(self, request, cancel=None):
                    return {"glb_b64": base64.b64encode(b"reference-glb").decode()}

            first = app.JobQueue(ArtifactPixal(), retained=2)
            submitted = first.submit({"reference": True})
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                status = first.status(submitted["id"], include_result=True)
                if status["state"] == "complete":
                    break
                time.sleep(0.01)
            self.assertEqual(status["state"], "complete")

            directory = Path(td) / "results" / submitted["id"]
            manifest = json.loads((directory / "job.json").read_text())
            self.assertEqual((manifest["version"], manifest["state"]), (2, "complete"))
            self.assertNotIn("request", manifest)
            self.assertFalse(any(path.name.endswith(".tmp") for path in directory.iterdir()))

            recovered = app.JobQueue(ArtifactPixal(), retained=2)
            status = recovered.status(submitted["id"], include_result=True)
            self.assertEqual(status["state"], "complete")
            self.assertTrue(status["result"]["ok"])
            self.assertEqual(recovered.artifact(submitted["id"], "native.glb").read_bytes(),
                             b"native-glb")
            self.assertEqual(recovered.artifact(submitted["id"], "native.ply").read_bytes(),
                             b"native-ply")
            self.assertEqual(recovered.artifact(submitted["id"], "reference.glb").read_bytes(),
                             b"reference-glb")
            self.assertTrue(recovered.delete(submitted["id"])["deleted"])
            self.assertFalse(directory.exists())

    def test_interrupted_job_is_failed_on_restart(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="job-interrupted-", dir=scratch) as td:
            class IdlePixal:
                work_dir = Path(td)

            job_id = "a" * 32
            directory = Path(td) / "results" / job_id
            directory.mkdir(parents=True)
            now = time.time()
            (directory / "native.glb").write_bytes(b"partial")
            (directory / "job.json").write_text(json.dumps({
                "version": 1, "id": job_id, "state": "running",
                "phase": "shape diffusion", "progress": 47,
                "created_at": now - 10, "updated_at": now - 1,
            }))

            recovered = app.JobQueue(IdlePixal(), retained=2)
            status = recovered.status(job_id)
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["phase"], "failed")
            self.assertEqual(status["progress"], 47)
            self.assertEqual(status["error_code"], "server_restarted")
            self.assertIn("server restarted", status["error"])
            self.assertFalse((directory / "native.glb").exists())
            manifest = json.loads((directory / "job.json").read_text())
            self.assertEqual(manifest["state"], "failed")
            self.assertEqual(manifest["error_code"], "server_restarted")

    def test_recovery_expires_old_and_removes_corrupt_jobs(self):
        scratch = app.ROOT / "tmp/pixal3d/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="job-expiry-", dir=scratch) as td:
            class IdlePixal:
                work_dir = Path(td)

            root = Path(td) / "results"
            old_id = "b" * 32
            corrupt_id = "c" * 32
            incomplete_id = "d" * 32
            old = root / old_id
            corrupt = root / corrupt_id
            incomplete = root / incomplete_id
            old.mkdir(parents=True)
            corrupt.mkdir()
            incomplete.mkdir()
            now = time.time()
            (old / "job.json").write_text(json.dumps({
                "version": 1, "id": old_id, "state": "failed",
                "phase": "failed", "progress": 0,
                "created_at": now - 20, "updated_at": now - 10,
                "error_code": "execution_failed", "error": "old",
            }))
            (corrupt / "job.json").write_text("not-json")
            (incomplete / "job.json").write_text(json.dumps({
                "version": 1, "id": incomplete_id, "state": "complete",
                "phase": "complete", "progress": 100,
                "created_at": now - 1, "updated_at": now,
                "result": {"ok": True},
            }))

            recovered = app.JobQueue(IdlePixal(), retained=2, ttl=1)
            with self.assertRaises(KeyError):
                recovered.status(old_id)
            with self.assertRaises(KeyError):
                recovered.status(corrupt_id)
            with self.assertRaises(KeyError):
                recovered.status(incomplete_id)
            self.assertFalse(old.exists())
            self.assertFalse(corrupt.exists())
            self.assertFalse(incomplete.exists())


if __name__ == "__main__":
    unittest.main()
