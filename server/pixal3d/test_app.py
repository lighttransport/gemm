import argparse
import base64
import json
from pathlib import Path
import subprocess
import tempfile
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
                views_dir = Path(command[command.index("--views-dir") + 1])
                captured["command"] = command
                captured["manifest"] = json.loads(
                    (views_dir / "transforms.json").read_text())
                output.write_bytes(b"glTF-test")
                Path(command[command.index("--profile-json") + 1]).write_text(
                    '{"peak_vram_mib": 1024}')
                return subprocess.CompletedProcess(command, 0, '{"views":2}\n', "")

            encoded = base64.b64encode(b"image-data").decode()
            request = {
                "backend": "cuda", "gpu_execution": "resident",
                "gpu_kernels": "mma", "gpu_flow_precision": "mixed",
                "seed": 7, "device": 0, "vram_budget_mib": 12288,
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
        self.assertEqual(result["stats"], {"views": 2})
        self.assertEqual(result["profile"]["peak_vram_mib"], 1024)
        self.assertIn("--views-dir", captured["command"])
        self.assertNotIn("--input", captured["command"])
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


if __name__ == "__main__":
    unittest.main()
