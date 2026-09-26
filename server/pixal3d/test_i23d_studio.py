"""Text/Image -> 3D studio (server/pixal3d/i23d.py) without a GPU: the mock
image backend and a fake Pixal3D runner that writes a minimal GLB."""
import json
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.pixal3d.i23d import Studio, StudioCancelled, StudioError  # noqa: E402
from qimg21_i23d.backends import MockBackend  # noqa: E402


def tiny_glb(path):
    body = json.dumps({"meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
                       "accessors": [{"count": 3, "min": [-1, -1, -1], "max": [1, 1, 1]}, {"count": 3}]}).encode()
    body += b" " * (-len(body) % 4)
    Path(path).write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(body)) +
                           struct.pack("<II", len(body), 0x4E4F534A) + body)


class FakeRunner:
    def __init__(self, name, log):
        self.name, self.backend, self.log = name, "cuda", log

    def available(self):
        return True, []

    def single(self, rgba, out, work, *, fov_rad, mesh_scale=1.0):
        self.log.append(("single", self.name, Path(rgba).name, fov_rad))
        tiny_glb(out)
        return {"runner": self.name, "output": str(out), "seconds": 1.0,
                "mesh": {"vertices": 3, "triangles": 1, "bounds": None}}

    def multiview(self, views_dir, out, work):
        frames = json.loads((Path(views_dir) / "transforms.json").read_text())["frames"]
        self.log.append(("multiview", self.name, len(frames)))
        tiny_glb(out)
        return {"runner": self.name, "output": str(out), "seconds": 1.0,
                "mesh": {"vertices": 3, "triangles": 1, "bounds": None}}


class Backend(MockBackend):
    """The mock image model; a wide request (a turnaround sheet) gets one
    figure per square panel, as the real model draws it."""

    def __init__(self, log):
        super().__init__()
        self.log = log

    def generate(self, request):
        if request.width < 2 * request.height:
            return super().generate(request)
        import numpy as np
        from qimg21_i23d import imageops
        from qimg21_i23d.backends import GenResult
        count = request.width // request.height
        yy, xx = np.mgrid[0:request.height, 0:request.width]
        sheet = np.zeros((request.height, request.width, 4), np.uint8)
        for i in range(count):
            cx, half = (i + 0.5) * request.height, (0.3 if i % 2 == 0 else 0.22) * request.height
            sheet[((xx - cx) / half) ** 2 + ((yy - request.height / 2) / (0.4 * request.height)) ** 2 <= 1] = \
                (230, 230, 240, 255)
        imageops.save_png(sheet, request.out)
        self.log.append(("sheet", request.width, request.height, request.prompt))
        return GenResult(Path(request.out), 0.1, "mock", {})

    def close(self):
        self.log.append(("close",))


class StudioTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(prefix="i23d-studio-", dir=ROOT / "tmp")
        self.log = []
        self.studio = Studio(Path(self.td.name), backend_factory=lambda: Backend(self.log),
                             runner_factory=lambda name, settings: FakeRunner(name, self.log))

        from qimg21_i23d import reconstruct
        self.reconstruct, self.compare = reconstruct, reconstruct.compare_meshes
        reconstruct.compare_meshes = lambda a, b, work, samples=50000: {"symmetric_chamfer_rms": 0.01}

    def tearDown(self):
        self.reconstruct.compare_meshes = self.compare
        self.td.cleanup()

    def run_stage(self, **request):
        return self.studio.run(request)

    def test_text_edit_undo_views_and_reconstruct(self):
        first = self.run_stage(stage="text", prompt="a red teapot", steps=2, seed=3)
        sid = first["session"]
        state = first["state"]
        self.assertEqual(len(state["history"]), 1)
        self.assertEqual(state["current"]["stage"], "text")
        self.assertTrue(state["current"]["url"].startswith(f"/v1/i23d/sessions/{sid}/files/images/"))
        self.assertIn("transparent", first["details"]["prompt"])
        self.assertTrue(self.studio.file(sid, state["current"]["file"]).is_file())

        edited = self.run_stage(stage="edit", session=sid, instruction="make it blue", rect=[0, 0, 256, 256],
                                steps=2)["state"]
        self.assertEqual([h["stage"] for h in edited["history"]], ["text", "edit"])
        self.assertTrue(all(h["seconds"] is not None for h in edited["history"]))

        views = self.run_stage(stage="views", session=sid, count=4, steps=2)["state"]["views"]
        self.assertEqual(len(views["urls"]), 4)
        self.assertEqual(views["of"], edited["current"]["file"])

        built = self.run_stage(stage="reconstruct", session=sid, runner="both", fov=20)
        self.assertEqual([r["runner"] for r in built["details"]["runs"]], ["native", "reference"])
        self.assertEqual(built["details"]["comparison"]["symmetric_chamfer_rms"], 0.01)
        # The image model is released before Pixal3D runs.
        first_recon = next(i for i, e in enumerate(self.log) if e[0] == "single")
        self.assertIn(("close",), self.log[:first_recon])
        self.assertEqual(self.log[first_recon][2], Path(edited["current"]["file"]).name)
        model = built["state"]["models"][-1]
        self.assertTrue(self.studio.file(sid, model["runs"][0]["file"]).is_file())

        multi = self.run_stage(stage="reconstruct", session=sid, mode="multiview", fov=20)
        self.assertEqual(self.log[-1], ("multiview", "native", 4))   # reference + 3 generated
        self.assertEqual(len(multi["state"]["models"]), 2)

        undone = self.studio.undo(sid)
        self.assertEqual(len(undone["history"]), 1)
        self.assertIsNone(undone["views"])      # views were of the undone image
        with self.assertRaises(StudioError):
            self.studio.undo(sid)

    def test_turnaround_views_feed_multiview(self):
        sid = self.run_stage(stage="text", prompt="a cute bunny", steps=2)["session"]
        state = self.run_stage(stage="turnaround", session=sid, count=4, prompt="a cute bunny", steps=2)["state"]
        sheet = next(e for e in self.log if e[0] == "sheet")
        self.assertEqual(sheet[1:3], (2048, 512))
        self.assertIn("turnaround reference sheet of a cute bunny", sheet[3])
        views = state["views"]
        self.assertEqual((views["kind"], views["azimuths"]), ("turnaround", [0.0, 90.0, 180.0, 270.0]))
        self.assertEqual(views["of"], state["current"]["file"])     # the front panel is the object now
        self.assertEqual(state["current"]["stage"], "turnaround")
        self.assertTrue(views["sheet_url"].endswith("/sheet.png"))
        self.run_stage(stage="reconstruct", session=sid, mode="multiview", fov=20)
        self.assertEqual(self.log[-1], ("multiview", "native", 4))
        with self.assertRaises(StudioError):
            self.run_stage(stage="turnaround", session=sid, count=5)

    def test_quality_presets_reach_the_runner(self):
        seen = []
        self.studio._runner_factory = lambda name, settings: seen.append(settings) or FakeRunner(name, self.log)
        sid = self.run_stage(stage="text", prompt="a cup", steps=2)["session"]
        self.run_stage(stage="reconstruct", session=sid, quality="preview", fov=20)
        self.run_stage(stage="reconstruct", session=sid, quality="high", triangle_target=500000, fov=20)
        self.assertEqual([(s.texture_size, s.triangle_target, s.flow_precision) for s in seen],
                         [(1024, 300000, "bf16"), (4096, 500000, "mixed")])
        self.assertEqual(self.studio.state(sid)["models"][-1]["quality"], "high")
        with self.assertRaises(StudioError):
            self.run_stage(stage="reconstruct", session=sid, quality="ultra")

    def test_multiview_needs_views_of_the_current_object(self):
        sid = self.run_stage(stage="text", prompt="a lamp", steps=2)["session"]
        self.run_stage(stage="views", session=sid, count=2, steps=2)
        self.run_stage(stage="edit", session=sid, instruction="brass", steps=2)
        with self.assertRaisesRegex(StudioError, "views of the current object"):
            self.run_stage(stage="reconstruct", session=sid, mode="multiview")

    def test_upload_stage_and_validation(self):
        import base64
        import io
        from PIL import Image
        image = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
        image.paste((200, 30, 30, 255), (100, 50, 200, 150))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        result = self.run_stage(stage="upload", image_b64=base64.b64encode(buffer.getvalue()).decode(),
                                method="alpha")
        self.assertEqual(result["state"]["current"]["stage"], "upload")
        sid = result["session"]
        for bad in ({"stage": "nope"}, {"stage": "edit", "session": sid, "instruction": ""},
                    {"stage": "edit", "session": sid, "instruction": "x", "rect": [1, 2]},
                    {"stage": "text", "prompt": "x", "width": 500},
                    {"stage": "reconstruct", "session": sid, "runner": "blender"},
                    {"stage": "upload", "method": "magic", "image_b64": "AA=="}):
            with self.assertRaises(StudioError):
                self.studio.run(bad)
        with self.assertRaises(StudioError):
            self.studio.run({"stage": "edit", "instruction": "x"})   # no session

    def test_files_stay_inside_the_session(self):
        sid = self.run_stage(stage="text", prompt="a cup", steps=2)["session"]
        other = self.run_stage(stage="text", prompt="a bowl", steps=2)["session"]
        self.assertTrue(self.studio.file(other, "session.json").is_file())
        for bad in (f"../{other}/session.json", "/etc/passwd", "images/../../x.png", "images/nope.png",
                    "uploads"):
            with self.assertRaises(KeyError):
                self.studio.file(sid, bad)
        with self.assertRaises(KeyError):
            self.studio.state("../etc")

    def test_cancel_between_steps(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(StudioCancelled):
            self.studio.run({"stage": "text", "prompt": "a cup", "steps": 2}, cancel)

    def test_undo_refused_while_a_stage_runs(self):
        sid = self.run_stage(stage="text", prompt="a cup", steps=2)["session"]
        self.run_stage(stage="edit", session=sid, instruction="blue", steps=2)
        self.studio.busy.add(sid)
        with self.assertRaisesRegex(StudioError, "running"):
            self.studio.undo(sid)


if __name__ == "__main__":
    unittest.main()
