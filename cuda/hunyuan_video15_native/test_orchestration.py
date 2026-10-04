"""Exercise actual packaging/cancellation and provenance rejection paths."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from unittest.mock import patch
import generate as gen
from vhuman_adapter import reviewed_dataset, publish, expression_frames, restore_clip, candidate_stage

SCRATCH = gen.ROOT / "tmp/hv15-native/tests"
SCRATCH.mkdir(parents=True, exist_ok=True)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def test_aotriton_bridge_backend_validation(self):
        bridge = self.root / "bridge.so"
        bridge.write_bytes(b"fake library")
        with self.assertRaisesRegex(ValueError, "requires ROCm"):
            gen.generate(model=self.root, out=self.root / "out", prompt="test",
                         task="t2v", allow_experimental=True, aotriton_bridge=bridge)
        with self.assertRaisesRegex(ValueError, "existing shared library"):
            gen.generate(model=self.root, out=self.root / "out", prompt="test",
                         task="t2v", backend="rocm", allow_experimental=True,
                         aotriton_bridge=self.root / "absent.so")
    def test_process_progress_and_cancel(self):
        progress = []
        gen.run_process([sys.executable, "-c", "print('PROGRESS 2 12')"],
                        progress=lambda a, b: progress.append((a, b)))
        self.assertEqual(progress, [(2, 12)])
        cancel = threading.Event()
        timer = threading.Timer(.15, cancel.set)
        timer.start()
        started = time.monotonic()
        with self.assertRaises(gen.Cancelled):
            gen.run_process([sys.executable, "-c", "import time; time.sleep(60)"], cancel=cancel)
        timer.join()
        self.assertLess(time.monotonic() - started, 3)
    def test_prepare_and_package(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg absent")
        from PIL import Image
        import numpy as np
        image = self.root / "original.png"
        Image.new("RGBA", (512, 512), (255, 0, 0, 128)).save(image)
        prepared, pixels = gen.prepare_image(image, self.root)
        with Image.open(prepared) as p:
            self.assertEqual(p.size, (480, 848))
            self.assertEqual(p.mode, "RGB")
        value = np.fromfile(pixels, dtype="<f4")
        self.assertEqual(value.size, 3 * 384 * 384)
        self.assertTrue(np.isfinite(value).all())
        frames = self.root / "frames"
        frames.mkdir()
        with self.assertRaises(ValueError):
            gen.package_frames(frames, self.root)
        Image.new("RGB", (480, 848), (30, 60, 90)).save(frames / "frame_00000.ppm")
        for i in range(1, 81):
            os.link(frames / "frame_00000.ppm", frames / f"frame_{i:05d}.ppm")
        gen.package_frames(frames, self.root)
        self.assertGreater((self.root / "clip.mp4").stat().st_size, 1000)
        self.assertTrue((self.root / "poster.png").is_file())
        portrait = self.root / "neutral.png"
        Image.new("RGB", (512, 512), (128, 128, 128)).save(portrait)
        take = self.root / "take"
        take.mkdir()
        gen.atomic_json(take / "animation.json", {"frames": [
            {"t": 0, "v": {"mouthSmileLeft": 0}},
            {"t": 40 / 24, "v": {"mouthSmileLeft": .35, "mouthSmileRight": .25}}]})
        entry = {"expression_frames": {"smile": 40}, "folder": self.root,
                 "manifest": {"prompt": "smile"}, "sha256": gen.digest(self.root / "clip.mp4")}
        restored_clip = restore_clip(entry, portrait, self.root / "observations")
        self.assertTrue(restored_clip.is_file())
        self.assertEqual(json.loads((restored_clip.parent / "transform.json").read_text())["crop_origin"], [184, 0])
        expression_frames([entry], [take], portrait, self.root / "expressions")
        fitted = json.loads((self.root / "expressions/manifest.json").read_text())
        self.assertEqual(fitted["expressions"]["smile"]["controls"]["mouthSmileLeft"], .35)
        with Image.open(self.root / "expressions/smile.png") as result:
            self.assertEqual(result.size, (512, 512))
            self.assertEqual(result.getpixel((0, 0)), (128, 128, 128))
    def test_manifest_integrity_and_path(self):
        model = self.root / "model"
        model.mkdir()
        payload = model / "weight"
        payload.write_bytes(b"verified")
        manifest = {"schema": "hunyuan_video15.model.v1", "checkpoints": {"quality_t2v": "weight"},
                    "components": {k: "weight" for k in ("vae", "qwen", "byt5", "tokenizer")},
                    "sources": {"weight": {"sha256": gen.digest(payload), "revision": "pinned", "bytes": 8}}}
        gen.atomic_json(model / "model.json", manifest)
        gen.model_manifest(model, "t2v", "quality")
        payload.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            gen.model_manifest(model, "t2v", "quality")
        manifest["checkpoints"]["quality_t2v"] = "../outside"
        gen.atomic_json(model / "model.json", manifest)
        with self.assertRaisesRegex(ValueError, "path"):
            gen.model_manifest(model, "t2v", "quality")
    def dataset(self):
        clips = []
        for i, split in enumerate(("train", "validation", "test")):
            folder = self.root / str(i)
            folder.mkdir()
            (folder / "clip.mp4").write_bytes(f"clip{i}".encode())
            gen.atomic_json(folder / "manifest.json", {"task": "i2v", "backend": "hv15n_cuda_experimental",
                "seed": i, "image_sha256": "portrait", "prompt": "smile"})
            clips.append({"clip": str(i), "clip_sha256": gen.digest(folder / "clip.mp4"), "split": split,
                          "generation_manifest_sha256": gen.digest(folder / "manifest.json"),
                          "review": {k: True for k in ("identity", "expression", "camera", "visibility", "artifacts")}})
        path = self.root / "dataset.json"
        gen.atomic_json(path, {"schema": "hv15n.synthetic_dataset.v1", "clips": clips})
        return path
    def test_reviews_splits_and_stale_assets(self):
        path = self.dataset()
        self.assertEqual(len(reviewed_dataset(path)[1]), 3)
        value = json.loads(path.read_text())
        value["clips"][0]["review"]["identity"] = False
        gen.atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, "review"):
            reviewed_dataset(path)
        value["clips"][0]["review"]["identity"] = True
        gen.atomic_json(path, value)
        manifest = json.loads((self.root / "1/manifest.json").read_text())
        manifest["seed"] = 0
        gen.atomic_json(self.root / "1/manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            reviewed_dataset(path)
        value["clips"][1]["generation_manifest_sha256"] = gen.digest(self.root / "1/manifest.json")
        gen.atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, "seed"):
            reviewed_dataset(path)
        manifest["seed"] = 1
        gen.atomic_json(self.root / "1/manifest.json", manifest)
        value["clips"][1]["generation_manifest_sha256"] = gen.digest(self.root / "1/manifest.json")
        gen.atomic_json(path, value)
        (self.root / "2/clip.mp4").write_bytes(b"changed after review")
        with self.assertRaisesRegex(ValueError, "stale"):
            reviewed_dataset(path)
    def test_candidate_failure_is_recorded(self):
        gen.atomic_json(self.root / "candidate.json", {"state": "building", "synthetic": True})
        cancel = threading.Event()
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            with candidate_stage(self.root, cancel):
                raise RuntimeError("fit failed")
        self.assertEqual(json.loads((self.root / "candidate.json").read_text())["state"], "failed")
        cancel.set()
        with self.assertRaises(gen.Cancelled):
            with candidate_stage(self.root, cancel):
                raise gen.Cancelled("cancelled")
        self.assertEqual(json.loads((self.root / "candidate.json").read_text())["state"], "cancelled")
    def test_failure_removes_only_owned_generation(self):
        marker = self.root / "unrelated"
        marker.write_text("keep")
        out = self.root / "run"
        with patch.object(gen, "model_manifest", return_value=({}, {})), \
             patch.object(gen, "run_process", side_effect=RuntimeError("runner failed")):
            with self.assertRaisesRegex(RuntimeError, "runner failed"):
                gen.generate(model=self.root, out=out, prompt="test", task="t2v", allow_experimental=True)
        self.assertFalse(out.exists())
        self.assertEqual(marker.read_text(), "keep")
    def test_publication_is_discoverable_and_atomic(self):
        from server.vhuman import gpu, video
        from server.vhuman.service import EyeService
        work = self.root / "work"
        head = work / "heads/abcdef123456"
        head.mkdir(parents=True)
        (head / "portrait.png").write_bytes(b"portrait")
        def generated(**options):
            out = options["out"]
            out.mkdir()
            (out / "clip.mp4").write_bytes(b"complete clip")
            (out / "poster.png").write_bytes(b"poster")
            return {"prompt": "smile", "preset": "quality", "seed": 42,
                    "backend": "hv15n_cuda_experimental", "task": "i2v"}
        with patch.object(gpu, "execution", side_effect=lambda *a: nullcontext()), \
             patch.object(gpu, "device_session", side_effect=lambda *a: nullcontext()), \
             patch("vhuman_adapter.generate", side_effect=generated):
            result = publish(work=work, head=head.name, model=self.root, prompt="smile", allow_experimental=True)
        published = Path(result["folder"])
        self.assertEqual(len(result["id"]), 32)
        self.assertTrue(json.loads((published / "manifest.json").read_text())["synthetic"])
        self.assertEqual((head / "portrait.png").read_bytes(), b"portrait")
        self.assertFalse(any(head.glob("videos/.partial-*")))
        self.assertEqual(video.list_videos(EyeService(work), head.name)[0]["id"], result["id"])


if __name__ == "__main__":
    unittest.main()
