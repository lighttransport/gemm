"""Quality campaigns reject mismatched runs and release/cancel owned resources."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import venv

import validate_quality as quality

SCRATCH = quality.ROOT / "tmp/hv15-native/tests"
SCRATCH.mkdir(parents=True, exist_ok=True)


class QualityCampaign(unittest.TestCase):
    def test_reference_interpreter_keeps_virtual_environment(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            root = Path(directory)
            environment = root / "reference-venv"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            args = argparse.Namespace(model=root, image=root / "input.png", out=root / "out",
                                      runner=root / "runner", reference_python=environment / "bin/python",
                                      adopt_run=None, adopt_captures=None)
            quality.normalize_paths(args)
            prefix = subprocess.check_output([str(args.reference_python), "-c", "import sys; print(sys.prefix)"],
                                             env=dict(os.environ, TMPDIR=str(root)), text=True).strip()
            self.assertEqual(Path(prefix), environment)

    def test_completed_generation_requires_exact_recipe_and_backend(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            root = Path(directory)
            model, run = root / "model", root / "run"
            model.mkdir()
            run.mkdir()
            (model / "model.json").write_text('{}')
            image = root / "portrait.png"
            image.write_bytes(b'portrait fixture')
            (run / "clip.mp4").write_bytes(b'clip fixture')
            args = argparse.Namespace(prompt=quality.PROMPT, negative_prompt="", seed=42,
                                      model=model, image=image)
            manifest = dict(backend="hv15n_cuda_experimental", task="i2v", preset="quality",
                            prompt=args.prompt, negative_prompt="", seed=42, steps=50, cfg=6,
                            flow_shift=5, frames=81, fps=24, width=480, height=848, gemm="repo",
                            gemm_fallback="error", runner_sha256="runner", model={},
                            image_sha256=quality.digest(image), metrics=dict(backend="hv15n_cuda",
                            memory_fit="pass", repo_gemm_calls=10, cublas_gemm_calls=0,
                            fallback_gemm_calls=0))
            def check(value):
                (run / "manifest.json").write_text(json.dumps(value))
                return quality.validate_generation(run, "i2v", args, "runner")
            check(manifest)
            for key, wrong in [("cfg", 1), ("steps", 12), ("runner_sha256", "stale"),
                               ("image_sha256", "stale"), ("model", {"stale": True})]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    check(dict(manifest, **{key: wrong}))
            for wrong in ({"cublas_gemm_calls": 1}, {"fallback_gemm_calls": 1}, {"memory_fit": "unverified"}):
                with self.subTest(metrics=wrong), self.assertRaises(ValueError):
                    check(dict(manifest, metrics=dict(manifest["metrics"], **wrong)))
            (run / "clip.mp4").unlink()
            with self.assertRaises(ValueError):
                check(manifest)

    def test_lock_wait_cancellation_and_release(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            path = Path(directory) / "gpu.lock"
            cancel = threading.Event()
            with path.open("a") as occupied:
                fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
                timer = threading.Timer(.02, cancel.set)
                timer.start()
                try:
                    with self.assertRaises(quality.Cancelled), quality.device_lock(cancel, path):
                        self.fail("acquired occupied lock")
                finally:
                    timer.join()
            cancel.clear()
            with quality.device_lock(cancel, path):
                with path.open("a") as contender, self.assertRaises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with path.open("a") as available:
                fcntl.flock(available, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_adoption_binds_output_and_cancels_original_wrapper(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            root = Path(directory)
            script, run = root / "generate.py", root / "run"
            run.mkdir()
            ready, cleaned = root / "ready", root / "cleaned"
            script.write_text("import time\nfrom pathlib import Path\n"
                              f"Path({str(ready)!r}).touch()\n"
                              "try:\n    time.sleep(30)\n"
                              f"finally:\n    Path({str(cleaned)!r}).touch()\n")
            child = subprocess.Popen([sys.executable, str(script), "--out", str(run)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     env=dict(os.environ, TMPDIR=str(root)))
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(ready.exists())
                self.assertTrue(quality.process_identity(child.pid, run))
                with self.assertRaises(ValueError):
                    quality.process_identity(child.pid, root / "another-run")
                cancel = threading.Event()
                cancel.set()
                with self.assertRaises(quality.Cancelled):
                    quality.wait_generation(child.pid, run, cancel)
                child.wait(timeout=5)
                self.assertTrue(cleaned.exists())
                with self.assertRaises(FileNotFoundError):
                    quality.process_identity(child.pid, run)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    unittest.main()
