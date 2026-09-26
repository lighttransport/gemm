import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

from server.qwen_image21.app import (MAX_EVENTS, Demo, Progress, REFERENCE_DEVICES, ROOT,
                                     StepPreviews, compare_runs, denoised_estimate, flow_sigmas,
                                     generation_seconds, timing_breakdown)
import time


def assume_reference(demo: Demo, *devices: str) -> None:
    """Pretend these reference devices are runnable.

    The demos in these tests use interpreter paths that do not exist, so the
    availability gate would refuse every reference request. Seeding the probe
    cache rather than the verdict keeps the real build-matching logic under
    test: a ROCm request only passes here because the interpreter claims HIP.
    """
    for device in devices:
        python = str(demo.reference_python(device))
        demo._probes[python] = {"ok": True, "reason": "", "torch": "0.0+test",
                                "hip": "7.2.0" if device == "rocm" else None,
                                "gpu": True, "python": python}


class QwenImage21RoutingTest(unittest.TestCase):
    def make_demo(self, root: Path) -> Demo:
        demo = Demo(root / "model", root / "quant", root / "cuda-python",
                    root / "work", root / "cuda-native", "127.0.0.1", 0,
                    native_rocm=root / "rocm-native",
                    python_rocm=root / "rocm-python",
                    python_cpu=root / "cpu-python")
        assume_reference(demo, *REFERENCE_DEVICES)
        return demo

    def test_legacy_and_explicit_backend_requests(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            legacy = demo._validate({"prompt": "apple", "mode": "cuda"})
            self.assertEqual((legacy["backend"], legacy["mode"]), ("cuda", "native"))
            rocm = demo._validate({"prompt": "apple", "backend": "rocm"})
            self.assertEqual((rocm["backend"], rocm["mode"]), ("rocm", "native"))
            compare = demo._validate({"prompt": "apple", "backend": "rocm", "mode": "compare"})
            self.assertEqual((compare["backend"], compare["mode"]), ("rocm", "compare"))

    def test_native_components_include_attention_plugin(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            self.assertEqual(demo.native_components("rocm")["attention"].name,
                             "libq21_hip_attention.so")
            self.assertEqual(demo.native_components("cuda")["attention"].name,
                             "libq21_cutlass_attention.so")

    def test_native_command_selects_rocm_driver(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = self.make_demo(root)
            cfg = demo._validate({"prompt": "apple", "backend": "rocm", "width": 256,
                                  "height": 256, "steps": 1})
            commands = []
            with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None, progress=None: commands.append(command)):
                demo._native(cfg, root / "out")
            command = commands[0]
            self.assertIn("--backend", command)
            self.assertEqual(command[command.index("--backend") + 1], "rocm")
            self.assertEqual(command[command.index("--native-bin") + 1], str(root / "rocm-native"))
            self.assertEqual(command[command.index("--native-attention") + 1], "wmma-fused")
            self.assertEqual(command[0], sys.executable)

    def test_rocm_quantized_uses_bf16_wmma_not_cuda_int8(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            (root / "quant").mkdir()
            demo = self.make_demo(root)
            cfg = demo._validate({"prompt": "apple", "backend": "rocm", "quantized": True})
            commands = []
            with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None, progress=None: commands.append(command)):
                demo._native(cfg, root / "out")
            self.assertIn("--quantized-transformer", commands[0])
            self.assertNotIn("--int8-tensor-core", commands[0])

    def test_cuda_quantized_keeps_int8_tensor_core(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            (root / "quant").mkdir()
            demo = self.make_demo(root)
            cfg = demo._validate({"prompt": "apple", "backend": "cuda", "quantized": True})
            commands = []
            with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None, progress=None: commands.append(command)):
                demo._native(cfg, root / "out")
            self.assertIn("--int8-tensor-core", commands[0])
            self.assertIn("--int8-bf16-tail-blocks", commands[0])

    def test_fast_preset_routes_to_fast_runner(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            package = root / "int8"
            package.mkdir()
            (package / "manifest.json").write_text("{}")
            (root / "fast").write_text("")
            demo = Demo(root / "model", root / "quant", root / "cuda-python", root / "work",
                        root / "cuda-native", "127.0.0.1", 0, fast=root / "fast",
                        fast_packages={"int8": package, "nvfp4": root / "missing"})
            self.assertTrue(demo.preset_available("low8"))
            self.assertFalse(demo.preset_available("low8-fp4"))
            cfg = demo._validate({"prompt": "apple", "backend": "cuda", "preset": "low8"})
            commands = []
            with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None, progress=None: commands.append(command)):
                demo._native(cfg, root / "out")
            command = commands[0]
            self.assertEqual(command[command.index("--runner") + 1], "fast")
            self.assertEqual(command[command.index("--preset") + 1], "low8")
            self.assertEqual(command[command.index("--native-bin") + 1], str(root / "fast"))
            self.assertEqual(command[command.index("--quant-package") + 1], str(package))
            self.assertNotIn("--native-attention", command)
            self.assertIn("--native-vae", command)
            with self.assertRaises(RuntimeError):
                demo._native(demo._validate({"prompt": "apple", "preset": "low8-fp4"}), root / "out2")

    def test_fast_preset_validation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            for request in ({"prompt": "a", "backend": "rocm", "preset": "low8"},
                            {"prompt": "a", "preset": "low8", "quantized": True},
                            {"prompt": "a", "preset": "turbo"}):
                with self.assertRaises(ValueError):
                    demo._validate(request)
            self.assertIsNone(demo._validate({"prompt": "a", "preset": ""})["preset"])


class QwenImage21TiledRefineTest(unittest.TestCase):
    """The coarse-to-fine settings: a base pass, a tiled refine and a tiled
    decode, all CUDA-fast-runner only."""

    def make_demo(self, root: Path) -> Demo:
        package = root / "int8"
        package.mkdir(exist_ok=True)
        (package / "manifest.json").write_text("{}")
        (root / "fast").write_text("")
        return Demo(root / "model", root / "quant", root / "cuda-python",
                    root / "work", root / "cuda-native", "127.0.0.1", 0,
                    fast=root / "fast",
                    fast_packages={"int8": package, "nvfp4": root / "missing"})

    def command_for(self, root: Path, demo: Demo, request: dict) -> list[str]:
        cfg = demo._validate(request)
        commands: list[list[str]] = []
        with mock.patch.object(demo, "_run",
                               side_effect=lambda command, cwd, log, env=None, progress=None: commands.append(command)):
            demo._native(cfg, root / "out")
        return commands[0]

    def test_defaults_omit_the_tile_picks_so_the_driver_can_size_them(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = self.make_demo(root)
            command = self.command_for(root, demo, {
                "prompt": "apple", "backend": "cuda", "preset": "low8",
                "width": 2048, "height": 2048, "steps": 20, "upscale": 2})
            self.assertEqual(command[command.index("--upscale") + 1], "2.0")
            self.assertIn("--refine-strength", command)
            self.assertIn("--refine-seed", command)
            # Left out on purpose: the driver probes its own plan for the
            # largest tile that fits, and the UI never sees that number.
            self.assertNotIn("--tile-tokens", command)
            self.assertNotIn("--vae-tile", command)
            self.assertNotIn("--base-steps", command)

    def test_pinned_settings_reach_the_driver(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = self.make_demo(root)
            command = self.command_for(root, demo, {
                "prompt": "apple", "backend": "cuda", "preset": "low8",
                "width": 2048, "height": 2048, "steps": 20, "upscale": 2,
                "base_steps": 12, "tile_tokens": 56, "tile_overlap": 12,
                "refine_strength": 0.35, "refine_seed": 99,
                "vae_tile": 48, "vae_tile_overlap": 8, "vae_tile_bleed": 2})
            for flag, value in (("--upscale", "2.0"), ("--base-steps", "12"),
                                ("--tile-tokens", "56"), ("--tile-overlap", "12"),
                                ("--refine-strength", "0.35"), ("--refine-seed", "99"),
                                ("--vae-tile", "48"), ("--vae-tile-overlap", "8"),
                                ("--vae-tile-bleed", "2")):
                self.assertEqual(command[command.index(flag) + 1], value, flag)

    def test_vae_tiling_is_independent_of_refine_tiling(self):
        """An untiled 2048 run still needs a tiled decode; the driver defaults
        it, and an explicit choice has to survive."""
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = self.make_demo(root)
            command = self.command_for(root, demo, {
                "prompt": "apple", "backend": "cuda", "preset": "low8",
                "width": 2048, "height": 2048, "steps": 4, "vae_tile": 40})
            self.assertNotIn("--upscale", command)
            self.assertEqual(command[command.index("--vae-tile") + 1], "40")

    def test_tiling_needs_a_cuda_fast_preset(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            for request in ({"prompt": "a", "backend": "rocm", "preset": "low8", "upscale": 2},
                            {"prompt": "a", "backend": "cuda", "upscale": 2},
                            {"prompt": "a", "backend": "cuda", "preset": "low8", "upscale": 2,
                             "mode": "compare"}):
                with self.assertRaises(ValueError):
                    demo._validate(request)

    def test_size_limits(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            ok = {"prompt": "a", "backend": "cuda", "preset": "low8"}
            # Untiled fast run: 2048 yes, 4096 no.
            self.assertEqual(demo._validate({**ok, "width": 2048, "height": 2048})["width"], 2048)
            with self.assertRaises(ValueError):
                demo._validate({**ok, "width": 4096, "height": 4096})
            # Tiled refine: 4096.
            self.assertEqual(
                demo._validate({**ok, "width": 4096, "height": 4096, "upscale": 2})["width"], 4096)
            # The reference path stays at 1024 whatever the preset.
            with self.assertRaises(ValueError):
                demo._validate({"prompt": "a", "backend": "cuda", "preset": "low8",
                                "mode": "reference", "width": 2048, "height": 2048})

    def test_bad_settings_are_rejected_with_a_reason(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            ok = {"prompt": "a", "backend": "cuda", "preset": "low8", "upscale": 2,
                  "width": 1024, "height": 1024}
            cases = [
                {**ok, "upscale": 0.5},                       # below 1
                {**ok, "upscale": 8},                         # absurd
                {**ok, "upscale": 3},                         # 341 px base, not 32
                {**ok, "refine_strength": 0},                 # no refine at all
                {**ok, "refine_strength": 1.5},               # full strength is not a refine
                {**ok, "refine_strength": "loud"},
                {**ok, "tile_tokens": 0},
                {**ok, "tile_tokens": 999},                   # larger than the output grid
                {**ok, "tile_tokens": 32, "tile_overlap": 32},
                {**ok, "vae_tile_bleed": 5, "vae_tile_overlap": 8},   # bleed > overlap/2
                {**ok, "vae_tile_bleed": 2},                            # no decode tile to apply it to
                {**ok, "base_steps": 0},
                {**ok, "base_steps": 500},
                {**ok, "refine_seed": -1},
                {**ok, "width": 250},
                {**ok, "width": 256, "height": 300},
            ]
            for request in cases:
                with self.assertRaises(ValueError, msg=request):
                    demo._validate(request)

    def test_bleed_is_allowed_up_to_half_the_overlap(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            cfg = demo._validate({"prompt": "a", "backend": "cuda", "preset": "low8",
                                  "upscale": 2, "width": 2048, "height": 2048,
                                  "vae_tile": 48, "vae_tile_overlap": 8,
                                  "vae_tile_bleed": 4})
            self.assertEqual(cfg["vae_tile_bleed"], 4)
            # Defaults fill in for whichever of the pair the client left out.
            filled = demo._validate({"prompt": "a", "backend": "cuda", "preset": "low8",
                                     "width": 2048, "height": 2048, "vae_tile": 48})
            self.assertEqual((filled["vae_tile_overlap"], filled["vae_tile_bleed"]), (8, 2))

    def test_upscale_of_one_is_the_untiled_path(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = self.make_demo(root)
            command = self.command_for(root, demo, {
                "prompt": "apple", "backend": "cuda", "preset": "low8",
                "width": 1024, "height": 1024, "steps": 4})
            for flag in ("--upscale", "--tile-tokens", "--refine-strength", "--vae-tile"):
                self.assertNotIn(flag, command)

    def test_summary_reports_the_plan_and_the_chosen_tiles(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            log = Path(td) / "run.log"
            log.write_text("\n".join([
                "fast: preset low8 = --vram-budget-mib 7168",
                "fast: plan 25 resident + 7 streamed blocks",
                "fast: weights ready in 3.7 s",
                "refine tile: 128 of 128 latent tokens is the largest that fits",
                "fast: prefill 0.36 s, 1 tiles x 8 refine steps 66.4 s",
                "qimg21-vae: 128x128 latent grid in 3x3 tiles of 48x48",
                "fast: step 1/8 sigma=0.6631000"]))
            summary = Demo._summary(log)
            self.assertTrue(any("plan 25 resident" in line for line in summary))
            self.assertTrue(any("largest that fits" in line for line in summary))
            self.assertTrue(any("3x3 tiles of 48x48" in line for line in summary))
            # Per-step chatter is noise.
            self.assertFalse(any(line.startswith("fast: step") for line in summary))
            self.assertEqual(Demo._summary(Path(td) / "missing.log"), [])


class QwenImage21ProgressTest(unittest.TestCase):
    """The progress projection: stage names, per-step device time and the
    accumulated figure. The step duration has to be the denoiser's own number,
    never a difference of poll timestamps, so the parser is tested against real
    log lines."""

    # Trimmed from a real tiled 1024x1024 run on the RTX 5060 Ti.
    LOG = [
        "+ /root/cuda/qimg21/test_cuda_qimg21_text --model M --prompt apple",
        "text: native tokenizer produced 25 tokens, drop-prefix=14",
        "text: layer 1/36 (25 tokens)",
        "text: layer 36/36 (25 tokens)",
        "  (test_cuda_qimg21_text: 4.6 s)",
        "+ /root/tmp/venv/bin/python /root/cuda/qimg21/make_native_fixture.py --out-dir W",
        "  (make_native_fixture.py: 2.2 s)",
        "+ /root/cuda/qimg21/test_cuda_qimg21_fast --preset low8 --height-tokens 32",
        "fast: preset low8 = --vram-budget-mib 7168 --weights int8 --attention sage",
        "fast: plan 27 resident + 5 streamed blocks (208.3 MiB max)",
        "fast: weights ready in 3.32 s (6311 MiB device)",
        "fast: step 1/20 sigma=1.0000000 1518.7 ms",
        "fast: step 2/20 sigma=0.9681666 76.5 ms",
        "fast: step 3/20 sigma=0.9349497 118.7 ms",
        "base pass: 512x512 px, 20 steps",
        "refine tile: 64 of 64 latent tokens is the largest that fits the low8 budget",
        "+ /root/cuda/qimg21/test_cuda_qimg21_fast --refine-from W/base.npy --tile-tokens 64",
        "fast: tiled refine 64x64 over a 32x32 base grid, 1x1 tiles of 64x64 tokens",
        "fast: step 1/8 sigma=0.5358800 1781.2 ms",
        "fast: step 2/8 sigma=0.4000000 1500.0 ms",
        "fast: tile 1/1 (row 0 col 0) at latent 0,0",
        "  (test_cuda_qimg21_fast: 19.8 s)",
        "+ /root/cuda/qimg21/test_cuda_qimg21_vae --latents W/native_latents.npy",
        "qimg21-vae: 64x64 latent grid in 2x2 tiles of 48x48 (stride 40, overlap 8, bleed 2)",
        "qimg21-vae: tile 1/4 (row 0 col 0) at latent 0,0",
        "qimg21-vae: tile 4/4 (row 1 col 1) at latent 16,16",
        "  (test_cuda_qimg21_vae: 5.2 s)",
    ]

    def replay(self, lines=None, now=100.0):
        tracker = Progress(50.0)
        events = [tracker.feed(line, now + i * 0.1) for i, line in enumerate(lines or self.LOG)]
        return tracker, [event for event in events if event]

    def test_stages_are_named_from_the_command_not_the_binary(self):
        tracker, events = self.replay()
        stages = [e["stage"] for e in events if e["kind"] == "stage"]
        self.assertEqual(stages, ["encode prompt", "sample noise", "denoise", "denoise", "decode"])
        # The driver's own "(binary: N s)" lines close a stage; there are four
        # in the log because the refine pass is a second denoiser invocation.
        done = [e for e in events if e["kind"] == "stage_done"]
        self.assertEqual([e["seconds"] for e in done], [4.6, 2.2, 19.8, 5.2])
        self.assertEqual([e["stage"] for e in done],
                         ["test_cuda_qimg21_text", "make_native_fixture.py",
                          "test_cuda_qimg21_fast", "test_cuda_qimg21_vae"])

    def test_each_step_reports_its_own_ms_and_the_running_total(self):
        tracker, events = self.replay()
        steps = [e for e in events if e["kind"] == "step"]
        self.assertEqual([(s["index"], s["total"]) for s in steps],
                         [(1, 20), (2, 20), (3, 20), (1, 8), (2, 8)])
        self.assertEqual([s["ms"] for s in steps], [1518.7, 76.5, 118.7, 1781.2, 1500.0])
        # Accumulated is the sum of the denoiser's own measurements, so the
        # first step dominates rather than being averaged away.
        self.assertAlmostEqual(steps[0]["accum_ms"], 1518.7, places=6)
        self.assertAlmostEqual(steps[1]["accum_ms"], 1518.7 + 76.5, places=6)
        self.assertAlmostEqual(steps[-1]["accum_ms"], 1518.7 + 76.5 + 118.7 + 1781.2 + 1500.0, places=6)
        self.assertEqual(tracker.total_steps, 5)

    def test_a_step_without_measured_ms_still_advances(self):
        # Without --profile the runner prints no duration; the step must still
        # be reported rather than dropped, and must not be invented.
        _, events = self.replay(["+ bin", "fast: step 1/4 sigma=1.0000000", "fast: step 2/4 sigma=0.5"])
        steps = [e for e in events if e["kind"] == "step"]
        self.assertEqual(len(steps), 2)
        self.assertIsNone(steps[0]["ms"])
        self.assertEqual(steps[0]["accum_ms"], 0.0)

    def test_text_layers_and_tiles_both_report_progress(self):
        _, events = self.replay()
        layers = [e for e in events if e["kind"] == "layer"]
        self.assertEqual([(l["index"], l["total"]) for l in layers], [(1, 36), (36, 36)])
        tiles = [e for e in events if e["kind"] == "tile"]
        # One refine tile from the denoiser, then four decode tiles.
        self.assertEqual([(t["index"], t["total"]) for t in tiles],
                         [(1, 1), (1, 4), (4, 4)])
        self.assertEqual([t["stage"] for t in tiles], ["denoise", "decode", "decode"])

    def test_plan_lines_are_kept_as_notes_for_the_result_summary(self):
        tracker, events = self.replay()
        plans = [e["text"] for e in events if e["kind"] == "plan"]
        self.assertTrue(any("plan 27 resident" in p for p in plans))
        self.assertTrue(any("tiled refine 64x64" in p for p in plans))
        self.assertEqual(tracker.notes, plans)
        self.assertLessEqual(len(tracker.notes), 12)

    def test_unrelated_lines_are_ignored(self):
        _, events = self.replay(["cuda_qimg: NVIDIA GeForce RTX 5060 Ti",
                                 "text: flash-exact causal GQA attention enabled",
                                 "native denoise trace: /root/tmp/x",
                                 ""])
        self.assertEqual(events, [])

    def test_a_job_records_events_behind_a_cursor(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = Demo(Path(td) / "m", Path(td) / "q", Path(td) / "py", Path(td) / "w",
                        Path(td) / "n", "127.0.0.1", 0)
            self.assertIsNone(demo.progress_state("nope"))
            demo._begin("job1", time.monotonic())
            sink = demo._progress("job1")
            for line in ProgressTestLog:
                sink(line)
            first = demo.progress_state("job1", 0)
            self.assertGreater(len(first["events"]), 0)
            self.assertTrue(first["events"][0]["kind"] == "stage")
            # A cursor makes the second poll incremental and nothing repeats.
            second = demo.progress_state("job1", first["cursor"])
            self.assertEqual(second["events"], [])
            self.assertEqual(second["cursor"], first["cursor"])
            self.assertEqual(demo.progress_state("job1", 0)["cursor"], first["cursor"])
            demo._say("job1", "hello")
            self.assertEqual(demo.progress_state("job1", 0)["events"][-1]["text"], "hello")
            demo._end("job1")
            self.assertTrue(demo.progress_state("job1")["done"])

    def test_the_cursor_stays_correct_after_old_events_are_dropped(self):
        """A long tiled run outgrows the event buffer. The cursor is absolute,
        so a client that was away for a while resumes where the log is now
        rather than being handed a shifted window or a repeat."""
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = Demo(Path(td) / "m", Path(td) / "q", Path(td) / "py", Path(td) / "w",
                        Path(td) / "n", "127.0.0.1", 0)
            demo._begin("job1", time.monotonic())
            sink = demo._progress("job1")
            for step in range(MAX_EVENTS + 500):
                sink(f"fast: step {step + 1}/{MAX_EVENTS + 500} sigma=1.0 12.5 ms")
            state = demo.progress_state("job1", 0)
            self.assertEqual(len(state["events"]), MAX_EVENTS)
            self.assertEqual(state["cursor"], MAX_EVENTS + 500)
            # The oldest survivors are the ones still held, and the cursor a
            # client last saw still lines up with the event it has not read.
            self.assertEqual(state["events"][0]["index"], 501)
            old_cursor = 700
            resumed = demo.progress_state("job1", old_cursor)
            self.assertEqual(resumed["events"][0]["index"], old_cursor + 1)
            self.assertEqual(resumed["cursor"], MAX_EVENTS + 500)
            # A cursor older than the buffer is not an error, and does not
            # replay from the beginning of the retained window.
            far_back = demo.progress_state("job1", 0)
            self.assertEqual(far_back["events"][0]["index"], 501)

    def test_a_malformed_poll_is_a_400_not_a_dropped_connection(self):
        """A polling client that gets no response cannot tell a typo from a dead
        server, so bad query parameters have to come back as a status. This is a
        real regression: do_GET had no error handling, so a ValueError escaped
        the handler and curl saw an empty reply."""
        import http.client
        import threading
        from http.server import ThreadingHTTPServer
        from server.qwen_image21.app import Handler

        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = Demo(Path(td) / "m", Path(td) / "q", Path(td) / "py", Path(td) / "w",
                        Path(td) / "n", "127.0.0.1", 0)
            demo._begin("job1", time.monotonic())
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.demo = demo  # type: ignore[attr-defined]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                def poll(query):
                    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    try:
                        conn.request("GET", "/api/progress?" + query)
                        response = conn.getresponse()
                        return response.status, response.read()
                    finally:
                        conn.close()
                status, body = poll("job=job1&since=0")
                self.assertEqual(status, 200)
                self.assertIn(b'"cursor": 0', body)
                # A space, a slash or a quote in the id is a client mistake, and
                # the id is a dictionary key, so it is refused before lookup.
                for bad in ("job=bad%20id", "job=a%2Fb", "job=" + "x" * 65, "since=abc"):
                    status, body = poll(bad)
                    self.assertEqual(status, 400, f"{bad} -> {status}")
                    self.assertIn(b'"ok": false', body)
                # An id that is well formed but unknown is a 404, not a 400:
                # the difference tells a client whether to keep its cursor.
                status, body = poll("job=nosuchjob")
                self.assertEqual(status, 404)
                # A negative cursor is clamped rather than refused: a client
                # that underflows its own bookkeeping gets the full history
                # back, which it already knows how to handle.
                status, _ = poll("job=job1&since=-5")
                self.assertEqual(status, 200)
            finally:
                server.shutdown()
                server.server_close()


ProgressTestLog = [
    "+ /root/cuda/qimg21/test_cuda_qimg21_fast --model M",
    "fast: step 1/2 sigma=1.0000000 100.0 ms",
    "fast: step 2/2 sigma=0.0200000 50.0 ms",
]


class QwenImage21ReferenceDeviceTest(unittest.TestCase):
    """The PyTorch reference can run on CUDA, ROCm or CPU. Those are different
    PyTorch builds, not just different devices, so the routing has to pick the
    right interpreter and refuse a build that is not installed -- before a run
    starts, not after it dies on an import error."""

    def make_demo(self, root: Path) -> Demo:
        demo = Demo(root / "model", root / "quant", root / "cuda-python",
                    root / "work", root / "cuda-native", "127.0.0.1", 0,
                    native_rocm=root / "rocm-native",
                    python_rocm=root / "rocm-python",
                    python_cpu=root / "cpu-python")
        return demo

    def test_the_reference_device_defaults_to_the_native_backend(self):
        """Existing requests must keep pointing where they always did."""
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            assume_reference(demo, *REFERENCE_DEVICES)
            cuda = demo._validate({"prompt": "apple", "backend": "cuda", "mode": "reference"})
            self.assertEqual(cuda["reference_device"], "cuda")
            rocm = demo._validate({"prompt": "apple", "backend": "rocm", "mode": "reference"})
            self.assertEqual(rocm["reference_device"], "rocm")
            # A native-only request does not need a reference at all, so it is
            # not refused when no PyTorch build is installed.
            native = demo._validate({"prompt": "apple", "backend": "rocm", "mode": "native"})
            self.assertEqual(native["reference_device"], "rocm")

    def test_the_reference_device_is_validated(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            assume_reference(demo, *REFERENCE_DEVICES)
            with self.assertRaises(ValueError) as caught:
                demo._validate({"prompt": "apple", "mode": "reference", "reference_device": "mps"})
            self.assertIn("reference_device", str(caught.exception))

    def test_a_missing_build_is_refused_with_the_reason(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            demo = self.make_demo(Path(td))
            # The ROCm interpreter here turns out to hold a CUDA build.
            demo._probes[str(demo.python_rocm)] = {
                "ok": True, "reason": "", "torch": "2.14.0+cu130", "hip": None,
                "gpu": True, "python": str(demo.python_rocm)}
            with self.assertRaises(ValueError) as caught:
                demo._validate({"prompt": "apple", "backend": "rocm", "mode": "reference"})
            message = str(caught.exception)
            self.assertIn("rocm", message)
            self.assertIn("CUDA build", message)

    def test_each_device_runs_under_its_own_interpreter(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = self.make_demo(root)
            self.assertEqual(demo.reference_python("cuda"), root / "cuda-python")
            self.assertEqual(demo.reference_python("rocm"), root / "rocm-python")
            self.assertEqual(demo.reference_python("cpu"), root / "cpu-python")
            # A CPU reference has no build of its own to require, so an unnamed
            # --python-cpu falls back to the CUDA environment rather than
            # inventing a third venv that has to be built first.
            fallback = Demo(root / "m", root / "q", root / "cuda-python", root / "w",
                            root / "n", "127.0.0.1", 0)
            self.assertEqual(fallback.reference_python("cpu"), root / "cuda-python")
            self.assertEqual(fallback.reference_python("rocm"), root / "cuda-python")

    def test_the_reference_command_carries_the_device(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = self.make_demo(root)
            for device, python in (("cuda", "cuda-python"), ("rocm", "rocm-python"),
                                   ("cpu", "cpu-python")):
                assume_reference(demo, device)
                cfg = demo._validate({"prompt": "apple", "width": 256, "height": 256,
                                      "steps": 1, "mode": "reference",
                                      "reference_device": device})
                commands = []
                with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None, progress=None: commands.append(command)):
                    demo._reference(cfg, root / "out")
                command = commands[0]
                self.assertEqual(command[command.index("--device") + 1], device)
                self.assertEqual(command[0], str(root / python))
                # The driver writes reference.png into the dump directory it was
                # given, so that is the file the server has to go back for.
                dump = Path(command[command.index("--dump-dir") + 1])
                self.assertEqual(dump.name, device)
                self.assertIn("cuda/qimg21/reference.py", command[1])

    def test_the_returned_reference_path_is_the_file_the_driver_wrote(self):
        """The server used to hand back a name the driver never wrote, which
        made every reference request fail reading the image."""
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = self.make_demo(root)
            assume_reference(demo, "cpu")
            cfg = demo._validate({"prompt": "apple", "width": 256, "height": 256,
                                  "steps": 1, "mode": "reference", "reference_device": "cpu"})

            def fake_run(command, cwd, log, env=None, progress=None):
                # Stand in for the driver: write exactly what it writes.
                dump = Path(command[command.index("--dump-dir") + 1])
                dump.mkdir(parents=True, exist_ok=True)
                (dump / "reference.png").write_bytes(b"png")

            with mock.patch.object(demo, "_run", side_effect=fake_run):
                image = demo._reference(cfg, root / "out")
            self.assertTrue(image.is_file(), f"{image} was not written by the run")
            self.assertTrue(demo._data_url(image).startswith("data:image/png;base64,"))

    def test_a_cpu_compare_says_it_is_not_a_parity_result(self):
        """A CPU reference runs different kernels on different hardware, so a
        side-by-side is a look at both pictures rather than agreement."""
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = self.make_demo(root)
            assume_reference(demo, "cpu")

            def fake_run(command, cwd, log, env=None, progress=None):
                # Stand in for both drivers: the reference writes into its dump
                # directory, the native one into --out.
                if "--dump-dir" not in command:
                    out = Path(command[command.index("--out") + 1])
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(b"png")
                    return
                dump = Path(command[command.index("--dump-dir") + 1])
                dump.mkdir(parents=True, exist_ok=True)
                (dump / "reference.png").write_bytes(b"png")

            with mock.patch.object(demo, "_run", side_effect=fake_run), \
                 mock.patch.object(demo, "_summary", return_value=[]):
                cpu = demo.generate({"prompt": "apple", "width": 256, "height": 256,
                                     "steps": 1, "mode": "compare", "reference_device": "cpu"})
            self.assertIn("note", cpu["reference"])
            self.assertIn("composition", cpu["reference"]["note"])
            self.assertEqual(cpu["reference"]["device"], "cpu")

            assume_reference(demo, "cuda")
            with mock.patch.object(demo, "_run", side_effect=fake_run), \
                 mock.patch.object(demo, "_summary", return_value=[]):
                cuda = demo.generate({"prompt": "apple", "width": 256, "height": 256,
                                      "steps": 1, "mode": "compare", "reference_device": "cuda"})
            self.assertNotIn("note", cuda["reference"])
            self.assertEqual(cuda["reference"]["device"], "cuda")

    def test_a_build_probe_classifies_what_it_finds(self):
        """The health endpoint has to tell the form which reference devices are
        real, and a CUDA build answering a ROCm request is the case that matters:
        both report torch.cuda.is_available() == True."""
        def probe(stdout, returncode=0, stderr=""):
            return mock.Mock(returncode=returncode, stdout=stdout, stderr=stderr)
        cases = [
            ("cuda", '{"torch": "2.14.0+cu130", "hip": null, "cuda": true}', True, ""),
            ("cuda", '{"torch": "2.14.0+cu130", "hip": null, "cuda": false}', False, "sees no GPU"),
            ("cuda", '{"torch": "2.9.0+rocm", "hip": "7.2.0", "cuda": true}', False, "ROCm build"),
            ("rocm", '{"torch": "2.9.0+rocm", "hip": "7.2.0", "cuda": true}', True, ""),
            ("rocm", '{"torch": "2.14.0+cu130", "hip": null, "cuda": true}', False, "CUDA build"),
            ("rocm", '{"torch": "2.9.0+rocm", "hip": "7.2.0", "cuda": false}', False, "sees no GPU"),
            # Any build runs on the CPU, so the CPU route only needs an import.
            ("cpu", '{"torch": "2.14.0+cu130", "hip": null, "cuda": true}', True, ""),
        ]
        for device, stdout, available, fragment in cases:
            with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
                root = Path(td)
                for name in ("cuda-python", "rocm-python", "cpu-python"):
                    (root / name).write_text("")
                demo = self.make_demo(root)
                with mock.patch("subprocess.run", return_value=probe(stdout)):
                    build = demo.torch_build(device)
                self.assertEqual(build["available"], available,
                                 f"{device} {stdout} -> {build}")
                if not available:
                    self.assertIn(fragment, build["reason"])
                self.assertEqual(build["torch"], stdout.split('"torch": "')[1].split('"')[0])

    def test_a_missing_interpreter_is_reported_not_probed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = self.make_demo(root)
            with mock.patch("subprocess.run") as run:
                build = demo.torch_build("rocm")
            run.assert_not_called()
            self.assertFalse(build["available"])
            self.assertIn("no interpreter", build["reason"])

    def test_one_interpreter_is_probed_once_for_two_devices(self):
        """The CPU and CUDA routes share an interpreter by default, so probing
        per device would import torch twice for the same answer on every health
        call."""
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            for name in ("cuda-python", "rocm-python", "cpu-python"):
                (root / name).write_text("")
            demo = Demo(root / "model", root / "quant", root / "cuda-python",
                        root / "work", root / "cuda-native", "127.0.0.1", 0,
                        native_rocm=root / "rocm-native",
                        python_rocm=root / "rocm-python")
            self.assertEqual(demo.reference_python("cpu"), demo.reference_python("cuda"))
            answer = mock.Mock(returncode=0,
                               stdout='{"torch": "2.14.0+cu130", "hip": null, "cuda": true}',
                               stderr="")
            with mock.patch("subprocess.run", return_value=answer) as run:
                for _ in range(3):
                    demo.reference_devices()
            # Three devices, two distinct interpreters, one call each.
            self.assertEqual(run.call_count, 2)


class QwenImage21CompareTest(unittest.TestCase):
    def test_a_compare_starts_the_runner_from_the_reference_noise(self):
        """Seed-derived noise only agrees when both sides draw it the same way;
        handing the reference's own noise to the runner makes every device pair
        comparable, so the reference has to run first."""
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = QwenImage21RoutingTest.make_demo(None, root)
            commands = []

            def fake_run(command, cwd, log, env=None, progress=None):
                commands.append(command)
                if "--dump-dir" in command:
                    dump = Path(command[command.index("--dump-dir") + 1])
                    dump.mkdir(parents=True, exist_ok=True)
                    (dump / "reference.png").write_bytes(b"png")
                    (dump / "initial_latents.npy").write_bytes(b"npy")
                    return
                out = Path(command[command.index("--out") + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"png")

            with mock.patch.object(demo, "_run", side_effect=fake_run), \
                 mock.patch.object(demo, "_summary", return_value=[]):
                result = demo.generate({"prompt": "apple", "width": 256, "height": 256,
                                        "steps": 1, "mode": "compare", "reference_device": "cuda"})
            reference, native = commands
            self.assertIn("cuda/qimg21/reference.py", reference[1])
            self.assertIn("--dump-initial-latents", reference)
            dump = Path(reference[reference.index("--dump-dir") + 1])
            self.assertEqual(Path(native[native.index("--initial-latents") + 1]),
                             dump / "initial_latents.npy")
            self.assertTrue(result["reference"]["matched_noise"])
            # Fake fixtures leave nothing to measure; the pictures still come back.
            self.assertIn("compare", result)
            self.assertIn("image", result["cuda"])

    def test_only_a_compare_hands_the_runner_the_reference_noise(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = QwenImage21RoutingTest.make_demo(None, root)
            cfg = demo._validate({"prompt": "apple", "mode": "native"})
            commands = []
            with mock.patch.object(demo, "_run", side_effect=lambda c, *a, **k: commands.append(c)):
                demo._native(cfg, root / "out")
                demo._reference(demo._validate({"prompt": "apple", "mode": "reference"}), root / "out")
            self.assertNotIn("--initial-latents", commands[0])
            # The reference always saves its noise: the step-0 preview needs it.
            self.assertIn("--dump-initial-latents", commands[1])

    def test_metrics_name_the_first_stage_that_leaves_the_reference(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            ref, work = root / "ref", root / "work"
            (work / "prompt").mkdir(parents=True); (work / "steps").mkdir(); ref.mkdir()
            rng = np.random.default_rng(0)
            noise = rng.standard_normal((16, 64)).astype(np.float32)
            np.save(ref / "initial_latents.npy", noise); np.save(work / "latents.npy", noise)
            text = rng.standard_normal((1, 8, 32)).astype(np.float32)
            np.save(ref / "prompt_embeds.npy", text); np.save(work / "prompt" / "prompt_embeds.npy", text[0])
            good = rng.standard_normal((1, 16, 64)).astype(np.float32)
            np.save(ref / "step_000.npy", good); np.save(work / "steps" / "step_000.npy", good)
            np.save(ref / "step_001.npy", good); np.save(work / "steps" / "step_001.npy", -good)
            # step 2 was dumped by the reference only.
            np.save(ref / "step_002.npy", good)
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            pixels[0, 0, 0] = 100
            np.save(ref / "reference_rgba.npy", np.dstack([pixels, np.full((32, 32), 255, np.uint8)]))
            shifted = pixels.copy(); shifted[0, 0, 0] = 140
            Image.fromarray(shifted).save(root / "native.png")

            m = compare_runs(ref, work, root / "native.png")
            stages = {s["stage"]: s for s in m["stages"]}
            self.assertAlmostEqual(stages["initial latents"]["cosine"], 1.0)
            # A squeezed batch axis on one side is the same tensor.
            self.assertAlmostEqual(stages["text embeddings"]["cosine"], 1.0)
            self.assertAlmostEqual(stages["step 0"]["cosine"], 1.0)
            self.assertAlmostEqual(stages["step 1"]["cosine"], -1.0)
            self.assertTrue(stages["step 2"]["missing"])
            self.assertEqual(m["first_divergence"], "step 1")
            self.assertEqual(m["image"]["max_abs"], 40.0)
            self.assertGreater(m["image"]["psnr"], 30)


class QwenImage21PreviewTest(unittest.TestCase):
    def test_the_schedule_matches_the_runner(self):
        # test_cuda_qimg21_fast printed these for a 512x512 (1024-token), 20-step run.
        sigmas = flow_sigmas(20, 1024)
        self.assertEqual(len(sigmas), 21)
        self.assertAlmostEqual(sigmas[0], 1.0, places=6)
        self.assertAlmostEqual(sigmas[1], 0.9682, places=4)
        self.assertAlmostEqual(sigmas[12], 0.5013, places=4)
        self.assertAlmostEqual(sigmas[19], 0.02, places=6)
        self.assertEqual(sigmas[20], 0.0)
        self.assertEqual(flow_sigmas(1, 256), [1.0, 0.0])

    def test_the_estimate_recovers_the_clean_latent_of_a_straight_path(self):
        import numpy as np
        rng = np.random.default_rng(1)
        clean, noise = rng.standard_normal((4, 64)), rng.standard_normal((4, 64))
        at = lambda sigma: (1 - sigma) * clean + sigma * noise
        estimate = denoised_estimate(at(0.6), at(0.8), 0.8, 0.6)
        np.testing.assert_allclose(estimate, clean, atol=1e-12)
        # Without a previous step there is nothing to extrapolate from.
        np.testing.assert_array_equal(denoised_estimate(at(0.6), None, 0.8, 0.6), at(0.6))

    def test_each_new_step_dump_becomes_one_preview(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            steps = root / "steps"; steps.mkdir()
            np.save(root / "latents.npy", np.zeros((16 * 16, 64), np.float32))
            events = []
            watcher = StepPreviews("native", steps, root / "latents.npy", [(16, 16, 3)], events.append)
            watcher.sweep()
            self.assertEqual(events, [])
            np.save(steps / "step_000.npy", np.ones((1, 256, 64), np.float32))
            (steps / "step_001.npy").write_bytes(b"\x93NUMPY half written")
            watcher.sweep()
            self.assertEqual([e["index"] for e in events], [1])
            np.save(steps / "step_001.npy", np.ones((256, 64), np.float32))
            np.save(steps / "block_00.npy", np.ones((3, 64), np.float32))
            watcher.sweep(); watcher.sweep()
            self.assertEqual([(e["source"], e["index"], e["total"]) for e in events],
                             [("native", 1, 3), ("native", 2, 3)])
            self.assertTrue(events[0]["image"].startswith("data:image/png;base64,"))

    def test_a_tiled_run_previews_its_base_grid(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            root = Path(td)
            demo = QwenImage21RoutingTest.make_demo(None, root)
            cfg = demo._validate({"prompt": "apple", "width": 512, "height": 512, "steps": 8,
                                  "preset": "accurate", "upscale": 2, "base_steps": 5})
            watcher = demo._previews("job", "native", root, root / "none.npy", cfg)
            self.assertEqual(watcher.grids, [(16, 16, 5), (32, 32, 8)])
            events = []
            watcher.emit = events.append
            np.save(root / "step_000.npy", np.zeros((256, 64), np.float32))
            watcher.sweep()
            self.assertEqual((events[0]["index"], events[0]["total"]), (1, 5))
            # Nobody is watching a run without a job id, so nothing is decoded.
            self.assertNotIsInstance(demo._previews(None, "native", root, root, cfg), StepPreviews)


class QwenImage21TimingTest(unittest.TestCase):
    LOG = """text: prompt embedding cache hit (abc)
timing: prompt embedding cache hit 0.000 s
+ /x/cuda/qimg21/test_cuda_qimg21_fast --preset fast12
timing: CUDA init + kernels + memory plan 0.238 s
timing: load transformer weights (6936 MiB) 2.630 s
timing: denoise 10 steps (image generation) 3.234 s
  (test_cuda_qimg21_fast: 6.6 s)
+ /x/cuda/qimg21/test_cuda_qimg21_text --prompt p
timing: 36 layers 3.447 s (13.9 GB of weights streamed at 4.0 GB/s, 2.842 s waiting on the upload)
"""

    def test_a_timing_line_becomes_an_event_under_its_stage(self):
        tracker = Progress(0.0)
        tracker.feed("+ /x/test_cuda_qimg21_text --prompt p", 0.0)
        event = tracker.feed("timing: 36 layers 3.447 s (13.9 GB streamed)", 1.0)
        self.assertEqual((event["kind"], event["stage"], event["label"], event["seconds"], event["detail"]),
                         ("timing", "encode prompt", "36 layers", 3.447, "13.9 GB streamed"))
        self.assertIsNone(tracker.feed("timing: nonsense", 1.0))

    def test_the_breakdown_groups_phases_and_finds_the_generation_time(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-") as td:
            log = Path(td) / "cuda.log"
            log.write_text(self.LOG)
            rows = timing_breakdown(log, "encode prompt")
        self.assertEqual([(r["stage"], r["label"]) for r in rows], [
            ("encode prompt", "prompt embedding cache hit"),
            ("denoise", "CUDA init + kernels + memory plan"),
            ("denoise", "load transformer weights (6936 MiB)"),
            ("denoise", "denoise 10 steps (image generation)"),
            ("denoise", "stage total"),
            ("encode prompt", "36 layers")])
        self.assertTrue(rows[4]["total"])
        self.assertAlmostEqual(generation_seconds(rows), 3.234)
        self.assertIsNone(generation_seconds(rows[:1]))
        self.assertEqual(timing_breakdown(Path(td) / "missing.log", "x"), [])


class QwenImage21RestartTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="qimg21-test-")
        self.root = Path(self.td.name)
        self.demo = QwenImage21RoutingTest.make_demo(None, self.root)
        self.source = "a" * 32
        work = self.root / "work" / self.source / "cuda-work"
        work.mkdir(parents=True)
        (work / "native_latents.npy").write_bytes(b"x")
        (work / "latents.npy").write_bytes(b"x")
        (work.parent / "request.json").write_text('{"width": 512, "height": 512}')

    def tearDown(self):
        self.td.cleanup()

    def request(self, **extra):
        base = {"prompt": "apple", "backend": "cuda", "mode": "native", "preset": "fast12",
                "width": 512, "height": 512, "steps": 20, "restart_from": self.source}
        base.update(extra)
        return base

    def test_a_refine_continues_the_earlier_run_from_its_noise(self):
        cfg = self.demo._validate(self.request(restart_keep=0.5))
        self.assertEqual(self.demo.restart_step(cfg), 10)
        commands = []
        with mock.patch.object(self.demo, "_run", side_effect=lambda c, *a, **k: commands.append(c)), \
             mock.patch.object(self.demo, "preset_available", return_value=True):
            self.demo._native(cfg, self.root / "out")
        command = commands[0]
        work = self.root / "work" / self.source / "cuda-work"
        self.assertEqual(command[command.index("--initial-latents") + 1], str(work / "latents.npy"))
        self.assertEqual(command[command.index("--restart-from") + 1], str(work / "native_latents.npy"))
        self.assertEqual(command[command.index("--restart-step") + 1], "10")
        # At least one step always runs, whatever keep says.
        self.assertEqual(self.demo.restart_step(self.demo._validate(self.request(steps=4, restart_keep=0.95))), 3)

    def test_a_refine_is_refused_when_it_cannot_continue_the_picture(self):
        bad = [({"restart_from": "../etc"}, "job id"),
               ({"preset": ""}, "fast preset"),
               ({"mode": "compare"}, "fast preset"),
               ({"restart_keep": 0.99}, "between 0 and 0.95"),
               ({"width": 768}, "size"),
               ({"restart_from": "b" * 32}, "gone")]
        for extra, reason in bad:
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, reason):
                    self.demo._validate(self.request(**extra))


if __name__ == "__main__":
    unittest.main()
