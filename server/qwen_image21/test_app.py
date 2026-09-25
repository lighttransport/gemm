import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

from server.qwen_image21.app import MAX_EVENTS, Demo, Progress, ROOT
import time


class QwenImage21RoutingTest(unittest.TestCase):
    def make_demo(self, root: Path) -> Demo:
        return Demo(root / "model", root / "quant", root / "cuda-python",
                    root / "work", root / "cuda-native", "127.0.0.1", 0,
                    native_rocm=root / "rocm-native",
                    python_rocm=root / "rocm-python")

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


if __name__ == "__main__":
    unittest.main()
