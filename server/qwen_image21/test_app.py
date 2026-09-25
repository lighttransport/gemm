import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

from server.qwen_image21.app import Demo, ROOT


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
            with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None: commands.append(command)):
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
            with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None: commands.append(command)):
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
            with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None: commands.append(command)):
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
            with mock.patch.object(demo, "_run", side_effect=lambda command, cwd, log, env=None: commands.append(command)):
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
                               side_effect=lambda command, cwd, log, env=None: commands.append(command)):
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


if __name__ == "__main__":
    unittest.main()
