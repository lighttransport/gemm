"""Timing reports distinguish measured capture intervals from cold startup."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hv15n_benchmark", HERE / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
SCRATCH = benchmark.ROOT / "tmp/hv15-native/tests"
SCRATCH.mkdir(parents=True, exist_ok=True)


class BenchmarkTiming(unittest.TestCase):
    def test_cold_step_not_mixed_into_warm_capture_mean(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            root = Path(directory)
            run, captures = root / "run", root / "captures"
            run.mkdir()
            captures.mkdir()
            (run / "manifest.json").write_text(json.dumps({"metrics": {"wall_seconds": 345}}))
            times = dict(noise_input=10000, latent_step_0=10200, latent_step_1=10210,
                         latent_step_2=10221, latent_final=10222, vae_decoded=10300)
            for name, stamp in times.items():
                raw = captures / (name + ".f32")
                raw.touch()
                os.utime(raw, (stamp, stamp))
                if name.startswith("latent_step"):
                    raw.with_suffix(".json").write_text('{}')
            result = benchmark.native_timing(run, captures)
            self.assertEqual(result["warm_step_seconds"]["values"], [10, 11])
            self.assertEqual(result["warm_step_seconds"]["mean"], 10.5)
            self.assertEqual(result["denoising_seconds"], 222)
            self.assertEqual(result["decode_seconds"], 78)
            self.assertEqual(result["generation_and_packaging_seconds"], 345)
            (run / "performance_pause.json").write_text(json.dumps({"intervals": [
                {"started_unix": 10212, "ended_unix": 10215, "duration_seconds": 3}]}))
            adjusted = benchmark.native_timing(run, captures)
            self.assertEqual(adjusted["warm_step_seconds"]["values"], [10, 8])
            self.assertEqual(adjusted["denoising_seconds"], 219)
            self.assertEqual(adjusted["active_generation_and_packaging_seconds"], 342)
            (captures / "latent_step_1.json").unlink()
            with self.assertRaises(ValueError):
                benchmark.native_timing(run, captures)

    def test_invalid_samples_cannot_report_speed(self):
        for values in ([], [0], [-1], [float('nan')], [float('inf')]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                benchmark.statistics_seconds(values)


if __name__ == "__main__":
    unittest.main()
