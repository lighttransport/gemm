"""Timing reports distinguish measured capture intervals from cold startup."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import call,patch

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

    def test_stalled_gpu_monitor_resumes_baseline(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            root=Path(directory);lock=root/'gpu.lock';lock.touch()
            stat=lock.stat()
            lock_id=f'{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}'
            native=dict(state='T',parent=20,started='123',command=['hv15n','--out-dir',str(root/'frames')])
            def record(pid):return native if pid==100 else dict(parent=30)
            monitor=['nvidia-smi','pmon','-c','1','-s','u']
            with patch.object(benchmark,'LOCK',lock),patch.object(benchmark,'proc_record',record),patch.object(Path,'read_text',return_value=f'1: FLOCK ADVISORY WRITE 30 {lock_id} 0 EOF\n'),patch.object(benchmark.os,'kill') as kill,patch.object(benchmark.subprocess,'check_output',side_effect=subprocess.TimeoutExpired(monitor,2)) as sample,patch.object(benchmark,'atomic_json'):
                with self.assertRaises(subprocess.TimeoutExpired):
                    with benchmark.gpu_reservation(100,root,{},lambda:None,threading.Event()):
                        self.fail('GPU reservation entered after stalled monitor')
                sample.assert_called_once_with(monitor,text=True,timeout=2)
                self.assertEqual(kill.call_args_list,[call(100,signal.SIGSTOP),call(100,signal.SIGCONT)])


if __name__ == "__main__":
    unittest.main()
