"""Run one owned GPU experiment with a hard 55-second wall-clock budget."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ref/hunyuan_video15_native'))
from benchmark import gpu_reservation
from cuda.hunyuan_video15_native.generate import atomic_json, digest, MemorySampler


@contextmanager
def experiment_lock(cancel):
    path=ROOT/'tmp/hv15-native/short-run.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as file:
        while True:
            try:
                fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if cancel.wait(.05):raise TimeoutError('deadline waiting for another short experiment')
        try:yield
        finally:fcntl.flock(file,fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native-pid', type=int)
    parser.add_argument('--native-run', type=Path, default=ROOT / 'tmp/hv15-native/full-quality-i2v-v1')
    parser.add_argument('--seconds', type=float, default=55)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 55 or not args.command:
        parser.error('a command and budget of 1..55 seconds are required')
    command = args.command[1:] if args.command[0] == '--' else args.command
    if not command:parser.error('a command is required after --')
    args.out = args.out.resolve()
    args.native_run = args.native_run.resolve()
    if args.out.exists():
        raise ValueError('experiment receipt must be new')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.out.parent / 'scratch'
    scratch.mkdir(exist_ok=True)
    env = dict(os.environ, TMPDIR=str(scratch), CUDA_CACHE_PATH=str(scratch / 'cuda'),
               TORCHINDUCTOR_CACHE_DIR=str(scratch / 'inductor'), TRITON_CACHE_DIR=str(scratch / 'triton'),
               PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='16')
    cancel = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda unused, unused_frame: cancel.set())
    started = time.monotonic()
    timer = threading.Timer(args.seconds - .75, cancel.set)
    timer.start()
    report = dict(schema='hv15n.short_experiment.v1', status='running', command=command,
                  budget_seconds=args.seconds, full_pipeline_acceptance=False,
                  launcher_sha256=digest(Path(__file__)),
                  executable_sha256=digest(Path(command[0])) if Path(command[0]).is_file() else None,
                  shared_source_sha256={name:digest(ROOT/name) for name in ('cuda/gemm/cuda_gemm_ptx_kernels.h','cuda/fa2/cuda_fa2_kernels.h')},
                  source_sha256={str(file.relative_to(ROOT)):digest(file) for directory in ('cuda/hunyuan_video15_native','ref/hunyuan_video15_native') for file in (ROOT/directory).iterdir() if file.suffix in ('.cpp','.hpp','.h','.py')})
    def write():
        atomic_json(args.out, report)
    write()
    process = None
    sampler = MemorySampler()
    try:
        with experiment_lock(cancel),gpu_reservation(args.native_pid, args.native_run, report, write, cancel):
            try:
                with args.out.with_suffix('.log').open('w') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                    sampler.start(process.pid)
                    while process.poll() is None:
                        if cancel.wait(.05):
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=.5)
                            raise TimeoutError('experiment cancelled or exceeded wall-clock budget')
                    report.update(returncode=process.returncode, status='pass' if process.returncode == 0 else 'failed')
            finally:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=.5)
    except BaseException as error:
        report.update(status='timeout' if isinstance(error, TimeoutError) else 'failed', error=str(error))
    finally:
        # Terminate before releasing/resuming the reservation even on errors.
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=.5)
        sampler.close()
        timer.cancel()
        report.update(elapsed_seconds=time.monotonic() - started,
                      sampled_peak_vram_mib=sampler.vram, sampled_peak_host_rss_mib=sampler.rss)
        if report['elapsed_seconds'] >= args.seconds:
            report['status'] = 'timeout'
        if sampler.vram is not None and sampler.vram > 14336:
            report.update(status='failed',error='sampled process VRAM exceeds 14336 MiB')
        write()
    print(json.dumps({k:report[k] for k in ('status','elapsed_seconds','sampled_peak_vram_mib','full_pipeline_acceptance')}))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
