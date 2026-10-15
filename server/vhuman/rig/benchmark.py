"""Repeatable native CUDA latency baseline, with CPU parity and device telemetry.

Run separately from playback_benchmark; concurrent GPU jobs invalidate isolation.
No Torch import is needed. Output contains the raw timing and NVML samples.
"""
from __future__ import annotations

import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

from . import safetensors
from .native import GPU_SOURCES, ROOT, NativeGPU, build_gpu_library


def receipt(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path.absolute()), bytes=path.stat().st_size, sha256=digest.hexdigest())


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('empty or nonfinite timing samples')
    return dict(samples=len(values), mean=float(values.mean()),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)), max=float(values.max()))


class Telemetry:
    """Sample total device memory, including the desktop, every 100 ms via NVML."""
    def __init__(self, device=0):
        self.lib = C.CDLL('libnvidia-ml.so.1')
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [C.c_uint, C.POINTER(C.c_void_p)]
        self.check(self.lib.nvmlInit_v2())
        self.handle = C.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(device, C.byref(self.handle)))
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'NVML error {code}')

    def sample(self):
        class Memory(C.Structure):
            _fields_ = [('version', C.c_uint), ('total', C.c_ulonglong), ('reserved', C.c_ulonglong),
                       ('free', C.c_ulonglong), ('used', C.c_ulonglong)]
        class Util(C.Structure):
            _fields_ = [('gpu', C.c_uint), ('memory', C.c_uint)]
        memory, util = Memory(), Util()
        memory.version = C.sizeof(Memory) | (2 << 24)
        self.check(self.lib.nvmlDeviceGetMemoryInfo_v2(self.handle, C.byref(memory)))
        self.check(self.lib.nvmlDeviceGetUtilizationRates(self.handle, C.byref(util)))
        result = dict(time=time.time(), memory_used_bytes=memory.used, memory_total_bytes=memory.total,
                      memory_reserved_bytes=memory.reserved,
                      gpu_percent=util.gpu)
        for key, function, args in (
            ('temperature_c', 'nvmlDeviceGetTemperature', [C.c_uint(0)]),
            ('graphics_mhz', 'nvmlDeviceGetClockInfo', [C.c_uint(0)]),
            ('memory_mhz', 'nvmlDeviceGetClockInfo', [C.c_uint(2)]),
            ('power_mw', 'nvmlDeviceGetPowerUsage', []),
        ):
            value = C.c_uint()
            code = getattr(self.lib, function)(self.handle, *args, C.byref(value))
            result[key] = value.value if not code else None
        return result

    def __enter__(self):
        self.samples.append(self.sample())
        self.error = None
        def poll():
            try:
                while not self.stop.wait(.1):
                    self.samples.append(self.sample())
            except Exception as exc:
                self.error = str(exc)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.samples.append(self.sample())
        self.check(self.lib.nvmlShutdown())

    def report(self):
        if self.error:
            raise RuntimeError(self.error)
        peak = max(s['memory_used_bytes'] for s in self.samples)
        return dict(scope='NVML v2 allocated whole-device memory, includes desktop, excludes driver/firmware reserved; sampled at 100 ms, not allocator high-water',
                    idle_bytes=self.samples[0]['memory_used_bytes'], peak_bytes=peak,
                    peak_minus_idle_bytes=peak-self.samples[0]['memory_used_bytes'],
                    final_bytes=self.samples[-1]['memory_used_bytes'], samples=self.samples)


def machine():
    return dict(utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                python=platform.python_version(), numpy=np.__version__,
                cpu=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                          if line.startswith('model name')), platform.processor()),
                gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,compute_cap,memory.total,power.limit',
                                             '--format=csv'], text=True).strip(),
                compiler=subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0],
                git=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())


def run(package, out, seconds=3., warmup=16, batches=(1, 32), seed=1, device=0):
    package, out = Path(package), Path(out)
    if out.exists():
        raise FileExistsError(f'refusing to overwrite baseline: {out}')
    if not np.isfinite(seconds) or seconds <= 0 or warmup < 1 or not batches or any(f < 1 or f > 1024 for f in batches):
        raise ValueError('positive duration/warmup and batch sizes 1..1024 required')
    out.mkdir(parents=True)
    lib = build_gpu_library(out/'native', backend='cuda')
    tensors, _ = safetensors.load(package)
    limits = tensors['controls.range'].copy()
    del tensors
    rng = np.random.default_rng(seed)
    controls = np.ascontiguousarray(rng.uniform(limits[:, 0], limits[:, 1], (256, len(limits))) *
                                   (rng.random((256, len(limits))) < .1), dtype=np.float32)
    np.save(out/'controls.npy', controls, allow_pickle=False)
    report = dict(machine=machine(), package=receipt(package), library=receipt(lib),
                  sources=[receipt(ROOT/p) for p in (*GPU_SOURCES, 'cuda/vhuman/vhuman_deformer_cuda.h',
                      'ryzen/vhuman_deformer.h', 'ryzen/lightrig_mlp2.h', 'cuda/cuda_runner_common.h',
                      'cuda/cuew.h', 'common/safetensors.h')], harness=receipt(Path(__file__)),
                  inputs=receipt(out/'controls.npy'),
                  configuration=dict(seconds_per_case=seconds, warmup=warmup, batches=list(batches), seed=seed,
                                     compiler_flags='-O3 -mavx2 -mfma -fPIC -shared',
                                     wall_scope='preallocated ctypes call, includes host rig, uploads, synchronization and optional readback; excludes rendering'),
                  cases=[])
    with Telemetry(device) as telemetry:
        start = time.perf_counter()
        native = NativeGPU(lib, package, device=device)
        report['load_and_nvrtc_ms'] = (time.perf_counter()-start)*1000
        try:
            version = C.CFUNCTYPE(C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int))(
                C.c_void_p.in_dll(native.lib, 'nvrtcVersion').value)
            major, minor = C.c_int(), C.c_int()
            if version(C.byref(major), C.byref(minor)):
                raise RuntimeError('cannot query NVRTC version')
            report['nvrtc'] = dict(version=f'{major.value}.{minor.value}',
                libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                  if '/libnvrtc' in line}))
            report['vertices'], report['morphs'] = native.V, native.M
            report['has_contacts'] = native.has_contacts
            modes = ([(True, True)] if native.has_contacts else []) + [(True, False), (False, False)]
            for ml, contacts in modes:
                native.set_contact_iterations(4 if contacts else 0)
                cpu_ms = []
                cpu_out = np.empty((native.V, 3), dtype=np.float32)
                cpu_op = native._p(cpu_out)
                for x in controls[:16]:
                    expected = native.eval(x, use_ml=ml)
                    actual, _ = native.eval_gpu(x[None], use_ml=ml)
                    if not np.isfinite(expected).all() or not np.isfinite(actual).all() or np.max(np.abs(actual[0]-expected)) > 1e-4:
                        raise RuntimeError('single-frame CPU/CUDA parity failure')
                for x in controls[:64]:
                    xp = native._p(x)
                    start = time.perf_counter()
                    native.lib.vh_deformer_eval(native.h, xp, int(ml), cpu_op)
                    cpu_ms.append((time.perf_counter()-start)*1000)
                for frames in batches:
                    x = np.ascontiguousarray(controls[np.arange(frames) % len(controls)])
                    output = np.empty((frames, native.V, 3), dtype=np.float32)
                    xp, op = native._p(x), native._p(output)
                    times = (C.c_double * 4)()
                    # Full-batch correctness is checked outside the timed loop.
                    if native.lib.vh_gpu_eval_batch(native.g, xp, frames, int(ml), op, times):
                        raise RuntimeError('CUDA evaluation failed')
                    error = 0.
                    for f in range(frames):
                        ref = native.eval(x[f], use_ml=ml)
                        if not np.isfinite(ref).all():
                            raise RuntimeError('nonfinite CPU reference')
                        error = max(error, float(np.max(np.abs(ref-output[f]))))
                    if not np.isfinite(output).all() or error > 1e-4:
                        raise RuntimeError(f'batch CPU/CUDA parity failure: {error}')
                    for readback in (True, False):
                        target = op if readback else None
                        for _ in range(warmup):
                            if native.lib.vh_gpu_eval_batch(native.g, xp, frames, int(ml), target, times):
                                raise RuntimeError('CUDA warmup failed')
                        raw = []
                        start = time.perf_counter()
                        while time.perf_counter()-start < seconds or len(raw) < 32:
                            t0 = time.perf_counter()
                            code = native.lib.vh_gpu_eval_batch(native.g, xp, frames, int(ml), target, times)
                            wall_ms = (time.perf_counter()-t0)*1000
                            if code:
                                raise RuntimeError('CUDA timed call failed')
                            raw.append([wall_ms, *times])
                        values = np.asarray(raw)
                        case = dict(ml=ml, contacts=contacts, frames=frames, readback=readback,
                                    max_cpu_error_m=error, cpu_single_call_ms=distribution(cpu_ms),
                                    timing_ms={name: distribution(values[:, i]) for i, name in enumerate(
                                        ('wall', 'host_rig', 'upload_and_wait', 'gpu_kernel', 'download'))}, raw_ms=raw)
                        report['cases'].append(case)
                        telemetry.samples.append(telemetry.sample())
                        print(f'ml={int(ml)} contacts={int(contacts)} frames={frames} readback={int(readback)} '
                              f'wall p50/p95={case["timing_ms"]["wall"]["p50"]:.4f}/{case["timing_ms"]["wall"]["p95"]:.4f} ms', flush=True)
        finally:
            native.close()
        time.sleep(.3)
    report['telemetry'] = telemetry.report()
    report['passed'] = True
    (out/'baseline.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=float, default=3)
    p.add_argument('--warmup', type=int, default=16)
    p.add_argument('--batches', nargs='+', type=int, default=[1, 32])
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()
    run(**vars(args))


if __name__ == '__main__':
    main()
