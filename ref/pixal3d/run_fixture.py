"""Run the native CLI and record external process/device memory observations.

This harness uses Python only to launch/measure the native executable. Inference
and GLB export run entirely in C/C++; no PyTorch is imported here.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import time
import psutil

ROOT=Path(__file__).resolve().parent.parent.parent
p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['cpu','cuda','rocm'],required=True)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--mask',type=Path)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--binary',type=Path,default=ROOT/'cpu/pixal3d/pixal3d')
p.add_argument('--fov',type=float,required=True)
p.add_argument('--seed',type=int,default=1)
p.add_argument('--threads',type=int,default=16)
p.add_argument('--dump',action='store_true')
p.add_argument('--attach',type=int,help='Monitor an already running native process')
p.add_argument('--gpu-execution',choices=['legacy','resident'],default='legacy')
p.add_argument('--gpu-kernels',choices=['auto','blas','mma'],default='auto')
p.add_argument('--timeout',type=float,default=14400)
a=p.parse_args()
assert a.timeout>0
class Memory(C.Structure):
    _fields_=[('total',C.c_ulonglong),('free',C.c_ulonglong),('used',C.c_ulonglong)]
class Process(C.Structure):
    _fields_=[('pid',C.c_uint),('usedGpuMemory',C.c_ulonglong),('gpuInstanceId',C.c_uint),('computeInstanceId',C.c_uint)]
nvml=None;amd=[];handle=C.c_void_p()
if a.backend=='cuda':
    nvml=C.CDLL('libnvidia-ml.so.1')
    assert nvml.nvmlInit_v2()==0
    assert nvml.nvmlDeviceGetHandleByIndex_v2(0,C.byref(handle))==0
elif a.backend=='rocm':
    for path in Path('/sys/class/drm').glob('card*/device/mem_info_vram_used'):
        if (path.parent/'vendor').read_text().strip()=='0x1002':amd.append(path)
    assert amd,'No AMD VRAM counter found'

def memory(pid):
    used=total=process=0
    if nvml:
        stats=Memory();assert nvml.nvmlDeviceGetMemoryInfo(handle,C.byref(stats))==0
        used,total=stats.used,stats.total
        count=C.c_uint(64);items=(Process*64)()
        if nvml.nvmlDeviceGetComputeRunningProcesses_v3(handle,C.byref(count),items)==0:
            process=sum(item.usedGpuMemory for item in items[:count.value] if item.pid==pid)
    elif amd:
        used=sum(int(path.read_text()) for path in amd)
        total=sum(int(path.with_name('mem_info_vram_total').read_text()) for path in amd)
    return used,total,process

a.output_dir.mkdir(parents=True,exist_ok=True)
command=[str(a.binary),'--backend',a.backend,'--input',str(a.input),'--output',str(a.output_dir/'mesh.glb'),
         '--fov',str(a.fov),'--seed',str(a.seed),'--threads',str(a.threads)]
command+=['--gpu-execution',a.gpu_execution,'--gpu-kernels',a.gpu_kernels,'--profile-json',str(a.output_dir/'profile.json')]
if a.dump:command+=['--dump-dir',str(a.output_dir/'dumps')]
if a.mask:command+=['--mask',str(a.mask)]
if a.attach:
    process=psutil.Process(a.attach)
else:
    stdout=(a.output_dir/'stats.json').open('w');stderr=(a.output_dir/'run.log').open('w')
    child=subprocess.Popen(command,stdout=stdout,stderr=stderr,cwd=ROOT)
    process=psutil.Process(child.pid)
start=time.monotonic();peak_device=peak_process=peak_host=0;baseline,total,_=memory(process.pid);samples=0
with (a.output_dir/'memory.jsonl').open('w') as log:
    while True:
        if not a.attach and time.monotonic()-start>a.timeout:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            break
        try:
            if not process.is_running() or process.status()==psutil.STATUS_ZOMBIE:break
            host=process.memory_info().rss
        except psutil.NoSuchProcess:break
        used,total,own=memory(process.pid)
        peak_device=max(peak_device,used);peak_process=max(peak_process,own);peak_host=max(peak_host,host);samples+=1
        log.write(json.dumps(dict(seconds=round(time.monotonic()-start,2),host_rss=host,device_used=used,process_device_used=own))+'\n');log.flush()
        time.sleep(1)
rc=None if a.attach else child.wait()
summary=dict(backend=a.backend,pid=process.pid,attached=bool(a.attach),command=command,
    exit_code=rc,observed_seconds=time.monotonic()-start,samples=samples,
    baseline_device_used=baseline,peak_device_total_used=peak_device,peak_process_device_used=peak_process,
    peak_host_rss=peak_host,device_capacity=total,
    note='AMD device counter includes other processes; NVML reports both device and process use. Sampling interval: 1 second.')
(a.output_dir/'memory-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
if nvml:nvml.nvmlShutdown()
if rc:raise SystemExit(rc)
