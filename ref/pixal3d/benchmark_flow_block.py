"""Measure a native flow block using recorded real conditioning/noise."""
import argparse
import ctypes as C
import json
from pathlib import Path
import time
import numpy as np
from safetensors.numpy import load_file,save_file
root=Path(__file__).resolve().parent.parent.parent
p=argparse.ArgumentParser()
p.add_argument('--library',type=Path,default=root/'cpu/pixal3d/libpixal3d_validation.so')
p.add_argument('--backend',choices=['cpu','cuda','rocm'],required=True)
p.add_argument('--dump-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--threads',type=int,default=16)
p.add_argument('--stage',choices=['structure','shape512','shape1024','texture'],default='shape1024')
p.add_argument('--model-dir',type=Path,default=Path('/mnt/disk2/models/Pixal3D'))
p.add_argument('--blocks',type=int,default=1)
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--gpu-execution',choices=['legacy','resident'],default='legacy')
p.add_argument('--gpu-kernels',choices=['auto','blas','mma'],default='auto')
p.add_argument('--gpu-flow-precision',choices=['bf16','fp32','mixed'],default='bf16')
p.add_argument('--vram-budget-mib',type=int,default=12288)
p.add_argument('--tokens',type=int,default=0,help='Limit recorded token count for bounded tuning')
p.add_argument('--check-cache',action='store_true',help='Check alternating guidance, changed content and restored content on the same engine')
a=p.parse_args()
assert 1<=a.blocks<=30 and a.repeats>0 and a.tokens>=0 and 512<a.vram_budget_mib<=14336
lib=C.CDLL(str(a.library));lib.px_test_error.restype=C.c_char_p
if hasattr(lib,'px_test_set_threads'):
    lib.px_test_set_threads.argtypes=[C.c_int]
    assert lib.px_test_set_threads(a.threads)==0
else:assert a.threads==16,'This older validation bridge has a fixed thread count of 16'
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
lib.px_test_set_gpu.argtypes=[C.c_int,C.c_int]
assert lib.px_test_set_gpu(int(a.gpu_execution=='resident'),['auto','blas','mma'].index(a.gpu_kernels))==0
lib.px_test_set_vram_budget.argtypes=[C.c_size_t]
assert lib.px_test_set_vram_budget(a.vram_budget_mib)==0
if hasattr(lib,'px_test_set_gpu_flow_precision'):
    lib.px_test_set_gpu_flow_precision.argtypes=[C.c_int]
    assert lib.px_test_set_gpu_flow_precision(['bf16','fp32','mixed'].index(a.gpu_flow_precision))==0
lib.px_test_flow_open.argtypes=[C.c_int,C.c_char_p];lib.px_test_flow_open.restype=C.c_void_p
lib.px_test_flow_close.argtypes=[C.c_void_p]
lib.px_test_flow_run.argtypes=[C.c_void_p,fp,fp,ip,C.c_int,C.c_int,fp,C.c_int,fp,C.c_int,C.c_float,C.c_int,C.c_int]
stems={'structure':'ss_flow_img_dit_1_3B_64_bf16','shape512':'slat_flow_img2shape_dit_1_3B_512_bf16',
       'shape1024':'slat_flow_img2shape_dit_1_3B_1024_bf16','texture':'slat_flow_imgshape2tex_dit_1_3B_1024_bf16'}
data=load_file(a.dump_dir/f'{a.stage}_noise.safetensors')
global_cond=load_file(a.dump_dir/f'{a.stage}_global.safetensors')['feats']
projected=load_file(a.dump_dir/f'{a.stage}_projected.safetensors')['feats']
x=data['feats'];coords=data['coords']
if a.stage=='texture':
    shape=load_file(a.dump_dir/'shape1024_step_12.safetensors')
    np.testing.assert_array_equal(coords,shape['coords'])
    x=np.concatenate([x,shape['feats']],axis=1)
if a.tokens:x=x[:a.tokens].copy();coords=coords[:a.tokens].copy();projected=projected[:a.tokens].copy()
actual=np.empty((len(x),data['feats'].shape[1]),np.float32);times=[]
start=time.monotonic()
session=lib.px_test_flow_open(['cpu','cuda','rocm'].index(a.backend),str(a.model_dir/'ckpts'/f'{stems[a.stage]}.safetensors').encode())
assert session,lib.px_test_error().decode()
load_seconds=time.monotonic()-start
try:
    for _ in range(a.repeats+1):
        start=time.monotonic()
        rc=lib.px_test_flow_run(session,actual,x,coords,len(x),x.shape[1],global_cond,global_cond.shape[1],projected,projected.shape[1],1.,a.blocks,1)
        if rc != 0:
            raise RuntimeError(lib.px_test_error().decode())
        times.append(time.monotonic()-start)
    if a.check_cache:
        saved=actual.copy();g=global_cond.copy();proj=projected.copy()
        for changed in [True,False]:
            if changed:g.fill(0);proj.fill(0)
            else:g[:]=global_cond;proj[:]=projected
            rc=lib.px_test_flow_run(session,actual,x,coords,len(x),x.shape[1],g,g.shape[1],proj,proj.shape[1],1.,a.blocks,1)
            assert rc==0,lib.px_test_error().decode()
            if changed:assert not np.array_equal(actual,saved), "Changed conditioning must change the prediction"
        np.testing.assert_array_equal(actual,saved)
        print('Conditioning cache alternating-content check PASS',flush=True)
finally:lib.px_test_flow_close(session)
a.output.parent.mkdir(parents=True,exist_ok=True)
save_file({'feats':actual},str(a.output))
report=dict(backend=a.backend,execution=a.gpu_execution,kernels=a.gpu_kernels,stage=a.stage,blocks=a.blocks,
    threads=a.threads,tokens=len(actual),vram_budget_mib=a.vram_budget_mib,load_seconds=load_seconds,
    cold_seconds=times[0],warm_seconds=times[1:],median=float(np.median(times[1:])))
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
