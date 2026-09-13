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
a=p.parse_args()
lib=C.CDLL(str(a.library));lib.px_test_error.restype=C.c_char_p
if hasattr(lib,'px_test_set_threads'):
    lib.px_test_set_threads.argtypes=[C.c_int]
    assert lib.px_test_set_threads(a.threads)==0
else:assert a.threads==16,'This older validation bridge has a fixed thread count of 16'
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
lib.px_test_flow.argtypes=[C.c_int,C.c_char_p,fp,fp,ip,C.c_int,C.c_int,fp,C.c_int,fp,C.c_int,C.c_float,C.c_int,C.c_int]
data=load_file(a.dump_dir/'shape1024_noise.safetensors');global_cond=load_file(a.dump_dir/'shape1024_global.safetensors')['feats'];projected=load_file(a.dump_dir/'shape1024_projected.safetensors')['feats']
actual=np.empty_like(data['feats']);times=[]
for _ in range(3):
    start=time.monotonic()
    rc=lib.px_test_flow(['cpu','cuda','rocm'].index(a.backend),b'/mnt/disk2/models/Pixal3D/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors',actual,data['feats'],data['coords'],len(actual),32,global_cond,1024,projected,2048,1.,1,1)
    assert rc==0,lib.px_test_error().decode();times.append(time.monotonic()-start)
save_file({'feats':actual},str(a.output))
print(json.dumps(dict(backend=a.backend,library=str(a.library),threads=a.threads,tokens=len(actual),seconds=times,median=float(np.median(times)))),flush=True)
