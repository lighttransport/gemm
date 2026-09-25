"""Replay saved denormalized latents through the native shape and texture decoders.

Development utility for decoder memory and speed work. It runs the shape decoder
and the guided texture decoder in pipeline order, applies the pipeline's texture
range mapping, and compares against saved decoded dumps when present.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
from safetensors.numpy import load_file, save_file
import numpy as np
root=Path(__file__).resolve().parent.parent.parent
p=argparse.ArgumentParser()
p.add_argument('--dump-dir',type=Path,required=True,help='Directory with shape/texture_denormalized dumps')
p.add_argument('--model-dir',type=Path,default=Path('/mnt/disk2/models/Pixal3D'))
p.add_argument('--vram-budget-mib',type=int,default=12288)
p.add_argument('--threads',type=int,default=8)
p.add_argument('--profile-json',type=Path,required=True)
p.add_argument('--expect-dir',type=Path,help='Compare with shape/texture_decoded dumps in this directory')
p.add_argument('--save-dir',type=Path,help='Write shape/texture_decoded dumps here')
a=p.parse_args()
shape=load_file(a.dump_dir/'shape_denormalized.safetensors')
texture=load_file(a.dump_dir/'texture_denormalized.safetensors')
np.testing.assert_array_equal(shape['coords'],texture['coords'])
coords=np.ascontiguousarray(shape['coords'],dtype=np.int32)
n=len(coords)
lib=C.CDLL(str(root/'cpu/pixal3d/libpixal3d_validation.so'))
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
FP=C.POINTER(C.c_float)
IP=C.POINTER(C.c_int32)
lib.px_test_error.restype=C.c_char_p
lib.px_test_set_gpu.argtypes=[C.c_int,C.c_int]
lib.px_test_set_gpu_flow_precision.argtypes=[C.c_int]
lib.px_test_set_vram_budget.argtypes=[C.c_size_t]
lib.px_test_set_threads.argtypes=[C.c_int]
lib.px_test_set_profile_json.argtypes=[C.c_char_p]
lib.px_test_free.argtypes=[C.c_void_p]
lib.px_test_decode_pair.argtypes=[C.c_int,C.c_char_p,C.c_char_p,fp,fp,ip,C.c_int,
                                  C.POINTER(FP),C.POINTER(FP),C.POINTER(IP),C.POINTER(C.c_int)]
assert lib.px_test_set_gpu(1,0)==0
assert lib.px_test_set_gpu_flow_precision(2)==0
assert lib.px_test_set_vram_budget(a.vram_budget_mib)==0
assert lib.px_test_set_threads(a.threads)==0
a.profile_json.parent.mkdir(parents=True,exist_ok=True)
assert lib.px_test_set_profile_json(str(a.profile_json).encode())==0
ckpts=a.model_dir/'ckpts'
shape_y,texture_y,out_coords,rows=FP(),FP(),IP(),C.c_int()
start=time.monotonic()
rc=lib.px_test_decode_pair(1,str(ckpts/'shape_dec_next_dc_f16c32_fp16.safetensors').encode(),
                           str(ckpts/'tex_dec_next_dc_f16c32_fp16.safetensors').encode(),
                           np.ascontiguousarray(shape['feats'],dtype=np.float32),
                           np.ascontiguousarray(texture['feats'],dtype=np.float32),coords,n,
                           C.byref(shape_y),C.byref(texture_y),C.byref(out_coords),C.byref(rows))
elapsed=time.monotonic()-start
assert rc==0,lib.px_test_error().decode()
m=rows.value
decoded={
    'shape':{'feats':np.ctypeslib.as_array(shape_y,(m,7)).copy(),
             'coords':np.ctypeslib.as_array(out_coords,(m,4)).copy()},
    'texture':{'feats':np.clip(np.ctypeslib.as_array(texture_y,(m,6))*np.float32(.5)+np.float32(.5),0,1),
               'coords':np.ctypeslib.as_array(out_coords,(m,4)).copy()},
}
for ptr in (shape_y,texture_y,out_coords):lib.px_test_free(C.cast(ptr,C.c_void_p))
profile=json.loads(a.profile_json.read_text())
report={'input_rows':n,'output_rows':m,'seconds':round(elapsed,3),
        'decoder_seconds':{k:v for k,v in profile['timings_seconds'].items() if k.endswith('.decoder')},
        'h2d_bytes':profile['h2d_bytes'],'peak_reserved_device_bytes':profile['peak_reserved_device_bytes'],
        'effective_budget_bytes':profile.get('effective_budget_bytes')}
for name,d in decoded.items():
    report[name+'_sha256']=hashlib.sha256(d['feats'].tobytes()+d['coords'].tobytes()).hexdigest()
    if a.save_dir:
        a.save_dir.mkdir(parents=True,exist_ok=True)
        save_file(d,str(a.save_dir/f'{name}_decoded.safetensors'))
    if a.expect_dir:
        e=load_file(a.expect_dir/f'{name}_decoded.safetensors')
        same_coords=e['coords'].shape==d['coords'].shape and np.array_equal(e['coords'],d['coords'])
        report[name+'_identical']=bool(same_coords and np.array_equal(e['feats'],d['feats']))
        if same_coords:report[name+'_max_abs']=float(np.abs(e['feats']-d['feats']).max())
print(json.dumps(report,indent=1),flush=True)
if a.expect_dir and not (report['shape_identical'] and report['texture_identical']):raise SystemExit(1)
