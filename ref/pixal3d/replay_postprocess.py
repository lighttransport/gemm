"""Replay saved native decoder outputs through the native mesh/PBR exporter.

Reference/development utility, useful when iterating on mesh processing without
repeating diffusion. All geometry, texture baking and export execute in C++.
"""
import argparse
import ctypes as C
from pathlib import Path
import time
from safetensors.numpy import load_file
import numpy as np
root=Path(__file__).resolve().parent.parent.parent
p=argparse.ArgumentParser()
p.add_argument('--dump-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
shape=load_file(a.dump_dir/'shape_decoded.safetensors')
texture=load_file(a.dump_dir/'texture_decoded.safetensors')
np.testing.assert_array_equal(shape['coords'],texture['coords'])
lib=C.CDLL(str(root/'cpu/pixal3d/libpixal3d_validation.so'))
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
lib.px_test_postprocess_dump.argtypes=[ip,fp,fp,C.c_int,C.c_char_p,C.c_char_p]
lib.px_test_error.restype=C.c_char_p
a.output.parent.mkdir(parents=True,exist_ok=True)
start=time.monotonic()
rc=lib.px_test_postprocess_dump(shape['coords'],shape['feats'],texture['feats'],len(shape['coords']),str(a.output).encode(),str(a.dump_dir).encode())
assert rc==0,lib.px_test_error().decode()
print(f'Native postprocess replay: {time.monotonic()-start:.3f}s; {a.output}',flush=True)
