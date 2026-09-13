"""Byte-exact original Pixal3D preprocessing, including large RGBA and RGB+mask."""
import argparse
import ast
import ctypes as C
from pathlib import Path
import numpy as np
from PIL import Image
from safetensors.numpy import load_file
from upstream_import import ROOT
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path)
p.add_argument('--dump-dir',type=Path)
a=p.parse_args()
source=ROOT/'upstream/pixal3d/pipelines/pixal3d_image_to_3d.py'
tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Pixal3DImageTo3DPipeline')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_image')
scope={'np':np,'Image':Image}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),scope)
lib=C.CDLL(str(ROOT.parent.parent/'cpu/pixal3d/libpixal3d_validation.so'))
lib.px_test_error.restype=C.c_char_p
fp=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
ub=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
lib.px_test_preprocess.argtypes=[ub,C.c_void_p,C.c_int,C.c_int,C.c_int,fp,fp]
rng=np.random.default_rng(823)
fixtures=[]
if a.input:fixtures.append((str(a.input),np.array(Image.open(a.input).convert('RGBA')),False))
else:
    for width,height in [(1473,1027),(331,251),(1024,1003)]:
        image=rng.integers(0,256,(height,width,4),dtype=np.uint8)
        yy,xx=np.indices((height,width));distance=((xx-width*.4)/(width*.3))**2+((yy-height*.6)/(height*.3))**2
        image[:,:,3]=np.clip((1.05-distance)*2550,0,255).astype(np.uint8)
        fixtures.append((f'{width}x{height} RGBA',image,False))
        fixtures.append((f'{width}x{height} RGB+mask',image,True))
for name,rgba,rgb_mask in fixtures:
    expected_image=scope['preprocess_image'](None,Image.fromarray(rgba))
    if a.dump_dir:
        actuals=[load_file(a.dump_dir/f'image{size}.safetensors')['feats'].reshape(size,size,3) for size in (512,1024)]
    else:
        pixels=np.ascontiguousarray(rgba[:,:,:3] if rgb_mask else rgba)
        mask=np.ascontiguousarray(rgba[:,:,3]) if rgb_mask else None
        actuals=[np.empty((size,size,3),np.float32) for size in (512,1024)]
        rc=lib.px_test_preprocess(pixels,mask.ctypes.data if mask is not None else None,
            pixels.shape[1],pixels.shape[0],pixels.shape[2],*actuals)
        assert rc==0,lib.px_test_error().decode()
    for size,actual in zip((512,1024),actuals):
        expected=np.asarray(expected_image.resize((size,size),Image.Resampling.LANCZOS)).astype(np.float32)/255
        print(dict(fixture=name,size=size,max_abs=float(abs(expected-actual).max()),different=int((expected!=actual).sum())),flush=True)
        np.testing.assert_array_equal(actual,expected)
print('Exact preprocessing PASS',flush=True)
