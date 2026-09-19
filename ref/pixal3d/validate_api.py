"""Fast native C ABI and CLI failure-path checks; no model inference is run."""
import ctypes as C
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
class Options(C.Structure):
    _fields_=[('backend',C.c_int),('device',C.c_int),('threads',C.c_int),('vram_budget_mib',C.c_size_t),
              ('model_dir',C.c_char_p),('dinov3_path',C.c_char_p),('naf_path',C.c_char_p),('dump_dir',C.c_char_p),
              ('seed',C.c_uint32),('texture_size',C.c_int),('decimation_target',C.c_int)]
class Camera(C.Structure):
    _fields_=[('fov',C.c_float),('distance',C.c_float),('mesh_scale',C.c_float)]
class Image(C.Structure):
    _fields_=[('pixels',C.c_void_p),('mask',C.c_void_p),('width',C.c_int),('height',C.c_int),('channels',C.c_int)]
class Stats(C.Structure):
    _fields_=[('elapsed_seconds',C.c_double),('peak_device_bytes',C.c_size_t),('peak_host_bytes',C.c_size_t),
              ('shape_tokens',C.c_int),('vertices',C.c_int),('triangles',C.c_int)]
class Result(C.Structure):
    _fields_=[(name,C.c_void_p) for name in ['vertices','normals','uvs','triangles','base_color_rgba','metallic_roughness_rgb']]+[
        ('vertex_count',C.c_int),('triangle_count',C.c_int),('texture_size',C.c_int),('stats',Stats)]
lib=C.CDLL(str(ROOT/'cpu/pixal3d/libpixal3d.so'))
lib.pixal3d_default_options.argtypes=[C.POINTER(Options)]
lib.pixal3d_create.argtypes=[C.POINTER(Options)];lib.pixal3d_create.restype=C.c_void_p
lib.pixal3d_last_error.argtypes=[C.c_void_p];lib.pixal3d_last_error.restype=C.c_char_p
lib.pixal3d_destroy.argtypes=[C.c_void_p]
lib.pixal3d_generate.argtypes=[C.c_void_p,C.POINTER(Image),C.POINTER(Camera),C.POINTER(Result)]
lib.pixal3d_result_free.argtypes=[C.POINTER(Result)]
lib.pixal3d_write_glb.argtypes=[C.c_char_p,C.POINTER(Result)]
for field,value in [('threads',0),('device',-1),('backend',3),('vram_budget_mib',14337),('texture_size',1536),('decimation_target',9999),('model_dir',b'/missing/pixal3d')]:
    options=Options();lib.pixal3d_default_options(C.byref(options));setattr(options,field,value)
    context=lib.pixal3d_create(C.byref(options))
    assert not context,field
    assert lib.pixal3d_last_error(None),field
options=Options();lib.pixal3d_default_options(C.byref(options));options.threads=1
context=lib.pixal3d_create(C.byref(options));assert context,lib.pixal3d_last_error(None)
try:
    assert not lib.pixal3d_last_error(None)
    pixels=(C.c_uint8*(16*16*4))(*([255]*16*16*4))
    image=Image(C.cast(pixels,C.c_void_p),None,16,16,4)
    for camera in [Camera(float('nan'),0,1),Camera(.85,float('inf'),1),Camera(.85,0,0)]:
        result=Result()
        assert lib.pixal3d_generate(context,C.byref(image),C.byref(camera),C.byref(result))==-1
        assert lib.pixal3d_last_error(context)
        assert not result.vertices and result.vertex_count==0
        lib.pixal3d_result_free(C.byref(result))
    camera=Camera(.85,0,1)
    assert lib.pixal3d_generate(context,None,C.byref(camera),None)==-1
    assert b'Missing generation' in lib.pixal3d_last_error(context)
    image.channels=3
    result=Result()
    assert lib.pixal3d_generate(context,C.byref(image),C.byref(camera),C.byref(result))==-1
    assert b'mask' in lib.pixal3d_last_error(context)
    assert lib.pixal3d_write_glb(None,C.byref(result))==-1
finally:lib.pixal3d_destroy(context)
base=[str(ROOT/'cpu/pixal3d/pixal3d'),'--input','unused.png','--output',str(ROOT/'tmp/pixal3d/unused.glb'),'--fov','.85']
for args in [[],['--backend','invalid'],['--seed','-1'],['--seed','4294967296'],['--threads','2x'],['--fov','nan'],['--distance','inf'],['--fov','.8junk'],['--texture-size','1536'],['--triangle-target','9999'],['--device']]:
    command=[base[0]] if not args else base+args
    run=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert run.returncode and 'Pixal3D:' in run.stderr,(args,run)
    assert 'Cannot load input' not in run.stderr,args
print('Native C API and CLI error handling: PASS',flush=True)
