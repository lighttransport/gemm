"""Exercise the public versioned GPU configuration ABI without a GPU run.

Requires the normal local model paths for context creation; weights are not
loaded into memory. Run through ref/pixal3d/run.sh cpu.
"""
import ctypes as C
from pathlib import Path


class Options(C.Structure):
    _fields_ = [(name, C.c_int) for name in ('backend', 'device', 'threads')] + [
        ('vram_budget_mib', C.c_size_t),
        ('model_dir', C.c_char_p), ('dinov3_path', C.c_char_p),
        ('naf_path', C.c_char_p), ('dump_dir', C.c_char_p),
        ('seed', C.c_uint32), ('texture_size', C.c_int), ('decimation_target', C.c_int),
    ]


class GpuOptions(C.Structure):
    _fields_ = [('struct_size', C.c_size_t), ('version', C.c_uint32),
                ('execution', C.c_int), ('kernels', C.c_int), ('profile_json', C.c_char_p)]


root = Path(__file__).resolve().parents[2]
lib = C.CDLL(str(root / 'cpu/pixal3d/libpixal3d.so'))
lib.pixal3d_default_options.argtypes = [C.POINTER(Options)]
lib.pixal3d_default_gpu_options.argtypes = [C.POINTER(GpuOptions)]
lib.pixal3d_create.argtypes = [C.POINTER(Options)]
lib.pixal3d_create.restype = C.c_void_p
lib.pixal3d_destroy.argtypes = [C.c_void_p]
lib.pixal3d_configure_gpu.argtypes = [C.c_void_p, C.POINTER(GpuOptions)]
lib.pixal3d_last_error.argtypes = [C.c_void_p]
lib.pixal3d_last_error.restype = C.c_char_p
options = Options()
lib.pixal3d_default_options(C.byref(options))
context = lib.pixal3d_create(C.byref(options))
assert context, lib.pixal3d_last_error(None)
try:
    gpu = GpuOptions()
    lib.pixal3d_default_gpu_options(C.byref(gpu))
    assert gpu.struct_size == C.sizeof(gpu) and gpu.version == 1
    assert lib.pixal3d_configure_gpu(context, C.byref(gpu)) == 0
    for field, value, message in [('version', 2, b'version'), ('struct_size', 8, b'version'),
                                  ('kernels', 99, b'Invalid'), ('execution', 1, b'requires')]:
        lib.pixal3d_default_gpu_options(C.byref(gpu))
        setattr(gpu, field, value)
        assert lib.pixal3d_configure_gpu(context, C.byref(gpu)) == -1
        assert message in lib.pixal3d_last_error(context)
    lib.pixal3d_default_gpu_options(C.byref(gpu))
    assert lib.pixal3d_configure_gpu(context, C.byref(gpu)) == 0
    print('Versioned GPU C API: PASS')
finally:
    lib.pixal3d_destroy(context)
