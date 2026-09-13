# SPDX-License-Identifier: MIT
"""Compile native GPU kernels through RTC without requiring a GPU device."""
import ctypes as C
import ctypes.util
from pathlib import Path
import sys


def compile_backend(backend):
    hip = backend == "hip"
    name = "hiprtc" if hip else "nvrtc"
    path = ctypes.util.find_library(name)
    if not path:
        for candidate in (["/opt/rocm/lib/libhiprtc.so", "/opt/rocm/core/lib/libhiprtc.so"] if hip else ["/usr/local/cuda/lib64/libnvrtc.so"]):
            if Path(candidate).exists():
                path = candidate
                break
    if not path:
        raise RuntimeError(f"{name} unavailable")
    lib = C.CDLL(path)
    create = getattr(lib, name+"CreateProgram")
    create.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p]
    build = getattr(lib, name+"CompileProgram")
    build.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
    logsize = getattr(lib, name+"GetProgramLogSize")
    logsize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    getlog = getattr(lib, name+"GetProgramLog")
    getlog.argtypes = [C.c_void_p, C.c_void_p]
    destroy = getattr(lib, name+"DestroyProgram")
    destroy.argtypes = [C.POINTER(C.c_void_p)]
    program = C.c_void_p()
    source = Path(__file__).with_name("gn_kernels.cu").read_bytes()
    assert create(C.byref(program), source, b"gn_kernels.cu", 0, None, None) == 0
    try:
        flags = [b"--std=c++17", b"--gpu-architecture=gfx1201" if hip else b"--gpu-architecture=compute_120"]
        if hip:
            flags += [b"-DGN_HIP=1", b"-O3"]
        result = build(program, len(flags), (C.c_char_p*len(flags))(*flags))
        size = C.c_size_t()
        logsize(program, C.byref(size))
        log = C.create_string_buffer(size.value+1)
        getlog(program, log)
        if result:
            raise RuntimeError(log.value.decode())
        print(f"PASS {name} compilation ({'gfx1201' if hip else 'sm120'}); GPU execution not tested")
    finally:
        destroy(C.byref(program))


if __name__ == "__main__":
    for backend in sys.argv[1:] or ["cuda", "hip"]:
        compile_backend(backend)
