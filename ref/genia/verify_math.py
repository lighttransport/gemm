"""Verify CUDA primitives against authenticated pinned upstream math captures."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FP = C.POINTER(C.c_float)
IP = C.POINTER(C.c_int32)
BP = C.POINTER(C.c_uint8)


def pointer(array, kind=FP):
    return array.ctypes.data_as(kind)


def validate_arrays(arrays):
    """Check native buffer contracts independently of self-described metadata."""
    coords = arrays.get("coords")
    cameras = arrays.get("cameras")
    if coords is None or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("invalid oracle coordinates")
    if cameras is None or cameras.ndim != 2 or cameras.shape[1] != 3:
        raise ValueError("invalid oracle cameras")
    n, views = len(coords), len(cameras)
    if not 1 <= n <= 64**3 or not 1 <= views <= 16:
        raise ValueError("invalid oracle dimensions")
    expected = {"coords": ((n, 3), "int32"), "cameras": ((views, 3), "float32"),
                "occupancy": ((64, 64, 64), "uint8"),
                "visibility": ((views, n), "float32"),
                "velocities": ((views, n, 8), "float32"),
                "weights": ((views, n), "float32")}
    expected.update({name: ((n, 8), "float32") for name in
                     ("fused", "latent", "gradient", "clean", "guided")})
    if set(arrays) != set(expected):
        raise ValueError("unexpected oracle arrays")
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError("invalid native oracle buffer: " + name)
        if not np.isfinite(array).all():
            raise ValueError("nonfinite oracle buffer: " + name)
    if np.any(coords < 0) or np.any(coords >= 64):
        raise ValueError("oracle coordinates outside grid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "tmp/genia-ref/math")
    parser.add_argument("--library", type=Path, default=ROOT / "cuda/genia/libgenia_cuda.so")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    (args.reference / "parity.json").unlink(missing_ok=True)
    manifest = json.loads((args.reference / "manifest.json").read_text())
    lock = json.loads((Path(__file__).parent / "lock.json").read_text())
    if manifest.get("format") != "gemm.genia_math_reference.v1" or manifest.get("lock") != lock:
        raise ValueError("unverified GenIA math oracle")
    if manifest.get("visibility_source_sha256") != lock["visibility_source_sha256"]:
        raise ValueError("unverified visibility implementation")
    arrays = {}
    for filename, rec in manifest["files"].items():
        path = args.reference / filename
        if not path.resolve().is_relative_to(args.reference.resolve()):
            raise ValueError("unsafe oracle filename")
        if hashlib.sha256(path.read_bytes()).hexdigest() != rec["sha256"]:
            raise ValueError("oracle checksum mismatch")
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != rec["shape"] or str(array.dtype) != rec["dtype"]:
            raise ValueError("oracle shape/dtype mismatch")
        arrays[path.stem] = np.ascontiguousarray(array)
    validate_arrays(arrays)
    lib = C.CDLL(str(args.library.resolve()))
    lib.genia_cuda_open.argtypes = [C.c_int, C.c_size_t]; lib.genia_cuda_open.restype = C.c_void_p
    lib.genia_cuda_close.argtypes = [C.c_void_p]
    lib.genia_cuda_error.restype = C.c_char_p
    lib.genia_cuda_visibility.argtypes = [C.c_void_p, IP, C.c_int, FP, BP, C.c_int, C.c_float, BP]
    lib.genia_cuda_fuse.argtypes = [C.c_void_p, FP, FP, C.c_int, C.c_int, C.c_int, C.c_float, C.c_float, FP]
    lib.genia_cuda_tweedie.argtypes = [C.c_void_p, FP, FP, C.c_int, C.c_float, FP]
    lib.genia_cuda_guidance.argtypes = [C.c_void_p, FP, FP, C.c_int, C.c_float, C.c_int, FP]
    ctx = lib.genia_cuda_open(args.device, 16 << 20)
    if not ctx:
        raise RuntimeError(lib.genia_cuda_error().decode())
    def check(code):
        if code:
            raise RuntimeError(lib.genia_cuda_error().decode())
    report = {"format": "gemm.genia_math_parity.v1", "revision": lock["revision"], "device": args.device}
    try:
        n = len(arrays["coords"])
        for i, camera in enumerate(arrays["cameras"]):
            visible = np.empty(n, np.uint8)
            check(lib.genia_cuda_visibility(ctx, pointer(arrays["coords"], IP), n, pointer(camera),
                pointer(arrays["occupancy"], BP), 64, manifest["neighbor_tolerance"], pointer(visible, BP)))
            if not np.array_equal(visible, arrays["visibility"][i]):
                raise AssertionError("DDA differs from pinned upstream")
        report["visibility"] = "exact"
        out = np.empty_like(arrays["fused"])
        check(lib.genia_cuda_fuse(ctx, pointer(arrays["visibility"]), pointer(arrays["velocities"]),
            len(arrays["cameras"]), n, 8, manifest["temperature"], manifest["floor_weight"], pointer(out)))
        report["fusion_max_abs"] = float(np.max(abs(out-arrays["fused"])))
        np.testing.assert_allclose(out, arrays["fused"], rtol=1e-6, atol=1e-6)
        check(lib.genia_cuda_tweedie(ctx, pointer(arrays["latent"]), pointer(arrays["fused"]),
            n*8, manifest["time"], pointer(out)))
        report["tweedie_max_abs"] = float(np.max(abs(out-arrays["clean"])))
        np.testing.assert_allclose(out, arrays["clean"], rtol=1e-6, atol=1e-6)
        check(lib.genia_cuda_guidance(ctx, pointer(arrays["fused"]), pointer(arrays["gradient"]),
            n*8, manifest["guidance_weight"], 1, pointer(out)))
        report["guidance_max_abs"] = float(np.max(abs(out-arrays["guided"])))
        np.testing.assert_allclose(out, arrays["guided"], rtol=1e-6, atol=1e-6)
        report["pass"] = True
    finally:
        lib.genia_cuda_close(ctx)
    (args.reference / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
