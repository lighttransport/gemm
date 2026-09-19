"""Compare a complete native mixed structure trajectory to PyTorch FP32."""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from upstream_import import ROOT, prepare


p = argparse.ArgumentParser()
p.add_argument("--dump-dir", type=Path, required=True)
p.add_argument("--model-dir", type=Path, default=Path("/mnt/disk2/models/Pixal3D"))
p.add_argument("--gpu-kernels", choices=("auto", "blas", "mma"), default="auto")
a = p.parse_args()
prepare()
torch.set_num_threads(16)
assert torch.cuda.is_available() and not torch.version.hip
torch.cuda.set_per_process_memory_fraction(.45)
torch.backends.cuda.matmul.allow_tf32 = False

pipeline = json.loads((a.model_dir / "pipeline.json").read_text())["args"]
stem = a.model_dir / pipeline["models"]["sparse_structure_flow_model"]
config = json.loads(stem.with_suffix(".json").read_text())["args"]
config["dtype"] = "float32"
from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel
from pixal3d.pipelines.samplers import FlowEulerGuidanceIntervalSampler

model = SparseStructureFlowModel(**config)
state = load_file(stem.with_suffix(".safetensors"))
model.load_state_dict(state)
model.eval().to("cuda")

def read(name):
    return load_file(a.dump_dir / f"{name}.safetensors")

noise = read("structure_noise")
x0 = noise["feats"].numpy().astype(np.float32)
coords = noise["coords"].numpy().astype(np.int32)
global_np = read("structure_global")["feats"].numpy().astype(np.float32)
projected_np = read("structure_projected")["feats"].numpy().astype(np.float32)

sample = torch.tensor(x0.T.reshape(1, 8, 16, 16, 16), device="cuda")
global_cond = torch.tensor(global_np[None], device="cuda")
projected = torch.tensor(projected_np[None], device="cuda")
cond = {"global": global_cond, "proj": projected}
neg_cond = {"global": torch.zeros_like(global_cond), "proj": torch.zeros_like(projected)}
spec = pipeline["sparse_structure_sampler"]
params = dict(spec["params"])
steps = params.pop("steps")
rescale_t = params.pop("rescale_t")
with torch.inference_mode():
    expected = FlowEulerGuidanceIntervalSampler(**spec["args"]).sample(
        model, sample, cond=cond, neg_cond=neg_cond, steps=steps,
        rescale_t=rescale_t, verbose=False, **params).samples
expected = expected[0].flatten(1).T.float().cpu().numpy()
del model, sample, cond, neg_cond
torch.cuda.empty_cache()

lib = C.CDLL(str(ROOT.parent.parent / "cpu/pixal3d/libpixal3d_validation.so"))
lib.px_test_error.restype = C.c_char_p
lib.px_test_set_gpu.argtypes = [C.c_int, C.c_int]
assert lib.px_test_set_gpu(1, ("auto", "blas", "mma").index(a.gpu_kernels)) == 0
lib.px_test_set_gpu_flow_precision.argtypes = [C.c_int]
assert lib.px_test_set_gpu_flow_precision(2) == 0
fp = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
lib.px_test_flow_open.argtypes = [C.c_int, C.c_char_p]
lib.px_test_flow_open.restype = C.c_void_p
lib.px_test_flow_close.argtypes = [C.c_void_p]
lib.px_test_flow_run.argtypes = [C.c_void_p, fp, fp, ip, C.c_int, C.c_int,
                                 fp, C.c_int, fp, C.c_int, C.c_float, C.c_int, C.c_int]
lib.pixal3d_euler_cfg.argtypes = [fp, fp, fp, C.c_size_t, C.c_float, C.c_float,
                                  C.c_float, C.c_float, C.c_float]

native = x0.copy()
positive = np.empty_like(native)
negative = np.empty_like(native)
zeros_global = np.zeros_like(global_np)
zeros_projected = np.zeros_like(projected_np)
times = np.linspace(1, 0, steps + 1)
times = rescale_t * times / (1 + (rescale_t - 1) * times)
lo, hi = params["guidance_interval"]
session = lib.px_test_flow_open(1, str(stem.with_suffix(".safetensors")).encode())
assert session, lib.px_test_error().decode()
try:
    for t, t_next in zip(times[:-1], times[1:]):
        assert lib.px_test_flow_run(session, positive, native, coords, len(native), 8,
                                    global_np, global_np.shape[1], projected_np,
                                    projected_np.shape[1], float(t), 30, 0) == 0
        guided = lo <= t <= hi
        if guided:
            assert lib.px_test_flow_run(session, negative, native, coords, len(native), 8,
                                        zeros_global, zeros_global.shape[1], zeros_projected,
                                        zeros_projected.shape[1], float(t), 30, 0) == 0
        else:
            negative[:] = positive
        lib.pixal3d_euler_cfg(native, positive, negative, native.size, float(t),
                              float(t_next), params["guidance_strength"] if guided else 1.0,
                              params["guidance_rescale"] if guided else 0.0,
                              spec["args"]["sigma_min"])
finally:
    lib.px_test_flow_close(session)

xx = native.astype(np.float64).ravel()
yy = expected.astype(np.float64).ravel()
nrmse = np.linalg.norm(xx - yy) / np.linalg.norm(yy)
cosine = np.dot(xx, yy) / (np.linalg.norm(xx) * np.linalg.norm(yy))
result = {"backend": "cuda", "stage": "structure", "precision": "mixed",
          "steps": steps, "nrmse": float(nrmse), "cosine": float(cosine),
          "max_abs": float(np.max(np.abs(xx - yy)))}
print(json.dumps(result))
assert nrmse < .001 and cosine > .999999
print("Mixed FP32-reference trajectory PASS")
