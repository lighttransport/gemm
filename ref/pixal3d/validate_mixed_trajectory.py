"""Compare a complete native flow trajectory to its PyTorch FP32 model.

The default checks the production mixed mode. The flow request flag passed to
px_test_flow_run selects the configured GPU mode; passing 0 would silently
run the FP32 flow instead."""
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
p.add_argument("--stage", choices=("structure", "shape512", "shape1024", "texture"),
               default="structure")
p.add_argument("--model-dir", type=Path, default=Path("/mnt/disk2/models/Pixal3D"))
p.add_argument("--gpu-kernels", choices=("auto", "blas", "mma"), default="auto")
p.add_argument("--precision", choices=("mixed", "fp32", "bf16"), default="mixed")
a = p.parse_args()
prepare()
torch.set_num_threads(16)
assert torch.cuda.is_available() and not torch.version.hip
torch.cuda.set_per_process_memory_fraction(.45)
torch.backends.cuda.matmul.allow_tf32 = False

pipeline = json.loads((a.model_dir / "pipeline.json").read_text())["args"]
keys = {
    "structure": ("sparse_structure_flow_model", "sparse_structure_sampler"),
    "shape512": ("shape_slat_flow_model_512", "shape_slat_sampler"),
    "shape1024": ("shape_slat_flow_model_1024", "shape_slat_sampler"),
    "texture": ("tex_slat_flow_model_1024", "tex_slat_sampler"),
}
model_key, sampler_key = keys[a.stage]
stem = a.model_dir / pipeline["models"][model_key]
config = json.loads(stem.with_suffix(".json").read_text())["args"]
config["dtype"] = "float32"
from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel
from pixal3d.models.structured_latent_flow import SLatFlowModel
from pixal3d.modules.sparse import SparseTensor
from pixal3d.pipelines.samplers import FlowEulerGuidanceIntervalSampler

model = (SparseStructureFlowModel if a.stage == "structure" else SLatFlowModel)(**config)
model.load_state_dict(load_file(stem.with_suffix(".safetensors")))
model.eval().to("cuda")


def read(name):
    return load_file(a.dump_dir / f"{name}.safetensors")

noise = read(a.stage + "_noise")
x0 = noise["feats"].numpy().astype(np.float32)
coords = noise["coords"].numpy().astype(np.int32)
global_np = read(a.stage + "_global")["feats"].numpy().astype(np.float32)
projected_np = read(a.stage + "_projected")["feats"].numpy().astype(np.float32)
global_cond = torch.tensor(global_np[None], device="cuda")
projected_tensor = torch.tensor(projected_np, device="cuda")
extra = {}
shape_np = None
if a.stage == "structure":
    sample = torch.tensor(x0.T.reshape(1, x0.shape[1], 16, 16, 16), device="cuda")
    projected = projected_tensor[None]
else:
    coords_tensor = torch.tensor(coords, device="cuda")
    sample = SparseTensor(feats=torch.tensor(x0, device="cuda"), coords=coords_tensor)
    projected = SparseTensor(feats=projected_tensor, coords=coords_tensor)
    if a.stage == "texture":
        shape = read("shape1024_step_12")
        shape_np = shape["feats"].numpy().astype(np.float32)
        np.testing.assert_array_equal(coords, shape["coords"].numpy())
        extra["concat_cond"] = SparseTensor(
            feats=torch.tensor(shape_np, device="cuda"), coords=coords_tensor)
cond = {"global": global_cond, "proj": projected}
negative_projected = torch.zeros_like(projected) if a.stage == "structure" else projected.replace(
    torch.zeros_like(projected.feats))
neg_cond = {"global": torch.zeros_like(global_cond), "proj": negative_projected}
spec = pipeline[sampler_key]
params = dict(spec["params"])
steps = params.pop("steps")
rescale_t = params.pop("rescale_t")
with torch.inference_mode():
    expected = FlowEulerGuidanceIntervalSampler(**spec["args"]).sample(
        model, sample, cond=cond, neg_cond=neg_cond, steps=steps,
        rescale_t=rescale_t, verbose=False, **params, **extra).samples
expected = (expected[0].flatten(1).T if a.stage == "structure" else expected.feats).float().cpu().numpy()
del model, sample, cond, neg_cond, projected
torch.cuda.empty_cache()

lib = C.CDLL(str(ROOT.parent.parent / "cpu/pixal3d/libpixal3d_validation.so"))
lib.px_test_error.restype = C.c_char_p
lib.px_test_set_gpu.argtypes = [C.c_int, C.c_int]
assert lib.px_test_set_gpu(1, ("auto", "blas", "mma").index(a.gpu_kernels)) == 0
lib.px_test_set_gpu_flow_precision.argtypes = [C.c_int]
assert lib.px_test_set_gpu_flow_precision(("bf16", "fp32", "mixed").index(a.precision)) == 0
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
        model_input = (np.ascontiguousarray(np.concatenate([native, shape_np], 1))
                       if shape_np is not None else native)
        assert lib.px_test_flow_run(session, positive, model_input, coords, len(native), model_input.shape[1],
                                    global_np, global_np.shape[1], projected_np,
                                    projected_np.shape[1], float(t), 30, 1) == 0, lib.px_test_error().decode()
        guided = lo <= t <= hi
        if guided:
            assert lib.px_test_flow_run(session, negative, model_input, coords, len(native), model_input.shape[1],
                                        zeros_global, zeros_global.shape[1], zeros_projected,
                                        zeros_projected.shape[1], float(t), 30, 1) == 0, lib.px_test_error().decode()
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
result = {"backend": "cuda", "stage": a.stage, "precision": a.precision,
          "steps": steps, "tokens": len(native), "nrmse": float(nrmse),
          "cosine": float(cosine), "max_abs": float(np.max(np.abs(xx - yy)))}
print(json.dumps(result))
# FP32 mode tracks the FP32 sampler closely. Mixed and BF16 GEMMs diverge
# through CFG amplification; their gates catch regressions, not parity.
gate = {"fp32": (.001, .999999), "mixed": (.08, .997), "bf16": (.15, .99)}[a.precision]
assert nrmse < gate[0] and cosine > gate[1], gate
print(f"{a.precision} FP32-reference trajectory PASS")
