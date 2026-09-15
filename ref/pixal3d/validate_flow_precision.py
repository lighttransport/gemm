"""Compare one complete native flow stack with a matching PyTorch dtype."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from upstream_import import ROOT, prepare

p = argparse.ArgumentParser()
p.add_argument('--backend', choices=['cuda', 'rocm'], required=True)
p.add_argument('--dump-dir', type=Path, required=True)
p.add_argument('--stage', choices=['structure', 'shape512', 'shape1024', 'texture'], required=True)
p.add_argument('--precision', choices=['bf16', 'fp32', 'mixed'], default='bf16')
p.add_argument('--model-dir', type=Path, default=Path('/mnt/disk2/models/Pixal3D'))
p.add_argument('--gpu-kernels', choices=['auto', 'blas', 'mma'], default='auto')
p.add_argument('--blocks', type=int, default=30)
a = p.parse_args(); prepare(); torch.set_num_threads(16)
assert torch.cuda.is_available() and bool(torch.version.hip) == (a.backend == 'rocm')
torch.cuda.set_per_process_memory_fraction(.45)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
lib = C.CDLL(str(ROOT.parent.parent / 'cpu/pixal3d/libpixal3d_validation.so'))
lib.px_test_error.restype = C.c_char_p
lib.px_test_set_gpu.argtypes = [C.c_int, C.c_int]
assert lib.px_test_set_gpu(1, ['auto', 'blas', 'mma'].index(a.gpu_kernels)) == 0
lib.px_test_set_gpu_flow_precision.argtypes = [C.c_int]
assert lib.px_test_set_gpu_flow_precision({'bf16': 0, 'fp32': 1, 'mixed': 2}[a.precision]) == 0
fp = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
lib.px_test_flow_open.argtypes = [C.c_int, C.c_char_p]; lib.px_test_flow_open.restype = C.c_void_p
lib.px_test_flow_close.argtypes = [C.c_void_p]
lib.px_test_flow_run.argtypes = [C.c_void_p, fp, fp, ip, C.c_int, C.c_int, fp, C.c_int, fp, C.c_int, C.c_float, C.c_int, C.c_int]
stems = {'structure': 'ss_flow_img_dit_1_3B_64_bf16', 'shape512': 'slat_flow_img2shape_dit_1_3B_512_bf16',
         'shape1024': 'slat_flow_img2shape_dit_1_3B_1024_bf16', 'texture': 'slat_flow_imgshape2tex_dit_1_3B_1024_bf16'}
model_keys = {'structure': 'sparse_structure_flow_model', 'shape512': 'shape_slat_flow_model_512',
              'shape1024': 'shape_slat_flow_model_1024', 'texture': 'tex_slat_flow_model_1024'}
stem = a.model_dir / 'ckpts' / stems[a.stage]
noise = load_file(a.dump_dir / f'{a.stage}_noise.safetensors')
x = noise['feats'].numpy().astype(np.float32); coords = noise['coords'].numpy().astype(np.int32)
if a.stage == 'texture':
    x = np.concatenate([x, load_file(a.dump_dir / 'shape1024_step_12.safetensors')['feats'].numpy()], 1)
g = load_file(a.dump_dir / f'{a.stage}_global.safetensors')['feats'].numpy().astype(np.float32)
proj = load_file(a.dump_dir / f'{a.stage}_projected.safetensors')['feats'].numpy().astype(np.float32)
config = json.loads(stem.with_suffix('.json').read_text())['args']; config['dtype'] = a.precision == 'bf16' and 'bfloat16' or 'float32'; config['num_blocks'] = a.blocks
if a.stage == 'structure':
    from pixal3d.models.sparse_structure_flow import SparseStructureFlowModel
    model = SparseStructureFlowModel(**config); sample = torch.tensor(x.T.reshape(1, x.shape[1], 16, 16, 16), device='cuda')
    cond = {'global': torch.tensor(g[None], device='cuda'), 'proj': torch.tensor(proj[None], device='cuda')}
else:
    from pixal3d.models.structured_latent_flow import SLatFlowModel
    from pixal3d.modules.sparse import SparseTensor
    model = SLatFlowModel(**config); sample = SparseTensor(feats=torch.tensor(x, device='cuda'), coords=torch.tensor(coords, device='cuda'))
    cond = {'global': torch.tensor(g[None], device='cuda'), 'proj': SparseTensor(feats=torch.tensor(proj, device='cuda'), coords=torch.tensor(coords, device='cuda'))}
from safetensors import safe_open
with safe_open(str(stem.with_suffix('.safetensors')), framework='pt') as f:
    state = {k: f.get_tensor(k) for k in model.state_dict() if k != 'rope_phases'}
model.load_state_dict(state, strict=False); model.eval().to('cuda')
with torch.inference_mode():
    expected = model(sample, torch.tensor([1000.], device='cuda'), cond)
    expected = expected[0].flatten(1).T if a.stage == 'structure' else expected.feats
actual = np.empty_like(x)
session = lib.px_test_flow_open({'cuda': 1, 'rocm': 2}[a.backend], str(stem.with_suffix('.safetensors')).encode())
assert session, lib.px_test_error().decode()
try:
    rc = lib.px_test_flow_run(session, actual, x, coords, len(x), x.shape[1], g, g.shape[1], proj, proj.shape[1], 1., a.blocks, int(a.precision == 'bf16'))
    assert rc == 0, lib.px_test_error().decode()
finally:
    lib.px_test_flow_close(session)
xx = actual.astype(np.float64).ravel(); yy = expected.detach().float().cpu().numpy().astype(np.float64).ravel()
nrmse = np.linalg.norm(xx - yy) / np.linalg.norm(yy); cosine = np.dot(xx, yy) / (np.linalg.norm(xx) * np.linalg.norm(yy))
print(json.dumps(dict(backend=a.backend, stage=a.stage, precision=a.precision, tokens=len(x), nrmse=nrmse, cosine=cosine, max_abs=float(np.max(np.abs(xx - yy))))))
assert nrmse < .02 and cosine > .999
