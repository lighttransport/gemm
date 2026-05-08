"""4-trial reproducer for tex_slat_decoder non-determinism on RDNA4.

Without any patch: trials produce {NaN, std=0.45 CLEAN, std=∞, NaN} — the
"clean" slot drifts with allocator state. With USE_RDNA4_LINEAR_CHUNK=1
(default): all 4 trials produce identical clean output [-1.22, 1.16] std=0.76.
Root cause = ROCm hipBLASLt mishandling SparseLinear(64->6) at M~1.5M rows.
"""
import os, sys, gc
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('ATTN_BACKEND', 'sdpa')
_REPO_ROOT = '/mnt/disk1/work/gemm/trellis2'
sys.path.insert(0, f'{_REPO_ROOT}/rdna4/trellis2')
sys.path.insert(0, f'{_REPO_ROOT}/cpu/trellis2/trellis2_repo')
sys.path.insert(0, f'{_REPO_ROOT}/ref/trellis2')
import texgen_sw_rast, cumesh_xatlas_shim, flash_attn_sdpa_shim
texgen_sw_rast.install_as_nvdiffrast(); cumesh_xatlas_shim.install_as_cumesh()
try:
    from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
    PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
except ImportError: pass
flash_attn_sdpa_shim.install_as_flash_attn()
import numpy as np, torch
from PIL import Image
from gen_stage2_ref import _patch_dinov3_extractor, _patch_birefnet_noop
import spconv_rdna4_ext
if os.environ.get('USE_RDNA4_SPCONV', '0') == '1':
    spconv_rdna4_ext.install()
if os.environ.get('USE_RDNA4_LINEAR_CHUNK', '1') == '1':
    spconv_rdna4_ext.install_sparse_linear_chunking()

@torch.no_grad()
def main():
    _patch_dinov3_extractor(os.environ.get('DINOV3_WEIGHTS','/mnt/disk1/models/dinov3-vitl16/model.safetensors'))
    _patch_birefnet_noop()
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor
    pipe = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B'); pipe.cuda()
    img = Image.open(f'{_REPO_ROOT}/cpu/trellis2/trellis2_repo/assets/example_image/T.png')
    if img.mode != 'RGBA':
        rgb = np.array(img.convert('RGB'))
        a = ((rgb.sum(-1) > 15) * 255).astype(np.uint8)
        img = Image.fromarray(np.dstack([rgb, a]), 'RGBA')
    pp = pipe.preprocess_image(img); torch.manual_seed(42)
    cond = pipe.get_cond([pp], 512)
    flow = pipe.models['sparse_structure_flow_model']
    if pipe.low_vram: flow.to(pipe.device)
    torch.manual_seed(42)
    noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution).cuda()
    z_s = pipe.sparse_structure_sampler.sample(flow, noise, **cond, **pipe.sparse_structure_sampler_params, verbose=False).samples
    if pipe.low_vram: flow.cpu()
    dec_ss = pipe.models['sparse_structure_decoder']
    if pipe.low_vram: dec_ss.to(pipe.device)
    decoded = dec_ss(z_s)
    if pipe.low_vram: dec_ss.cpu()
    occ = decoded > 0
    if 32 != decoded.shape[2]:
        r = decoded.shape[2] // 32
        occ = torch.nn.functional.max_pool3d(occ.float(), r, r, 0) > 0.5
    coords = torch.argwhere(occ)[:, [0,2,3,4]].int()
    flow_sh = pipe.models['shape_slat_flow_model_512']
    if pipe.low_vram: flow_sh.to(pipe.device)
    torch.manual_seed(43)
    n_sh = torch.randn(coords.shape[0], flow_sh.in_channels).cuda()
    sh = pipe.shape_slat_sampler.sample(flow_sh, SparseTensor(feats=n_sh, coords=coords),
        **cond, **pipe.shape_slat_sampler_params, verbose=False).samples
    if pipe.low_vram: flow_sh.cpu()
    std = torch.tensor(pipe.shape_slat_normalization['std'])[None].cuda()
    mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].cuda()
    shape_slat = sh * std + mean
    flow_tx = pipe.models['tex_slat_flow_model_512']
    if pipe.low_vram: flow_tx.to(pipe.device)
    torch.manual_seed(44)
    cat_dim = flow_tx.in_channels - shape_slat.feats.shape[1]
    n_tx = torch.randn(shape_slat.coords.shape[0], cat_dim).cuda()
    cc = shape_slat.replace(feats=(shape_slat.feats - mean) / std)
    tx = pipe.tex_slat_sampler.sample(flow_tx, shape_slat.replace(feats=n_tx),
        concat_cond=cc, **cond, **pipe.tex_slat_sampler_params, verbose=False).samples
    if pipe.low_vram: flow_tx.cpu()
    std_t = torch.tensor(pipe.tex_slat_normalization['std'])[None].cuda()
    mean_t = torch.tensor(pipe.tex_slat_normalization['mean'])[None].cuda()
    tex_slat_base = tx * std_t + mean_t
    pipe.models['shape_slat_decoder'].set_resolution(512)
    if pipe.low_vram:
        pipe.models['shape_slat_decoder'].to(pipe.device); pipe.models['shape_slat_decoder'].low_vram = True
    meshes, subs = pipe.models['shape_slat_decoder'](shape_slat, return_subs=True)
    if pipe.low_vram:
        pipe.models['shape_slat_decoder'].cpu(); pipe.models['shape_slat_decoder'].low_vram = False

    # Snapshot tex_slat_base feats so we can reset between trials
    feats_snap = tex_slat_base.feats.detach().clone()
    coords_snap = tex_slat_base.coords.detach().clone()
    sub_feats_snap = [s.feats.detach().clone() for s in subs]
    sub_coords_snap = [s.coords.detach().clone() for s in subs]

    dec = pipe.models['tex_slat_decoder']
    if pipe.low_vram: dec.to(pipe.device)
    dec.eval()

    for trial in range(4):
        # Rebuild SparseTensor each trial — fresh _spatial_cache
        ts = SparseTensor(feats=feats_snap.clone(), coords=coords_snap.clone())
        fresh_subs = [SparseTensor(feats=sub_feats_snap[i].clone(), coords=sub_coords_snap[i].clone())
                      for i in range(len(subs))]
        out = dec(ts, guide_subs=fresh_subs)
        f = out.feats.detach().float().cpu().numpy()
        nans = bool(np.isnan(f).any())
        print(f'Trial {trial}: N={f.shape[0]}  min={f.min() if not nans else float("nan"):>10.3f}  '
              f'max={f.max() if not nans else float("nan"):>10.3f}  std={f.std() if not nans else float("nan"):>9.3f}  NaN={nans}')
        torch.cuda.empty_cache(); gc.collect()

if __name__ == '__main__': main()
