"""Live per-block probe: runs the full Trellis2ImageTo3DPipeline up through
shape_slat decode, then hooks tex_slat_decoder before its forward call so
the SparseTensor _spatial_cache state is the real one (not reloaded npy).

Outputs per-block stats; first row with frac>100 spiking pinpoints the layer.
"""
import os, sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runner')))
from shim_bootstrap import install_all  # noqa: E402
install_all()

import argparse, gc
import numpy as np
import torch
from PIL import Image
from gen_stage2_ref import _patch_dinov3_extractor, _patch_birefnet_noop
if os.environ.get('USE_RDNA4_SPCONV', '0') == '1':
    from kernels import spconv_rdna4_ext
    spconv_rdna4_ext.install()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default=os.path.join(_REPO_ROOT,
                    'cpu/trellis2/trellis2_repo/assets/example_image/T.png'))
    ap.add_argument('--dinov3', default=os.environ.get('DINOV3_WEIGHTS',
                    '/mnt/disk1/models/dinov3-vitl16/model.safetensors'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force-fp32-tex-dec', action='store_true',
                    help='Run tex_slat_decoder.convert_to_fp32() before forward.')
    ap.add_argument('--sweep-flex-algos', action='store_true',
                    help='After main run, re-run dec with each FLEX_GEMM_ALGO.')
    args = ap.parse_args()

    _patch_dinov3_extractor(args.dinov3)
    _patch_birefnet_noop()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor

    pipe = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipe.cuda()

    img = Image.open(args.image)
    if img.mode != 'RGBA':
        rgb = np.array(img.convert('RGB'))
        a = ((rgb.sum(-1) > 15) * 255).astype(np.uint8)
        img = Image.fromarray(np.dstack([rgb, a]), 'RGBA')
    pp = pipe.preprocess_image(img)
    torch.manual_seed(args.seed)
    cond = pipe.get_cond([pp], 512)

    torch.manual_seed(args.seed)
    flow = pipe.models['sparse_structure_flow_model']
    if pipe.low_vram: flow.to(pipe.device)
    noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution).to(pipe.device)
    z_s = pipe.sparse_structure_sampler.sample(flow, noise, **cond,
        **pipe.sparse_structure_sampler_params, verbose=True, tqdm_desc='SS').samples
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
    torch.manual_seed(args.seed + 1)
    n_sh = torch.randn(coords.shape[0], flow_sh.in_channels).to(pipe.device)
    sh = pipe.shape_slat_sampler.sample(flow_sh, SparseTensor(feats=n_sh, coords=coords),
        **cond, **pipe.shape_slat_sampler_params, verbose=True, tqdm_desc='SH').samples
    if pipe.low_vram: flow_sh.cpu()
    std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(sh.device)
    mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(sh.device)
    shape_slat = sh * std + mean

    flow_tx = pipe.models['tex_slat_flow_model_512']
    if pipe.low_vram: flow_tx.to(pipe.device)
    torch.manual_seed(args.seed + 2)
    cat_dim = flow_tx.in_channels - shape_slat.feats.shape[1]
    n_tx = torch.randn(shape_slat.coords.shape[0], cat_dim).to(pipe.device)
    cc = shape_slat.replace(feats=(shape_slat.feats - mean) / std)
    tx = pipe.tex_slat_sampler.sample(flow_tx, shape_slat.replace(feats=n_tx),
        concat_cond=cc, **cond, **pipe.tex_slat_sampler_params,
        verbose=True, tqdm_desc='TX').samples
    if pipe.low_vram: flow_tx.cpu()
    std_t = torch.tensor(pipe.tex_slat_normalization['std'])[None].to(tx.device)
    mean_t = torch.tensor(pipe.tex_slat_normalization['mean'])[None].to(tx.device)
    tex_slat = tx * std_t + mean_t
    torch.cuda.empty_cache(); gc.collect()

    pipe.models['shape_slat_decoder'].set_resolution(512)
    if pipe.low_vram:
        pipe.models['shape_slat_decoder'].to(pipe.device)
        pipe.models['shape_slat_decoder'].low_vram = True
    meshes, subs = pipe.models['shape_slat_decoder'](shape_slat, return_subs=True)
    if pipe.low_vram:
        pipe.models['shape_slat_decoder'].cpu()
        pipe.models['shape_slat_decoder'].low_vram = False

    dec = pipe.models['tex_slat_decoder']
    if pipe.low_vram: dec.to(pipe.device)
    if args.force_fp32_tex_dec:
        print('!! converting tex_slat_decoder to fp32')
        dec.convert_to_fp32()
        dec.dtype = torch.float32

    print('\n stage | block |   N×C       |  min     |  max     |  mean    |  std    | >10% | >100% | type')
    print('-' * 130)
    rows = []
    def make_hook(si, bj, cls):
        def h(_m, _i, out):
            if hasattr(out, 'feats'): f = out.feats.detach().float().cpu().numpy()
            elif isinstance(out, tuple) and hasattr(out[0], 'feats'):
                f = out[0].feats.detach().float().cpu().numpy()
            else: return
            f10 = (np.abs(f) > 10).mean() * 100
            f100 = (np.abs(f) > 100).mean() * 100
            print(f' {si:>3d}   |  {bj:>3d}  | {f.shape[0]:>7d}×{f.shape[1]:<3d} '
                  f'| {f.min():>8.2f} | {f.max():>8.2f} | {f.mean():>8.3f} | {f.std():>7.2f} '
                  f'| {f10:>4.2f}% | {f100:>5.3f}% | {cls}', flush=True)
            rows.append((si, bj, cls, float(f.min()), float(f.max()), f10, f100))
        return h

    handles = []
    for i, st in enumerate(dec.blocks):
        for j, blk in enumerate(st):
            handles.append(blk.register_forward_hook(make_hook(i, j, type(blk).__name__)))

    out = dec(tex_slat, guide_subs=subs)
    f = out.feats.detach().float().cpu().numpy()
    print(f'\nFINAL output_layer.out  min={f.min():.4g}  max={f.max():.4g}  '
          f'mean={f.mean():.4g}  std={f.std():.4g}  >10%: {(np.abs(f)>10).mean()*100:.4f}%')

    for h in handles: h.remove()
    spike = next((r for r in rows if r[6] > 0.001), None)
    if spike:
        print(f'\nFIRST >100% spike: stage {spike[0]} block {spike[1]} ({spike[2]}) '
              f'min={spike[3]:.1f} max={spike[4]:.1f} >10={spike[5]:.2f}% >100={spike[6]:.3f}%')

    if args.sweep_flex_algos:
        from trellis2.modules.sparse.conv import config as gconf
        algos = ['explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk',
                 'masked_implicit_gemm', 'masked_implicit_gemm_splitk']
        print(f'\n--- FLEX_GEMM_ALGO sweep on tex_slat_decoder ---')
        for algo in algos:
            gconf.FLEX_GEMM_ALGO = algo
            try:
                o = dec(tex_slat, guide_subs=subs)
                ff = o.feats.detach().float().cpu().numpy()
                nans = bool(np.isnan(ff).any())
                print(f'  algo={algo:32s}  N={ff.shape[0]:>7d}  min={ff.min():>9.3f}  max={ff.max():>9.3f}  '
                      f'mean={ff.mean():>7.4f}  std={ff.std():>7.3f}  NaN={nans}  '
                      f'>10%={(np.abs(ff)>10).mean()*100:>6.3f}%')
            except Exception as e:
                print(f'  algo={algo:32s}  FAIL: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
