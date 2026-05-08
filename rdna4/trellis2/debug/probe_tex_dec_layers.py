"""Probe tex_slat_decoder block-by-block to find the layer where extreme
outliers first appear on ROCm.

Loads the same pipeline + ROCm dumps used by dump_rocm.py and re-runs the
tex_slat_decoder while attaching forward hooks to each child block, dumping
{min, max, mean, std, frac>10, frac>100} of the SparseTensor.feats output
after every block.

Output: stdout table; first row with frac>10 jumping >0 localises the bug.
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runner')))
from shim_bootstrap import install_all  # noqa: E402
install_all()

import argparse
import numpy as np
import torch

from gen_stage2_ref import _patch_dinov3_extractor, _patch_birefnet_noop


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rocm-dumps', default=os.path.join(RDNA4_DIR, 'verify-dumps-rocm'))
    ap.add_argument('--model-id', default='microsoft/TRELLIS.2-4B')
    ap.add_argument('--dinov3', default=os.environ.get('DINOV3_WEIGHTS',
                                       '/mnt/disk1/models/dinov3-vitl16/model.safetensors'))
    args = ap.parse_args()

    _patch_dinov3_extractor(args.dinov3)
    _patch_birefnet_noop()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_id)
    pipeline.cuda()

    # Reload tex_slat_denorm + sub coords/feats from the ROCm dump so we
    # exercise *exactly* the same input that produced the broken texture.
    feats = torch.from_numpy(np.load(f'{args.rocm_dumps}/12_tex_slat_denorm_feats.npy')).cuda()
    coords = torch.from_numpy(np.load(f'{args.rocm_dumps}/05_ss_coords.npy')).cuda()
    tex_slat = SparseTensor(feats=feats, coords=coords)

    subs = []
    for i in range(8):
        cp = f'{args.rocm_dumps}/14_shape_sub{i}.coords.npy'
        fp = f'{args.rocm_dumps}/14_shape_sub{i}.feats.npy'
        if not os.path.exists(cp):
            break
        sc = torch.from_numpy(np.load(cp)).cuda()
        sf = torch.from_numpy(np.load(fp).astype(np.float32)).cuda()
        subs.append(SparseTensor(feats=sf, coords=sc))
    print(f'loaded {len(subs)} substructures, tex_slat coords={coords.shape} feats={feats.shape}')

    dec = pipeline.models['tex_slat_decoder']
    if pipeline.low_vram:
        dec.to(pipeline.device)

    # Walk dec.blocks (ModuleList[ModuleList[block]]) and register a forward
    # hook on each leaf block. Print rolling stats.
    print('\n stage_id | block_id |  N feats   |  min     |  max      |  mean    |  std     |  >10%   |  >100%')
    print('-' * 120)
    rows = []
    def make_hook(stage_i, block_j, block_class):
        def hook(_mod, _inp, out):
            if hasattr(out, 'feats'):
                f = out.feats.detach().float().cpu().numpy()
            elif isinstance(out, tuple) and hasattr(out[0], 'feats'):
                f = out[0].feats.detach().float().cpu().numpy()
            else:
                return
            mn, mx = float(f.min()), float(f.max())
            me, st = float(f.mean()), float(f.std())
            n = f.size
            f10 = float((np.abs(f) > 10).mean()) * 100
            f100 = float((np.abs(f) > 100).mean()) * 100
            print(f'  {stage_i:>4d}   |   {block_j:>3d}    | {f.shape[0]:>7d}x{f.shape[1]:<3d} '
                  f'| {mn:>8.2f} | {mx:>9.2f} | {me:>8.3f} | {st:>8.2f} | {f10:>5.3f}% | {f100:>5.3f}% '
                  f'| {block_class}', flush=True)
            rows.append((stage_i, block_j, mn, mx, st, f10, f100))
        return hook

    handles = []
    for i, stage in enumerate(dec.blocks):
        for j, block in enumerate(stage):
            handles.append(block.register_forward_hook(make_hook(i, j, type(block).__name__)))

    out = dec(tex_slat, guide_subs=subs)
    print('\nfinal output before *0.5+0.5 (post-output_layer):')
    f = out.feats.detach().float().cpu().numpy()
    print(f'   min={f.min():.3f} max={f.max():.3f} mean={f.mean():.4f} std={f.std():.3f} '
          f'>10%: {(np.abs(f)>10).mean()*100:.3f}%  >100%: {(np.abs(f)>100).mean()*100:.3f}%')

    for h in handles: h.remove()

    # First stage where frac>10 is non-zero is the suspect.
    first = next((r for r in rows if r[5] > 0.001), None)
    if first:
        print(f'\nFIRST OVERFLOW at stage {first[0]} block {first[1]}: '
              f'min={first[2]:.1f} max={first[3]:.1f} std={first[4]:.1f} '
              f'frac>10={first[5]:.3f}% frac>100={first[6]:.3f}%')
    else:
        print('\nNo per-block overflow detected; bug must be in output_layer / layer_norm / dtype cast.')


if __name__ == '__main__':
    main()
