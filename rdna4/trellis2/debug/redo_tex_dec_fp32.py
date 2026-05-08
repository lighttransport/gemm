"""Re-run only the tex_slat_decoder with fp32 inner blocks (convert_to_fp32),
loading inputs from the existing rdna4/trellis2/verify-dumps-rocm/ tree.

If the fp32 path produces tex_voxels.feats in the expected [-0.1, 1.1] range,
the bug is fp16 overflow inside SparseConvNeXtBlock3d — and forcing fp32 in
the Trellis2ImageTo3DPipeline tex decoder is a one-line stopgap.
"""
import os, sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'runner')))
from shim_bootstrap import install_all  # noqa: E402
install_all()

import numpy as np
import torch
from gen_stage2_ref import _patch_dinov3_extractor, _patch_birefnet_noop


@torch.no_grad()
def main():
    dumps = os.path.join(_REPO_ROOT, 'rdna4', 'trellis2', 'verify-dumps-rocm')
    _patch_dinov3_extractor(os.environ['DINOV3_WEIGHTS'])
    _patch_birefnet_noop()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()

    feats = torch.from_numpy(np.load(f'{dumps}/12_tex_slat_denorm_feats.npy')).cuda()
    coords = torch.from_numpy(np.load(f'{dumps}/05_ss_coords.npy')).cuda()
    tex_slat = SparseTensor(feats=feats, coords=coords)

    subs = []
    for i in range(8):
        cp = f'{dumps}/14_shape_sub{i}.coords.npy'; fp = f'{dumps}/14_shape_sub{i}.feats.npy'
        if not os.path.exists(cp): break
        sc = torch.from_numpy(np.load(cp)).cuda()
        sf = torch.from_numpy(np.load(fp).astype(np.float32)).cuda()
        subs.append(SparseTensor(feats=sf, coords=sc))

    dec = pipeline.models['tex_slat_decoder']
    dec.to(pipeline.device)

    print('=== fp16 (original) ===')
    out = dec(tex_slat, guide_subs=subs)
    f = out.feats.detach().float().cpu().numpy()
    print(f'  out  min={f.min():.3f} max={f.max():.3f} mean={f.mean():.4f} std={f.std():.3f}')

    print('\n=== fp32 (force convert_to_fp32) ===')
    dec.convert_to_fp32()
    dec.dtype = torch.float32
    out2 = dec(tex_slat, guide_subs=subs)
    g = out2.feats.detach().float().cpu().numpy()
    print(f'  out  min={g.min():.3f} max={g.max():.3f} mean={g.mean():.4f} std={g.std():.3f}')
    voxels = (g * 0.5 + 0.5)
    print(f'  *0.5+0.5 RGB  mean={voxels[:,:3].mean(0)}  metallic={voxels[:,3].mean():.3f}  '
          f'roughness={voxels[:,4].mean():.3f}  alpha={voxels[:,5].mean():.3f}')
    print(f'  >10%: {(np.abs(g)>10).mean()*100:.4f}%   >1%: {(np.abs(g)>1).mean()*100:.3f}%')

    out_dir = os.path.join(_REPO_ROOT, 'rdna4', 'trellis2', 'verify-dumps-rocm-fp32')
    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/15_tex_voxels.coords.npy', out2.coords.cpu().numpy())
    np.save(f'{out_dir}/15_tex_voxels.feats.npy', voxels)
    print(f'\nSaved fp32 dump -> {out_dir}/15_tex_voxels.{{coords,feats}}.npy')


if __name__ == '__main__':
    main()
