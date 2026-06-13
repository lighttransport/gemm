"""Load cached tex_slat_feats/coords + c2s caches, run ONLY the tex_slat_decoder
with per-stage hidden dumps. Skips DiT sampling for fast bisection iteration.

Expects the input dir to contain (from a prior gen_stage2_ref.py run):
  tex_slat_feats.npy, tex_slat_coords.npy
  cache_scale{16,8,4,2}_c2s_x_coords.npy, _idx.npy, _subidx.npy
"""
import argparse, os, sys, numpy as np, torch
import torch.nn.functional as F

os.environ.setdefault('ATTN_BACKEND', 'sdpa')
RDNA4_DIR = '/mnt/disk1/work/gemm/main/rdna4/trellis2'
REPO_DIR = '/mnt/disk1/work/gemm/main/cpu/trellis2/trellis2_repo'
sys.path.insert(0, RDNA4_DIR)
sys.path.insert(0, REPO_DIR)
import texgen_sw_rast, cumesh_xatlas_shim, flash_attn_sdpa_shim
texgen_sw_rast.install_as_nvdiffrast()
cumesh_xatlas_shim.install_as_cumesh()
from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
flash_attn_sdpa_shim.install_as_flash_attn()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', default='microsoft/TRELLIS.2-4B')
    ap.add_argument('--config', default='texturing_pipeline.json')
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Stub out DINOv3 + BiRefNet loaders to avoid gated HF fetches — decoder only.
    from trellis2.modules import image_feature_extractor as _ife
    from trellis2.pipelines.rembg.BiRefNet import BiRefNet as _BiRefNet
    class _Stub:
        def __init__(self, *a, **kw): self.model = None
        def to(self, *a, **kw): return self
        def cpu(self, *a, **kw): return self
        def __call__(self, *a, **kw):
            raise RuntimeError('stubbed — decoder-only dump script')
    _ife.DinoV3FeatureExtractor = _Stub
    _BiRefNet.__init__ = _Stub.__init__
    _BiRefNet.to = _Stub.to
    _BiRefNet.cpu = _Stub.cpu

    from trellis2.pipelines import Trellis2TexturingPipeline
    import trellis2.modules.sparse as sp
    from fractions import Fraction

    pipe = Trellis2TexturingPipeline.from_pretrained(args.model_id, config_file=args.config)
    # Move only decoder to cuda; skip others (incl. stubbed cond).
    dec = pipe.models['tex_slat_decoder'].cuda()

    feats = torch.from_numpy(np.load(os.path.join(args.input_dir, 'tex_slat_feats.npy'))).cuda()
    coords = torch.from_numpy(np.load(os.path.join(args.input_dir, 'tex_slat_coords.npy'))).cuda().int()
    # _scale = (16,16,16) at tex_slat: coarsest grid fed into the decoder.
    tex_slat = sp.SparseTensor(feats=feats, coords=coords,
                               scale=(Fraction(16,1), Fraction(16,1), Fraction(16,1)))

    # Reconstruct spatial cache. Real key format is str(self._scale), e.g.
    # "(Fraction(16, 1), Fraction(16, 1), Fraction(16, 1))".
    cache = {}
    for scale in [16, 8, 4, 2]:
        xc = np.load(os.path.join(args.input_dir, f'cache_scale{scale}_c2s_x_coords.npy'))
        idx = np.load(os.path.join(args.input_dir, f'cache_scale{scale}_c2s_idx.npy'))
        sub = np.load(os.path.join(args.input_dir, f'cache_scale{scale}_c2s_subidx.npy'))
        f = Fraction(scale, 1)
        key = str((f, f, f))
        cache[key] = {'channel2spatial_2': (
            torch.from_numpy(xc).cuda().int(),
            torch.from_numpy(idx).cuda(),
            torch.from_numpy(sub).cuda(),
        )}
    tex_slat._spatial_cache = cache

    def _save(name, t):
        if torch.is_tensor(t): t = t.detach().float().cpu().numpy()
        else: t = np.asarray(t).astype(np.float32) if t.dtype != np.int32 and t.dtype != np.int64 else t
        np.save(os.path.join(args.output_dir, name), t)

    with torch.no_grad():
        h = dec.from_latent(tex_slat).type(dec.dtype)
        _save('stage0_from_latent_feats.npy', h.feats)
        for i, res in enumerate(dec.blocks):
            is_last = (i == len(dec.blocks) - 1)
            for j, block in enumerate(res):
                if not is_last and j == len(res) - 1:
                    _save(f'stage{i}_pre_c2s_feats.npy', h.feats)
                    _save(f'stage{i}_pre_c2s_coords.npy', h.coords)
                    h = block(h)
                else:
                    h = block(h)
            if is_last:
                _save(f'stage{i}_pre_out_feats.npy', h.feats)
                _save(f'stage{i}_pre_out_coords.npy', h.coords)
            else:
                _save(f'stage{i}_post_c2s_feats.npy', h.feats)
                _save(f'stage{i}_post_c2s_coords.npy', h.coords)
            print(f'stage {i}: N={h.coords.shape[0]} C={h.feats.shape[1]} '
                  f'min={h.feats.min().item():.3f} max={h.feats.max().item():.3f}')

        h = h.type(tex_slat.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        _save('pre_output_ln_feats.npy', h.feats)
        h = dec.output_layer(h)
        pbr = h * 0.5 + 0.5
        # h here is SparseTensor; scale its feats.
        _save('pbr_voxel_feats.npy', pbr.feats)
        _save('pbr_voxel_coords.npy', pbr.coords)
        print(f'final: N={pbr.coords.shape[0]} C={pbr.feats.shape[1]} '
              f'min={pbr.feats.min().item():.3f} max={pbr.feats.max().item():.3f}')

if __name__ == '__main__':
    main()
