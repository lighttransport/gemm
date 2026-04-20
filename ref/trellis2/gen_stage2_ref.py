"""PyTorch-ROCm reference driver for TRELLIS.2 Stage 2 (texturing).

Loads a mesh (from Stage 1) + image, runs the texturing pipeline end-to-end,
saves intermediate tensors + a final textured GLB.

Prereqs:
  - venv: /mnt/disk1/work/gemm/main/rdna4/trellis2/.venv (torch 2.11+rocm7.2)
  - shims: rdna4/trellis2/texgen_sw_rast.py + cumesh_xatlas_shim.py
  - env: ATTN_BACKEND=sdpa (set automatically)
"""

import argparse
import os
import sys

os.environ.setdefault('ATTN_BACKEND', 'sdpa')

RDNA4_DIR = '/mnt/disk1/work/gemm/main/rdna4/trellis2'
REPO_DIR = '/mnt/disk1/work/gemm/main/cpu/trellis2/trellis2_repo'
sys.path.insert(0, RDNA4_DIR)
sys.path.insert(0, REPO_DIR)

import texgen_sw_rast
import cumesh_xatlas_shim
import flash_attn_sdpa_shim
texgen_sw_rast.install_as_nvdiffrast()
cumesh_xatlas_shim.install_as_cumesh()
# Patch transformers' package-distribution map so `is_flash_attn_2_available()`
# doesn't KeyError on our synthetic flash_attn module.
from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
flash_attn_sdpa_shim.install_as_flash_attn()

import numpy as np
import torch
try:
    import trimesh
except ModuleNotFoundError:
    trimesh = None
from PIL import Image
from safetensors.torch import load_file

from trellis2.modules import image_feature_extractor as _ife
from trellis2.pipelines.rembg.BiRefNet import BiRefNet as _BiRefNetClass


def _patch_birefnet_noop():
    """Avoid gated HF fetch of briaai/RMBG-2.0 by making BiRefNet.__init__ a no-op.

    Safe when input images are already RGBA with real alpha (no bg removal needed).
    If BiRefNet is actually invoked (RGB input), we raise a clear error.
    """
    def _init(self, *a, **kw):
        self.model = None
    def _call(self, image):
        raise RuntimeError('BiRefNet rembg disabled; pass an RGBA image with real alpha.')
    def _to(self, *a, **kw): pass
    def _cpu(self): pass
    _BiRefNetClass.__init__ = _init
    _BiRefNetClass.__call__ = _call
    _BiRefNetClass.to = _to
    _BiRefNetClass.cpu = _cpu


def _patch_dinov3_extractor(timm_path: str):
    """Replace HF-gated `DinoV3FeatureExtractor.__init__` with local timm loader."""
    from transformers import DINOv3ViTModel, DINOv3ViTConfig

    def _init(self, model_name: str, image_size: int = 1024):
        self.model_name = model_name
        self.image_size = image_size
        config = DINOv3ViTConfig(
            hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
            intermediate_size=4096, patch_size=16, image_size=image_size,
            num_register_tokens=4, rope_theta=100.0,
        )
        model = DINOv3ViTModel(config)
        timm_sd = load_file(timm_path)
        new_sd = {
            'embeddings.cls_token': timm_sd['cls_token'],
            'embeddings.register_tokens': timm_sd['reg_token'],
            'embeddings.patch_embeddings.weight': timm_sd['patch_embed.proj.weight'],
            'embeddings.patch_embeddings.bias': timm_sd['patch_embed.proj.bias'],
        }
        for i in range(24):
            tp, hp = f'blocks.{i}.', f'model.layer.{i}.'
            new_sd[f'{hp}norm1.weight'] = timm_sd[f'{tp}norm1.weight']
            new_sd[f'{hp}norm1.bias']   = timm_sd[f'{tp}norm1.bias']
            new_sd[f'{hp}norm2.weight'] = timm_sd[f'{tp}norm2.weight']
            new_sd[f'{hp}norm2.bias']   = timm_sd[f'{tp}norm2.bias']
            q_w, k_w, v_w = timm_sd[f'{tp}attn.qkv.weight'].chunk(3, dim=0)
            new_sd[f'{hp}attention.q_proj.weight'] = q_w
            new_sd[f'{hp}attention.k_proj.weight'] = k_w
            new_sd[f'{hp}attention.v_proj.weight'] = v_w
            new_sd[f'{hp}attention.q_proj.bias'] = torch.zeros(1024)
            new_sd[f'{hp}attention.v_proj.bias'] = torch.zeros(1024)
            new_sd[f'{hp}attention.o_proj.weight'] = timm_sd[f'{tp}attn.proj.weight']
            new_sd[f'{hp}attention.o_proj.bias']   = timm_sd[f'{tp}attn.proj.bias']
            new_sd[f'{hp}layer_scale1.lambda1']    = timm_sd[f'{tp}gamma_1']
            new_sd[f'{hp}layer_scale2.lambda1']    = timm_sd[f'{tp}gamma_2']
            new_sd[f'{hp}mlp.up_proj.weight']   = timm_sd[f'{tp}mlp.fc1.weight']
            new_sd[f'{hp}mlp.up_proj.bias']     = timm_sd[f'{tp}mlp.fc1.bias']
            new_sd[f'{hp}mlp.down_proj.weight'] = timm_sd[f'{tp}mlp.fc2.weight']
            new_sd[f'{hp}mlp.down_proj.bias']   = timm_sd[f'{tp}mlp.fc2.bias']
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        self.model = model
        from torchvision import transforms as _tx
        self.transform = _tx.Compose([
            _tx.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _extract(self, image):
        import torch.nn.functional as F
        m = self.model  # DINOv3ViTModel: .embeddings, .rope_embeddings, .model.layer
        image = image.to(m.embeddings.patch_embeddings.weight.dtype)
        h = m.embeddings(image, bool_masked_pos=None)
        pe = m.rope_embeddings(image)
        for layer in m.model.layer:
            h = layer(h, position_embeddings=pe)
        return F.layer_norm(h, h.shape[-1:])

    _ife.DinoV3FeatureExtractor.__init__ = _init
    _ife.DinoV3FeatureExtractor.extract_features = _extract


from trellis2.pipelines import Trellis2TexturingPipeline


def _save(od, name, t):
    if torch.is_tensor(t):
        # Cast to fp32 for float tensors so C-side read_npy_f32 consumers
        # (e.g. test_hip_tex_dec) don't misinterpret fp16 bytes. Keep int
        # tensors as-is.
        if t.dtype.is_floating_point:
            t = t.detach().float().cpu().numpy()
        else:
            t = t.detach().cpu().numpy()
    np.save(os.path.join(od, name), t)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', required=True, help='Input mesh (.obj/.ply/.glb)')
    ap.add_argument('--image', required=True, help='Conditioning image')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--resolution', type=int, default=1024, choices=[512, 1024])
    ap.add_argument('--texture-size', type=int, default=1024)
    ap.add_argument('--model-id', default='microsoft/TRELLIS.2-4B')
    ap.add_argument('--config', default='texturing_pipeline.json')
    ap.add_argument('--dinov3', default='/mnt/disk1/models/dinov3-vitl16/model.safetensors',
                    help='Local timm DINOv3 ViT-L/16 safetensors (avoids HF gated repo)')
    ap.add_argument('--dump-stages', action='store_true',
                    help='Dump tex_slat_decoder per-stage hidden feats for HIP bisection')
    ap.add_argument('--skip-dit', action='store_true',
                    help='Reuse cached tex_slat_{feats,coords}.npy from --output-dir; '
                         'still runs shape encoder so the decoder inherits the real '
                         '_spatial_cache. Huge speedup for decoder-only bisection.')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _patch_dinov3_extractor(args.dinov3)
    _patch_birefnet_noop()

    pipeline = Trellis2TexturingPipeline.from_pretrained(args.model_id, config_file=args.config)
    pipeline.to('cuda')

    # Install intermediate-dump hooks by calling stage methods manually.
    mesh = trimesh.load(args.mesh, process=False)

    if not args.skip_dit:
        image = Image.open(args.image)
        pre_image = pipeline.preprocess_image(image)
        pre_image.save(os.path.join(args.output_dir, 'preprocess_image.png'))
    else:
        pre_image = None

    pre_mesh = pipeline.preprocess_mesh(mesh)
    pre_mesh.export(os.path.join(args.output_dir, 'preprocess_mesh.obj'))

    import gc

    def _vram(tag):
        alloc = torch.cuda.memory_allocated() / 2**30
        reserv = torch.cuda.memory_reserved() / 2**30
        print(f'[vram {tag}] alloc={alloc:.2f} GiB reserved={reserv:.2f} GiB', flush=True)

    _vram('after_init')
    torch.manual_seed(args.seed)
    if not args.skip_dit:
        cond = pipeline.get_cond([pre_image], args.resolution)
        _save(args.output_dir, 'cond_features', cond['cond'])
        _save(args.output_dir, 'cond_neg_features', cond['neg_cond'])
    else:
        cond = None
    gc.collect(); torch.cuda.empty_cache()
    _vram('after_cond')

    shape_slat = pipeline.encode_shape_slat(pre_mesh, args.resolution)
    print(f'[shape_slat] feats={tuple(shape_slat.feats.shape)} dtype={shape_slat.feats.dtype} '
          f'coords={tuple(shape_slat.coords.shape)}', flush=True)
    _save(args.output_dir, 'shape_slat_feats', shape_slat.feats)
    _save(args.output_dir, 'shape_slat_coords', shape_slat.coords)

    # Dump the subdivision caches the encoder populated on shape_slat's
    # _spatial_cache — the tex decoder's C2S blocks consume these (see
    # SparseChannel2Spatial.forward). Each entry is (new_coords, idx, subidx).
    # _spatial_cache is {scale_tuple_str: {op_name: value}}. We want
    # 'channel2spatial_2' entries at every scale so the CPU tex decoder can
    # reproduce the pruned 8x expansion without running the shape encoder.
    # We tag each dump with its coarse scale (scale of the tensor holding it).
    import re
    cache = getattr(shape_slat, '_spatial_cache', None) or {}
    for scale_key, sub in cache.items():
        if not isinstance(sub, dict):
            continue
        m = re.search(r'Fraction\((\d+),\s*1\)', scale_key)
        if not m:
            continue
        scale = int(m.group(1))
        v = sub.get('channel2spatial_2')
        if not (isinstance(v, tuple) and len(v) == 3):
            continue
        x_coords, idx, subidx = v
        tag = f'cache_scale{scale}_c2s'
        _save(args.output_dir, f'{tag}_x_coords', x_coords)
        _save(args.output_dir, f'{tag}_idx', idx)
        _save(args.output_dir, f'{tag}_subidx', subidx)
        print(f'[cache] scale={scale} c2s: N_fine={x_coords.shape[0]}', flush=True)

    gc.collect(); torch.cuda.empty_cache()
    _vram('after_shape_slat')

    tex_model = pipeline.models[
        'tex_slat_flow_model_512' if args.resolution == 512 else 'tex_slat_flow_model_1024'
    ]
    if args.skip_dit:
        # Reuse cached DiT output. Start from shape_slat (which carries the real
        # _spatial_cache from the shape encoder), then overwrite feats/coords
        # with the cached tensors. sample_tex_slat normally re-uses shape_slat's
        # structure, so this preserves cache identity.
        import trellis2.modules.sparse as sp
        cached_feats = torch.from_numpy(
            np.load(os.path.join(args.output_dir, 'tex_slat_feats.npy'))).cuda()
        cached_coords = torch.from_numpy(
            np.load(os.path.join(args.output_dir, 'tex_slat_coords.npy'))).cuda().int()
        tex_slat = sp.SparseTensor(feats=cached_feats, coords=cached_coords,
                                   scale=shape_slat._scale)
        tex_slat._spatial_cache = shape_slat._spatial_cache
        print(f'[skip-dit] reused tex_slat: N={cached_feats.shape[0]} C={cached_feats.shape[1]}',
              flush=True)
    else:
        tex_slat = pipeline.sample_tex_slat(cond, tex_model, shape_slat, {})
        _save(args.output_dir, 'tex_slat_feats', tex_slat.feats)
        _save(args.output_dir, 'tex_slat_coords', tex_slat.coords)
    torch.cuda.empty_cache()

    # Optional per-stage hidden-state dumps for HIP bisection.
    if args.dump_stages:
        import torch.nn.functional as F
        dec = pipeline.models['tex_slat_decoder']
        dec.to('cuda')
        orig = dec.forward
        def hooked(x, guide_subs=None, return_subs=False):
            h = dec.from_latent(x).type(dec.dtype)
            _save(args.output_dir, 'stage0_from_latent_feats', h.feats)

            # Sub-op dump inside stage 0 block 0 for HIP bisection.
            b0 = dec.blocks[0][0]
            hc = b0.conv(h)
            _save(args.output_dir, 'stage0_b0_post_conv_feats', hc.feats)
            # Dump flex_gemm's neighbor cache for this conv so HIP can reproduce
            # its neighbor-indexing convention.
            _nc_key = 'SubMConv3d_neighbor_cache_3x3x3_dilation(1, 1, 1)'
            _nc = h.get_spatial_cache(_nc_key)
            if _nc is not None:
                print(f'[neighbor_cache] type={type(_nc).__name__}', flush=True)
                if isinstance(_nc, (tuple, list)):
                    for idx, item in enumerate(_nc):
                        if torch.is_tensor(item):
                            _save(args.output_dir, f'stage0_b0_nbr_cache_{idx}', item)
                            print(f'  [{idx}] tensor shape={tuple(item.shape)} dtype={item.dtype}', flush=True)
                        else:
                            print(f'  [{idx}] non-tensor: {item!r}', flush=True)
                elif torch.is_tensor(_nc):
                    _save(args.output_dir, 'stage0_b0_nbr_cache_0', _nc)
                    print(f'  tensor shape={tuple(_nc.shape)}', flush=True)
                else:
                    # Custom object (e.g. SubMConv3dNeighborCache). Walk ALL attrs.
                    print(f'  dir: {[a for a in dir(_nc) if not a.startswith("__")]}', flush=True)
                    for attr in dir(_nc):
                        if attr.startswith('__'):
                            continue
                        try:
                            val = getattr(_nc, attr)
                        except Exception as e:
                            print(f'    .{attr}: <err {e}>', flush=True)
                            continue
                        if torch.is_tensor(val):
                            _save(args.output_dir, f'stage0_b0_nbr_cache_{attr}', val)
                            print(f'    .{attr}: tensor shape={tuple(val.shape)} dtype={val.dtype}', flush=True)
                        elif callable(val):
                            continue
                        else:
                            print(f'    .{attr}: {type(val).__name__} = {val!r}', flush=True)
            hn = hc.replace(b0.norm(hc.feats))
            _save(args.output_dir, 'stage0_b0_post_ln_feats', hn.feats)
            hm = hn.replace(b0.mlp(hn.feats))
            _save(args.output_dir, 'stage0_b0_post_mlp_feats', hm.feats)

            _nc_key = 'SubMConv3d_neighbor_cache_3x3x3_dilation(1, 1, 1)'
            def _dump_nmap(tag, tensor):
                _nc = tensor.get_spatial_cache(_nc_key)
                if _nc is not None and hasattr(_nc, 'neighbor_map'):
                    _save(args.output_dir, f'{tag}_nmap', _nc.neighbor_map)
            for i, res in enumerate(dec.blocks):
                is_last = (i == len(dec.blocks) - 1)
                for j, block in enumerate(res):
                    if not is_last and j == len(res) - 1:
                        _save(args.output_dir, f'stage{i}_pre_c2s_feats', h.feats)
                        _save(args.output_dir, f'stage{i}_pre_c2s_coords', h.coords)
                        _dump_nmap(f'stage{i}_convnext', h)
                        if i == 0:
                            # Inline stage 0 C2S with per-op dumps for HIP bisection.
                            _dump_nmap(f'stage{i}_convnext', h)
                            x_in = h
                            hh = h.replace(block.norm1(h.feats))
                            hh = hh.replace(F.silu(hh.feats))
                            _save(args.output_dir, f'stage{i}_c2s_pre_conv1', hh.feats)
                            hh = block.conv1(hh)
                            _save(args.output_dir, f'stage{i}_c2s_post_conv1', hh.feats)
                            hh = block.updown(hh, None)
                            _save(args.output_dir, f'stage{i}_c2s_post_updown_h', hh.feats)
                            xx = block.updown(x_in, None)
                            _save(args.output_dir, f'stage{i}_c2s_post_updown_x', xx.feats)
                            hh = hh.replace(block.norm2(hh.feats))
                            hh = hh.replace(F.silu(hh.feats))
                            _save(args.output_dir, f'stage{i}_c2s_pre_conv2', hh.feats)
                            hh = block.conv2(hh)
                            _save(args.output_dir, f'stage{i}_c2s_post_conv2', hh.feats)
                            skip = block.skip_connection(xx)
                            _save(args.output_dir, f'stage{i}_c2s_skip', skip.feats)
                            h = hh + skip
                        else:
                            h = block(h, subdiv=guide_subs[i] if guide_subs else None)
                        _dump_nmap(f'stage{i}_post_c2s', h)
                        _save(args.output_dir, f'stage{i}_post_c2s_feats', h.feats)
                    else:
                        h = block(h)
                        # Per-block dumps for stage 0 bisection.
                        if i == 0:
                            _save(args.output_dir, f'stage0_block{j}_feats', h.feats)
                if is_last:
                    _dump_nmap(f'stage{i}_convnext', h)
                if is_last:
                    _save(args.output_dir, f'stage{i}_pre_out_feats', h.feats)
                    _save(args.output_dir, f'stage{i}_pre_out_coords', h.coords)
            print('[dtype] x.dtype', x.dtype, 'h.feats.dtype before cast', h.feats.dtype, flush=True)
            h = h.type(x.dtype)
            print('[dtype] after h.type(x.dtype):', h.feats.dtype, flush=True)
            _save(args.output_dir, 'final_pre_ln_feats', h.feats)
            h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
            _save(args.output_dir, 'final_post_ln_feats', h.feats)
            _save(args.output_dir, 'output_layer_weight', dec.output_layer.weight.detach())
            _save(args.output_dir, 'output_layer_bias', dec.output_layer.bias.detach())
            h_in = h.feats.detach().clone()
            W_ol = dec.output_layer.weight
            b_ol = dec.output_layer.bias
            print('[ol] h_in', h_in.dtype, h_in.device, 'contig', h_in.is_contiguous(),
                  'stride', h_in.stride(), 'W', W_ol.dtype, W_ol.device, flush=True)
            print('[ol] h_in stats:', h_in.min().item(), h_in.max().item(),
                  'any nan', torch.isnan(h_in).any().item(), flush=True)
            h = dec.output_layer(h)
            _save(args.output_dir, 'final_post_outlayer_feats', h.feats)
            import torch.nn.functional as F2
            manual = F2.linear(h_in, W_ol, b_ol)
            _save(args.output_dir, 'final_post_outlayer_manual', manual)
            manual_c = F2.linear(h_in.contiguous().float(), W_ol.float(), b_ol.float())
            print('[check] max|hooked - manual| =', (h.feats - manual).abs().max().item(),
                  'max|hooked - manual_contig_f32| =', (h.feats.float() - manual_c).abs().max().item(),
                  'manual_c stats', manual_c.min().item(), manual_c.max().item(), flush=True)
            return h
        dec.forward = hooked
        pbr_voxel = dec(tex_slat) * 0.5 + 0.5
        dec.forward = orig
    else:
        pbr_voxel = pipeline.decode_tex_slat(tex_slat)
    _save(args.output_dir, 'pbr_voxel_feats', pbr_voxel.feats)
    _save(args.output_dir, 'pbr_voxel_coords', pbr_voxel.coords)

    if args.skip_dit:
        print('[skip-dit] skipping postprocess_mesh (decoder dumps complete)', flush=True)
        return
    textured = pipeline.postprocess_mesh(pre_mesh, pbr_voxel, args.resolution, args.texture_size)
    out_glb = os.path.join(args.output_dir, 'textured.glb')
    textured.export(out_glb)
    print('wrote', out_glb)


if __name__ == '__main__':
    main()
