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
import trimesh
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
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _patch_dinov3_extractor(args.dinov3)
    _patch_birefnet_noop()

    pipeline = Trellis2TexturingPipeline.from_pretrained(args.model_id, config_file=args.config)
    pipeline.to('cuda')

    # Install intermediate-dump hooks by calling stage methods manually.
    image = Image.open(args.image)
    mesh = trimesh.load(args.mesh, process=False)

    pre_image = pipeline.preprocess_image(image)
    pre_image.save(os.path.join(args.output_dir, 'preprocess_image.png'))

    pre_mesh = pipeline.preprocess_mesh(mesh)
    pre_mesh.export(os.path.join(args.output_dir, 'preprocess_mesh.obj'))

    import gc

    def _vram(tag):
        alloc = torch.cuda.memory_allocated() / 2**30
        reserv = torch.cuda.memory_reserved() / 2**30
        print(f'[vram {tag}] alloc={alloc:.2f} GiB reserved={reserv:.2f} GiB', flush=True)

    _vram('after_init')
    torch.manual_seed(args.seed)
    cond = pipeline.get_cond([pre_image], args.resolution)
    _save(args.output_dir, 'cond_features', cond['cond'])
    _save(args.output_dir, 'cond_neg_features', cond['neg_cond'])
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
    cache = getattr(shape_slat, '_spatial_cache', None) or {}
    for k, v in cache.items():
        if not k.startswith('channel2spatial_'):
            continue
        if not (isinstance(v, tuple) and len(v) == 3):
            continue
        new_coords, idx, subidx = v
        _save(args.output_dir, f'cache_{k}_coords', new_coords)
        _save(args.output_dir, f'cache_{k}_idx', idx)
        _save(args.output_dir, f'cache_{k}_subidx', subidx)
        print(f'[cache] {k}: coords={tuple(new_coords.shape)} idx={tuple(idx.shape)} '
              f'subidx={tuple(subidx.shape)}', flush=True)

    gc.collect(); torch.cuda.empty_cache()
    _vram('after_shape_slat')

    tex_model = pipeline.models[
        'tex_slat_flow_model_512' if args.resolution == 512 else 'tex_slat_flow_model_1024'
    ]
    tex_slat = pipeline.sample_tex_slat(cond, tex_model, shape_slat, {})
    _save(args.output_dir, 'tex_slat_feats', tex_slat.feats)
    _save(args.output_dir, 'tex_slat_coords', tex_slat.coords)
    torch.cuda.empty_cache()

    pbr_voxel = pipeline.decode_tex_slat(tex_slat)
    _save(args.output_dir, 'pbr_voxel_feats', pbr_voxel.feats)
    _save(args.output_dir, 'pbr_voxel_coords', pbr_voxel.coords)

    textured = pipeline.postprocess_mesh(pre_mesh, pbr_voxel, args.resolution, args.texture_size)
    out_glb = os.path.join(args.output_dir, 'textured.glb')
    textured.export(out_glb)
    print('wrote', out_glb)


if __name__ == '__main__':
    main()
