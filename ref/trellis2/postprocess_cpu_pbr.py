"""Run TRELLIS.2 postprocess_mesh on CPU-decoded PBR voxels.

Loads `<prefix>_feats.npy [N,6]` and `<prefix>_coords.npy [N,4]` from
`cpu/trellis2/test_tex_dec --dump <prefix>`, builds a SparseTensor, and
invokes the pipeline's postprocess_mesh (reusing the same ROCm shims as
gen_stage2_ref.py) to produce a textured GLB.
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
from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
flash_attn_sdpa_shim.install_as_flash_attn()

import numpy as np
import torch
import trimesh
from safetensors.torch import load_file

from trellis2.modules import image_feature_extractor as _ife
from trellis2.pipelines.rembg.BiRefNet import BiRefNet as _BiRefNetClass
from trellis2.modules.sparse import SparseTensor


def _patch_birefnet_noop():
    def _init(self, *a, **kw): self.model = None
    def _call(self, image):
        raise RuntimeError('BiRefNet rembg disabled.')
    _BiRefNetClass.__init__ = _init
    _BiRefNetClass.__call__ = _call
    _BiRefNetClass.to = lambda self, *a, **kw: None
    _BiRefNetClass.cpu = lambda self: None


def _patch_dinov3_extractor(timm_path: str):
    from transformers import DINOv3ViTModel, DINOv3ViTConfig

    def _init(self, model_name: str, image_size: int = 1024):
        self.model_name = model_name
        self.image_size = image_size
        cfg = DINOv3ViTConfig(
            hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
            intermediate_size=4096, patch_size=16, image_size=image_size,
            num_register_tokens=4, rope_theta=100.0,
        )
        model = DINOv3ViTModel(cfg)
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
    _ife.DinoV3FeatureExtractor.__init__ = _init

from trellis2.pipelines import Trellis2TexturingPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', required=True, help='Preprocessed mesh (e.g. preprocess_mesh.obj from gen_stage2_ref.py)')
    ap.add_argument('--prefix', required=True, help='Path prefix: <prefix>_feats.npy + <prefix>_coords.npy')
    ap.add_argument('--output', required=True, help='Output .glb path')
    ap.add_argument('--resolution', type=int, default=512, choices=[512, 1024])
    ap.add_argument('--texture-size', type=int, default=1024)
    ap.add_argument('--model-id', default='microsoft/TRELLIS.2-4B')
    ap.add_argument('--config', default='texturing_pipeline.json')
    ap.add_argument('--dinov3', default='/mnt/disk1/models/dinov3-vitl16/model.safetensors')
    args = ap.parse_args()

    _patch_dinov3_extractor(args.dinov3)
    _patch_birefnet_noop()

    feats = np.load(args.prefix + '_feats.npy')      # [N, 6] float32, post-affine
    coords = np.load(args.prefix + '_coords.npy')    # [N, 4] int32, [b, z, y, x]
    print(f'loaded feats={feats.shape} coords={coords.shape}', flush=True)

    pipeline = Trellis2TexturingPipeline.from_pretrained(args.model_id, config_file=args.config)
    pipeline.to('cuda')

    mesh = trimesh.load(args.mesh, process=False)

    feats_t = torch.from_numpy(feats).float().cuda()
    coords_t = torch.from_numpy(coords).int().cuda()
    # SparseTensor expects coords as [b, x, y, z] upstream convention; our dump
    # order matches the reference pbr_voxel_coords.npy (same layout) so pass
    # through verbatim.
    R = args.resolution
    shape = torch.Size([1, feats.shape[1]])
    pbr_voxel = SparseTensor(feats_t, coords_t, shape=shape)
    # Pin spatial_shape so grid_sample_3d normalizes against full grid, not
    # the coord-bbox (which may be smaller if outer voxels are empty).
    pbr_voxel.register_spatial_cache('shape', torch.Size([R, R, R]))

    textured = pipeline.postprocess_mesh(mesh, pbr_voxel, args.resolution, args.texture_size)
    textured.export(args.output)
    print('wrote', args.output)


if __name__ == '__main__':
    main()
