"""TRELLIS.2 image -> textured GLB driver (CUDA, 16 GB VRAM friendly).

Mirrors upstream `trellis2_repo/example.py` but:
  - loads weights from a local model_root with a 512-only pipeline.json
  - patches DinoV3FeatureExtractor to load timm DINOv3 weights from local file
  - patches BiRefNet to no-op (input must be RGBA with real alpha)
  - exports both pre-texture mesh.obj and final textured.glb

Defaults assume the layout produced by run_image_to_3d.sh.
"""
import argparse
import os
import sys
import gc

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TRELLIS2_REPO = os.path.join(_REPO_ROOT, 'cpu', 'trellis2', 'trellis2_repo')
RDNA4_DIR = os.path.join(_REPO_ROOT, 'rdna4', 'trellis2')
sys.path.insert(0, TRELLIS2_REPO)
sys.path.insert(0, RDNA4_DIR)

import numpy as np
import torch
import trimesh
from PIL import Image
from safetensors.torch import load_file


def _patch_dinov3_extractor(timm_path: str):
    from trellis2.modules import image_feature_extractor as _ife
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
        layer_prefix = 'model.layer' if hasattr(model, 'model') else 'layer'
        for i in range(24):
            tp, hp = f'blocks.{i}.', f'{layer_prefix}.{i}.'
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
        m = self.model
        image = image.to(m.embeddings.patch_embeddings.weight.dtype)
        h = m.embeddings(image, bool_masked_pos=None)
        pe = m.rope_embeddings(image)
        layers = m.model.layer if hasattr(m, 'model') else m.layer
        for layer in layers:
            h = layer(h, position_embeddings=pe)
        return F.layer_norm(h, h.shape[-1:])

    _ife.DinoV3FeatureExtractor.__init__ = _init
    _ife.DinoV3FeatureExtractor.extract_features = _extract


def _patch_birefnet_noop():
    from trellis2.pipelines.rembg.BiRefNet import BiRefNet as _BiRefNetClass

    def _init(self, *a, **kw):
        self.model = None
    def _call(self, image):
        raise RuntimeError('BiRefNet rembg disabled; pass an RGBA image with real alpha.')
    def _to(self, *a, **kw):
        pass

    _BiRefNetClass.__init__ = _init
    _BiRefNetClass.__call__ = _call
    _BiRefNetClass.to = _to
    _BiRefNetClass.cpu = lambda self: None


def _vram(tag):
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.memory_reserved() / 2**30
    print(f'[vram {tag}] alloc={a:.2f} GiB reserved={r:.2f} GiB', flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--model-root', required=True,
                    help='Local dir with pipeline.json + ckpts/')
    ap.add_argument('--dinov3', required=True,
                    help='timm DINOv3 ViT-L/16 safetensors')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pipeline-type', default='512',
                    choices=['512', '1024', '1024_cascade', '1536_cascade'])
    ap.add_argument('--texture-size', type=int, default=1024)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _patch_dinov3_extractor(args.dinov3)
    _patch_birefnet_noop()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    print(f'[load] {args.model_root}', flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_root)
    pipeline.cuda()
    _vram('after_load')

    image = Image.open(args.image)
    if image.mode != 'RGBA':
        # Synthesize alpha so BiRefNet (patched to error) is skipped.
        rgb = np.array(image.convert('RGB'))
        alpha = ((rgb.sum(-1) > 15) * 255).astype(np.uint8)
        image = Image.fromarray(np.dstack([rgb, alpha]), 'RGBA')

    torch.manual_seed(args.seed)
    print(f'[run] pipeline_type={args.pipeline_type} seed={args.seed}', flush=True)
    out = pipeline.run(image, seed=args.seed, pipeline_type=args.pipeline_type)
    mesh = out[0]
    _vram('after_run')

    # Pre-texture mesh as OBJ.
    v = mesh.vertices.detach().cpu().numpy()
    f = mesh.faces.detach().cpu().numpy()
    trimesh.Trimesh(v, f, process=False).export(
        os.path.join(args.output_dir, 'mesh.obj')
    )
    print(f'[mesh] verts={v.shape[0]} faces={f.shape[0]}', flush=True)

    # nvdiffrast tri budget; example.py uses 16777216
    mesh.simplify(16777216)

    print('[bake] o_voxel.postprocess.to_glb', flush=True)
    glb = o_voxel.postprocess.to_glb(
        vertices            = mesh.vertices,
        faces               = mesh.faces,
        attr_volume         = mesh.attrs,
        coords              = mesh.coords,
        attr_layout         = mesh.layout,
        voxel_size          = mesh.voxel_size,
        aabb                = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target   = 1000000,
        texture_size        = args.texture_size,
        remesh              = True,
        remesh_band         = 1,
        remesh_project      = 0,
        verbose             = True,
    )
    glb_path = os.path.join(args.output_dir, 'textured.glb')
    glb.export(glb_path, extension_webp=False)
    print(f'[done] {glb_path}', flush=True)
    print(f'[vram peak] {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)

    # Extract textures from the GLB for easy visual diff.
    s = trimesh.load(glb_path, process=False)
    geom = list(s.geometry.values())[0]
    m = geom.visual.material
    if getattr(m, 'baseColorTexture', None) is not None:
        m.baseColorTexture.save(os.path.join(args.output_dir, 'basecolor.png'))
    if getattr(m, 'metallicRoughnessTexture', None) is not None:
        m.metallicRoughnessTexture.save(os.path.join(args.output_dir, 'metalrough.png'))


if __name__ == '__main__':
    main()
