"""TRELLIS.2 ground-truth dumper (CUDA reference path).

Replicates Trellis2ImageTo3DPipeline.run() for pipeline_type='512' and writes
intermediate tensors at each stage boundary as .npy files for diffing against
the ROCm/HIP path.

All dumps land in --output-dir (default cuda/trellis2/verify-dumps/) along with
a manifest.json describing shape, dtype, sha256, and provenance for each entry.

Determinism: seed is fixed (default 42) and torch.use_deterministic_algorithms
is requested where possible. Sparse-attention backend = flash_attn (sm_120).
"""
import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

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
from PIL import Image

# Reuse DINOv3 / BiRefNet patches from the e2e driver.
from gen_image_to_3d import _patch_dinov3_extractor, _patch_birefnet_noop


MANIFEST: list[dict] = []


def _sha256(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _stat(arr: np.ndarray) -> dict:
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return {}
    a = arr.astype(np.float64, copy=False)
    return {
        'min': float(a.min()),
        'max': float(a.max()),
        'mean': float(a.mean()),
        'std': float(a.std()),
    }


def dump(out_dir: str, name: str, tensor, *, note: str = '') -> None:
    """Save a torch / numpy tensor as .npy and append a manifest entry."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    path = os.path.join(out_dir, f'{name}.npy')
    np.save(path, arr)
    entry = {
        'name': name,
        'file': f'{name}.npy',
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
        'sha256': _sha256(arr),
        'nbytes': int(arr.nbytes),
        **_stat(arr),
    }
    if note:
        entry['note'] = note
    MANIFEST.append(entry)
    print(f'[dump] {name:48s} shape={arr.shape} dtype={arr.dtype}', flush=True)


def dump_sparse(out_dir: str, prefix: str, st, *, note: str = '') -> None:
    """Save SparseTensor coords + feats."""
    dump(out_dir, f'{prefix}.coords', st.coords, note=note + ' (coords [B,x,y,z] int)')
    dump(out_dir, f'{prefix}.feats', st.feats, note=note + ' (feats [N,C] float)')


def dump_step_list(out_dir: str, prefix: str, items, *, max_steps: int | None = None) -> None:
    """Save per-diffusion-step tensors (pred_x_t / pred_x_0)."""
    n = len(items) if max_steps is None else min(len(items), max_steps)
    for i in range(n):
        x = items[i]
        if hasattr(x, 'feats'):  # SparseTensor
            dump(out_dir, f'{prefix}.step{i:03d}.feats', x.feats)
        else:
            dump(out_dir, f'{prefix}.step{i:03d}', x)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--model-root', required=True)
    ap.add_argument('--dinov3', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pipeline-type', default='512', choices=['512'],
                    help='Only 512 supported in dump driver (16 GB VRAM target).')
    ap.add_argument('--dump-per-step', action='store_true',
                    help='Also dump per-diffusion-step latents (pred_x_t).')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    _patch_dinov3_extractor(args.dinov3)
    _patch_birefnet_noop()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor

    print(f'[load] {args.model_root}', flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_root)
    pipeline.cuda()

    # ---- 0. Image preprocessing (alpha-synthesized, no rembg) ----
    img = Image.open(args.image)
    if img.mode != 'RGBA':
        rgb = np.array(img.convert('RGB'))
        alpha = ((rgb.sum(-1) > 15) * 255).astype(np.uint8)
        img = Image.fromarray(np.dstack([rgb, alpha]), 'RGBA')
    img_pp = pipeline.preprocess_image(img)
    dump(args.output_dir, '00_image_preprocessed',
         np.array(img_pp), note='RGB uint8 after preprocess_image()')

    torch.manual_seed(args.seed)

    # ---- 1. DINOv3 conditioning (512) ----
    cond_512 = pipeline.get_cond([img_pp], 512)
    dump(args.output_dir, '01_dinov3_cond_512', cond_512['cond'],
         note='DINOv3 ViT-L/16 features at image_size=512, [1,N,1024] bf16')
    dump(args.output_dir, '01_dinov3_neg_cond_512', cond_512['neg_cond'],
         note='Zero negative cond, same shape as cond')

    # ---- 2. Sparse structure: latent + coords ----
    torch.manual_seed(args.seed)  # match pipeline.run() ordering
    flow_ss = pipeline.models['sparse_structure_flow_model']
    reso, in_ch = flow_ss.resolution, flow_ss.in_channels
    noise_ss = torch.randn(1, in_ch, reso, reso, reso).to(pipeline.device)
    dump(args.output_dir, '02_ss_noise', noise_ss,
         note=f'Initial noise for sparse-structure flow, shape [1,{in_ch},{reso}^3]')
    if pipeline.low_vram:
        flow_ss.to(pipeline.device)
    ss_params = {**pipeline.sparse_structure_sampler_params}
    ss_out = pipeline.sparse_structure_sampler.sample(
        flow_ss, noise_ss, **cond_512, **ss_params,
        verbose=True, tqdm_desc='Sampling sparse structure',
    )
    z_s = ss_out.samples
    dump(args.output_dir, '03_ss_latent', z_s,
         note='Sparse-structure latent z_s after diffusion (dense grid)')
    if args.dump_per_step:
        dump_step_list(args.output_dir, '03_ss_pred_x_t', ss_out.pred_x_t)
    if pipeline.low_vram:
        flow_ss.cpu()

    decoder = pipeline.models['sparse_structure_decoder']
    if pipeline.low_vram:
        decoder.to(pipeline.device)
    decoded = decoder(z_s)
    dump(args.output_dir, '04_ss_decoder_logits', decoded,
         note='Sparse-structure decoder output (pre-threshold logits)')
    occ = decoded > 0
    if 32 != decoded.shape[2]:
        ratio = decoded.shape[2] // 32
        occ = torch.nn.functional.max_pool3d(occ.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(occ)[:, [0, 2, 3, 4]].int()
    dump(args.output_dir, '05_ss_coords', coords,
         note='Occupancy coords [N,4]=(B,x,y,z), int32, resolution=32')
    if pipeline.low_vram:
        decoder.cpu()

    # ---- 3. Shape SLat ----
    flow_shape = pipeline.models['shape_slat_flow_model_512']
    torch.manual_seed(args.seed + 1)  # match implicit ordering in pipeline.run
    noise_shape_feats = torch.randn(coords.shape[0], flow_shape.in_channels).to(pipeline.device)
    dump(args.output_dir, '06_shape_slat_noise_feats', noise_shape_feats,
         note=f'Initial noise feats for shape SLat, [N,{flow_shape.in_channels}]')
    noise_shape = SparseTensor(feats=noise_shape_feats, coords=coords)
    if pipeline.low_vram:
        flow_shape.to(pipeline.device)

    # ---- 3a. Single-step SLAT DiT dump (for verify_slat_dit) ----
    # One forward pass on the noise input at t=0.5 with positive cond,
    # so a CPU/HIP port can verify a single block-30 forward without re-running
    # the full sampler. Captures (x_t, t, cond, velocity).
    with torch.no_grad():
        t_step = torch.tensor([0.5], device=pipeline.device, dtype=torch.float32)
        cond_pos = cond_512['cond']  # [1, N_cond, 1024]
        v_step = flow_shape(noise_shape, t_step, cond_pos)
    dump(args.output_dir, '06b_slat_dit_step_x_t', noise_shape_feats,
         note='SLAT-DiT single-step input feats x_t = noise_shape_feats')
    dump(args.output_dir, '06b_slat_dit_step_coords', coords,
         note='SLAT-DiT single-step coords [N,4] (b,z,y,x)')
    dump(args.output_dir, '06b_slat_dit_step_t', t_step,
         note='SLAT-DiT single-step timestep t=0.5')
    dump(args.output_dir, '06b_slat_dit_step_cond', cond_pos,
         note='SLAT-DiT single-step positive cond [1,N_cond,1024]')
    dump(args.output_dir, '06b_slat_dit_step_velocity', v_step.feats,
         note='SLAT-DiT single-step output velocity [N,32]')

    shape_params = {**pipeline.shape_slat_sampler_params}
    shape_out = pipeline.shape_slat_sampler.sample(
        flow_shape, noise_shape, **cond_512, **shape_params,
        verbose=True, tqdm_desc='Sampling shape SLat',
    )
    shape_slat_raw = shape_out.samples
    dump(args.output_dir, '07_shape_slat_raw_feats', shape_slat_raw.feats,
         note='Shape SLat feats after diffusion, BEFORE std/mean denormalization')
    if args.dump_per_step:
        dump_step_list(args.output_dir, '07_shape_slat_pred_x_t', shape_out.pred_x_t)
    if pipeline.low_vram:
        flow_shape.cpu()

    std = torch.tensor(pipeline.shape_slat_normalization['std'])[None].to(shape_slat_raw.device)
    mean = torch.tensor(pipeline.shape_slat_normalization['mean'])[None].to(shape_slat_raw.device)
    shape_slat = shape_slat_raw * std + mean
    dump(args.output_dir, '08_shape_slat_denorm_feats', shape_slat.feats,
         note='Shape SLat feats after denormalization (= raw*std+mean)')

    # ---- 4. Texture SLat ----
    flow_tex = pipeline.models['tex_slat_flow_model_512']
    shape_for_tex_feats = (shape_slat.feats - mean) / std  # match sample_tex_slat
    in_ch_tex = flow_tex.in_channels
    cat_dim = in_ch_tex - shape_for_tex_feats.shape[1]
    torch.manual_seed(args.seed + 2)
    noise_tex_feats = torch.randn(shape_slat.coords.shape[0], cat_dim).to(pipeline.device)
    dump(args.output_dir, '09_tex_slat_noise_feats', noise_tex_feats,
         note=f'Initial noise feats for texture SLat, [N,{cat_dim}]')
    noise_tex = shape_slat.replace(feats=noise_tex_feats)
    concat_cond = shape_slat.replace(feats=shape_for_tex_feats)
    dump(args.output_dir, '10_tex_concat_cond_feats', concat_cond.feats,
         note='Re-normalized shape SLat used as concat_cond into tex flow')

    if pipeline.low_vram:
        flow_tex.to(pipeline.device)
    tex_params = {**pipeline.tex_slat_sampler_params}
    tex_out = pipeline.tex_slat_sampler.sample(
        flow_tex, noise_tex, concat_cond=concat_cond, **cond_512, **tex_params,
        verbose=True, tqdm_desc='Sampling texture SLat',
    )
    tex_slat_raw = tex_out.samples
    dump(args.output_dir, '11_tex_slat_raw_feats', tex_slat_raw.feats,
         note='Tex SLat feats after diffusion, BEFORE std/mean denormalization')
    if args.dump_per_step:
        dump_step_list(args.output_dir, '11_tex_slat_pred_x_t', tex_out.pred_x_t)
    if pipeline.low_vram:
        flow_tex.cpu()

    std_t = torch.tensor(pipeline.tex_slat_normalization['std'])[None].to(tex_slat_raw.device)
    mean_t = torch.tensor(pipeline.tex_slat_normalization['mean'])[None].to(tex_slat_raw.device)
    tex_slat = tex_slat_raw * std_t + mean_t
    dump(args.output_dir, '12_tex_slat_denorm_feats', tex_slat.feats,
         note='Tex SLat feats after denormalization')

    torch.cuda.empty_cache()
    gc.collect()

    # ---- 5. Decode shape SLat -> mesh + substructures ----
    res = 512
    pipeline.models['shape_slat_decoder'].set_resolution(res)
    if pipeline.low_vram:
        pipeline.models['shape_slat_decoder'].to(pipeline.device)
        pipeline.models['shape_slat_decoder'].low_vram = True
    meshes, subs = pipeline.models['shape_slat_decoder'](shape_slat, return_subs=True)
    if pipeline.low_vram:
        pipeline.models['shape_slat_decoder'].cpu()
        pipeline.models['shape_slat_decoder'].low_vram = False
    mesh = meshes[0]
    dump(args.output_dir, '13_mesh_vertices', mesh.vertices,
         note='Decoded mesh vertices [V,3] float, world coords')
    dump(args.output_dir, '13_mesh_faces', mesh.faces,
         note='Decoded mesh faces [F,3] int')
    for i, s in enumerate(subs):
        dump_sparse(args.output_dir, f'14_shape_sub{i}', s,
                    note=f'Shape decoder substructure {i}')

    # ---- 6. Decode tex SLat -> voxel attrs ----
    if pipeline.low_vram:
        pipeline.models['tex_slat_decoder'].to(pipeline.device)
    tex_voxels = pipeline.models['tex_slat_decoder'](tex_slat, guide_subs=subs) * 0.5 + 0.5
    if pipeline.low_vram:
        pipeline.models['tex_slat_decoder'].cpu()
    dump_sparse(args.output_dir, '15_tex_voxels', tex_voxels,
                note='Decoded texture voxels (PBR attrs); '
                     'feats[:, :3]=basecolor, [3]=metallic, [4]=roughness, [5]=alpha')

    # ---- write manifest ----
    manifest = {
        'created': datetime.now(timezone.utc).isoformat(),
        'image': os.path.abspath(args.image),
        'model_root': os.path.abspath(args.model_root),
        'dinov3': os.path.abspath(args.dinov3),
        'pipeline_type': args.pipeline_type,
        'seed': args.seed,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'compute_capability': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        'sparse_attn_backend': os.environ.get('SPARSE_ATTN_BACKEND', 'flash_attn'),
        'shape_slat_normalization': pipeline.shape_slat_normalization,
        'tex_slat_normalization': pipeline.tex_slat_normalization,
        'entries': MANIFEST,
    }
    with open(os.path.join(args.output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f'[done] {len(MANIFEST)} dumps -> {args.output_dir}', flush=True)
    print(f'[vram peak] {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)


if __name__ == '__main__':
    main()
