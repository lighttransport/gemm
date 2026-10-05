"""Bounded ROCm image preprocessing for the native H3 packed DiT.

Only Qwen's visual tower and the video VAE encoder run in PyTorch. The native
runner executes all language layers, the DiT and video decoding. Encoders are
loaded sequentially and reference latents never enter the Euler state.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = '2472a20bd291451acc303917059ab14dfc380478'
DEFAULT_COMFY = ROOT/'tmp/video-rocm/pytorch-bench-comfy'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(*, model, out, prompt, variant, images=(), first_frame=None, last_frame=None,
            width=480, height=832, frames=22, seed=42, device=0, vram_budget_mib=12288,
            comfy=DEFAULT_COMFY, model_receipt=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from safetensors import safe_open
    from tokenizers import Tokenizer
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('image preprocessing requires PyTorch ROCm and AMD device access')
    if variant not in ('ref2va', 'fl2va') or (variant == 'ref2va' and (first_frame or last_frame)):
        raise ValueError('reference images require ref2va; first/last frames require fl2va')
    if variant == 'fl2va' and images:
        raise ValueError('fl2va accepts first/last frames, not reference images')
    sources = list(images) if variant == 'ref2va' else [p for p in (first_frame, last_frame) if p]
    if not 1 <= len(sources) <= 9:
        raise ValueError('provide 1..9 reference images or 1..2 keyframes')
    if width < 64 or height < 64 or width % 32 or height % 32 or width*height > 1344*768:
        raise ValueError('invalid target dimensions')
    if frames < 5 or frames > 362 or (frames-5) % 17 or not 4096 <= vram_budget_mib <= 14336:
        raise ValueError('invalid frame count or VRAM budget')
    comfy = Path(comfy).resolve()
    revision = subprocess.check_output(['git', '-C', str(comfy), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM:
        raise ValueError('image preprocessing needs the pinned ComfyUI revision '+UPSTREAM)
    sys.path.insert(0, str(comfy))
    from comfy.text_encoders.qwen3vl import Qwen3VLVisionModel, QWEN3VL_VISION_COMMON, QWEN3VL_VISION
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images, qwen2vl_mrope_position_ids
    from comfy.text_encoders.llama import precompute_freqs_cis
    from comfy.text_encoders.minimax import token_tags_from_embeds_info
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    from comfy.ldm.minimax.model import PackedLayout, patchify_video
    from comfy import ops
    model, out = Path(model).resolve(), Path(out)
    component_names = (f'diffusion_models/minimax_h3_{variant}_pruned_int8_convrot.safetensors',
                       'text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
                       'vae/minimax_h3_video_vae_fp16.safetensors', 'tokenizer/tokenizer.json')
    snapshot = {name: (model/name).stat() for name in component_names}
    if model_receipt:
        verified = json.loads(Path(model_receipt).read_text())
        if verified['model'] != str(model):
            raise ValueError('encoder model receipt directory mismatch')
        for name, stamp in snapshot.items():
            if verified['snapshot'][name] != [stamp.st_size, stamp.st_mtime_ns, stamp.st_ino]:
                raise ValueError('encoder model changed after verification')
        components = verified['verified_components']
    else:
        components = {name: {'bytes': snapshot[name].st_size, 'sha256': digest(model/name)}
                      for name in component_names}
    out.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(min(vram_budget_mib*1048576/
        torch.cuda.get_device_properties(device).total_memory, .9), device)
    torch.cuda.reset_peak_memory_stats()
    target_device = torch.device('cuda', device)
    qwen_file = model/'text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'
    vae_file = model/'vae/minimax_h3_video_vae_fp16.safetensors'
    tokenizer = Tokenizer.from_file(str(model/'tokenizer/tokenizer.json'))
    resized, refs, keyframes = [], [], []
    for index, source in enumerate(sources):
        image = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
        if variant == 'ref2va':
            scale = min(1, math.sqrt(width*height/(len(sources)*image.width*image.height)),
                        2048/image.width, 2048/image.height)
            size = (max(32, round(image.width*scale/32)*32), max(32, round(image.height*scale/32)*32))
            image = image.resize(size, Image.Resampling.LANCZOS)
            refs.append({'kind': 'image', 'latent_h': size[1]//16, 'latent_w': size[0]//16})
        else:
            last = bool(last_frame and (not first_frame or index == 1))
            image = (ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS) if last
                     else image.resize((width, height), Image.Resampling.LANCZOS))
            keyframes.append({'resolved_frame_index': frames-1 if last else 0})
        image.save(out/f'input_{index}.png')
        resized.append(torch.from_numpy(np.asarray(image).copy()).float()[None]/255)
    config = {**QWEN3VL_VISION_COMMON, **QWEN3VL_VISION['qwen3vl_32b'], 'out_hidden_size': 5120}
    tower = Qwen3VLVisionModel(config, device='cpu', dtype=torch.bfloat16, ops=ops.disable_weight_init)
    with safe_open(qwen_file, framework='pt', device='cpu') as weights:
        state = {k[len('visual.'):]: weights.get_tensor(k) for k in weights.keys() if k.startswith('visual.')}
    tower.load_state_dict(state, strict=True, assign=True)
    del state
    # The pinned Qwen visual preprocessing consumes FP32 patches/weights.
    tower.to(target_device, dtype=torch.float32).eval()
    entries, info, stacks = [], [], []
    def text(value):
        entries.extend(tokenizer.encode(value, add_special_tokens=False).ids)
    with torch.inference_mode():
        for index, image in enumerate(resized):
            if variant == 'ref2va':
                text(f'<Picture {index+1}>: ')
            entries.append(151652)
            patches, grid = process_qwen2vl_images(image, patch_size=16, image_mean=[.5]*3,
                image_std=[.5]*3, max_pixels=max(4096, (1536//len(sources))*1024))
            merged, deepstack = tower(patches.to(target_device), grid.to(target_device))
            start = len(entries)
            entries.extend([-1]*len(merged))
            stacks.append([s.bfloat16().float().cpu() for s in deepstack])
            info.append({'type': 'image', 'index': start, 'size': len(merged), 'extra': grid.cpu()})
            # Save only the visual embeddings; free this encoder before the VAE.
            merged.bfloat16().float().cpu().numpy().astype('<f4').tofile(out/f'vision_{index}.f32')
            del patches, grid, merged, deepstack
            entries.append(151653)
        text(prompt)
    del tower
    torch.cuda.empty_cache()
    count = len(entries)
    if not 1 <= count <= 2048:
        raise ValueError('conditioned Qwen sequence exceeds 2048 tokens; reduce prompt or image count')
    inputs = np.zeros((count, 5120), np.float32)
    with safe_open(qwen_file, framework='pt', device='cpu') as weights:
        embedding = weights.get_slice('model.embed_tokens.weight')
        for index, token in enumerate(entries):
            if token >= 0:
                inputs[index] = embedding[token:token+1].float().numpy()[0]
    ds = [np.zeros_like(inputs) for _ in range(3)]
    for index, item in enumerate(info):
        a, b = item['index'], item['index']+item['size']
        inputs[a:b] = np.fromfile(out/f'vision_{index}.f32', '<f4').reshape(-1, 5120)
        for layer in range(3):
            ds[layer][a:b] = stacks[index][layer].numpy()
    positions = qwen2vl_mrope_position_ids(info, count, 'cpu')
    cosine, sine, _ = precompute_freqs_cis(128, positions.to(target_device), 5000000.,
        rope_dims=[24,20,20], interleaved_mrope=True, device=target_device)
    rotation = torch.stack((cosine.reshape(count, 128)[:, :64], sine.reshape(count, 64)), -1).float().cpu().numpy()
    def write(name, value):
        np.asarray(value, '<f4').tofile(out/(name+'.f32'))
    write('qwen_inputs', inputs)
    write('qwen_rotation', rotation)
    write('text_tags', token_tags_from_embeds_info(count, info).numpy())
    for layer in range(3):
        write(f'deepstack_{layer}', ds[layer])
    del ds, stacks, inputs, rotation, positions, cosine, sine
    torch.cuda.empty_cache()
    # No decoder weights are loaded into this preprocessing process.
    vae = MiniMaxH3VideoVAE(num_layers=0)
    with safe_open(vae_file, framework='pt', device='cpu') as weights:
        state = {k: weights.get_tensor(k) for k in weights.keys()
                 if k.startswith(('encoder.', 'quant_conv.')) or k in ('latents_mean','latents_std')}
    missing, unexpected = vae.load_state_dict(state, strict=False, assign=True)
    if unexpected or any(k.startswith(('encoder.', 'quant_conv.')) for k in missing):
        raise ValueError('VAE encoder checkpoint mismatch')
    del state
    vae.encoder.to(target_device).eval()
    vae.quant_conv.to(target_device).eval()
    conditions = []
    latent_shapes = []
    with torch.inference_mode():
        for index, image in enumerate(resized):
            pixels = (image.permute(0, 3, 1, 2).unsqueeze(2)*2-1).to(target_device, torch.float16)
            latent = vae.encode(pixels).float().cpu()
            latent_shapes.append(list(latent.shape))
            if latent.shape[2] != 1:
                raise ValueError('image encoder must produce exactly one latent frame')
            if variant == 'ref2va':
                refs[index]['latent'] = latent
            else:
                keyframes[index]['latent'] = latent
            rows = patchify_video(latent)
            noise = torch.randn(rows.shape, generator=torch.Generator('cpu').manual_seed(seed))
            conditions.append(.999*rows+.001*noise)
            write(f'latent_{index}', latent)
    del vae
    torch.cuda.empty_cache()
    latent_t = (frames-5)//17*5+2
    audio_t = round(frames/24*40)
    layout = PackedLayout(count, latent_t, height//16, width//16, audio_t, keyframes=keyframes, refs=refs)
    checkpoint = model/f'diffusion_models/minimax_h3_{variant}_pruned_int8_convrot.safetensors'
    with safe_open(checkpoint, framework='pt', device='cpu') as weights:
        freq = weights.get_tensor('rope.inv_freq').float()
    phases = layout.position_ids.float()[..., None]*freq
    write('dit_phases', phases.flatten(1).numpy())
    write('condition_patches', torch.cat(conditions).numpy())
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in out.iterdir() if p.is_file()}
    for name, stamp in snapshot.items():
        current = (model/name).stat()
        if (current.st_size, current.st_mtime_ns, current.st_ino) != (stamp.st_size, stamp.st_mtime_ns, stamp.st_ino):
            raise ValueError('encoder weights changed during preprocessing')
    receipt = {'schema': 'h3.image_conditioning.v1', 'variant': variant, 'prompt': prompt,
        'width': width, 'height': height, 'frames': frames, 'seed': seed,
        'text_rows': count, 'condition_rows': sum(len(r) for r in conditions),
        'upstream_revision': revision, 'encoder_backend': 'pytorch_rocm', 'rocm': torch.version.hip,
        'verified_components': components,
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/1048576,
        'sources': [{'path': str(p), 'sha256': digest(p)} for p in sources], 'files': files,
        'segments': layout.segments, 'visual_cond_timestep': .999,
        'latent_shapes': latent_shapes,
        'keyframe_indices': [k['resolved_frame_index'] for k in keyframes],
        'policy': 'reference scale uses target area per image, rounded to 32 pixels; Qwen vision tokens capped at 1536 total'}
    (out/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'out', 'prompt', 'variant'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--images', nargs='*', default=[])
    parser.add_argument('--first-frame')
    parser.add_argument('--last-frame')
    parser.add_argument('--comfy', default=str(DEFAULT_COMFY))
    parser.add_argument('--model-receipt')
    for key, default in (('width',480),('height',832),('frames',22),('seed',42),('device',0),('vram-budget-mib',12288)):
        parser.add_argument('--'+key, type=int, default=default)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
