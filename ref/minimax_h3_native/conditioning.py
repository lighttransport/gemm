"""Independent conditioned H3 DiT/VAE comparison using encoder bundle inputs.

The encoders are upstream PyTorch preprocessing. Qwen must pass its separate
independent reference first. No native denoised state is used as a reference
input; only initial noise is shared. This checker certifies diagnostics only.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from safetensors import safe_open
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ref.minimax_h3_native.reference import Reference, digest
from ref.minimax_h3_native.verify import capture, comparison
from ref.minimax_h3_native.captures import read_f32


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('generation', 'native', 'qwen-reference', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/mnt/disk01/models/h3/weights'))
    args = parser.parse_args()
    generation = json.loads((args.generation/'manifest.json').read_text())
    bundle = args.generation/'conditioning'
    receipt = json.loads((bundle/'manifest.json').read_text())
    qwen_provenance = json.loads((args.qwen_reference/'provenance.json').read_text())
    qwen_report = json.loads((args.qwen_reference/'report.json').read_text())
    if not qwen_report['pass'] or qwen_provenance['conditioning_manifest_sha256'] != digest(bundle/'manifest.json'):
        raise ValueError('conditioned Qwen reference is missing, failed or mismatched')
    if qwen_provenance['qwen_hidden_sha256'] != digest(args.qwen_reference/'qwen_hidden.npy'):
        raise ValueError('Qwen reference hash mismatch')
    for name, item in generation['verified_components'].items():
        if digest(args.model/name) != item['sha256']:
            raise ValueError('generation weights changed')
    for name, item in receipt['files'].items():
        if digest(bundle/name) != item['sha256']:
            raise ValueError('encoder bundle changed')
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda', 0)
    report = {}
    def check(name, value):
        if isinstance(value, torch.Tensor):
            value = value.float().cpu().numpy()
        actual = (read_f32(args.native, name, value.size).reshape(value.shape)
                  if name.startswith('frame_') else capture(args.native, name))
        report[name] = comparison(actual, value)
        print(name, report[name], flush=True)
    video = torch.from_numpy(capture(args.native, 'noise_video')).to(device)
    audio = torch.from_numpy(capture(args.native, 'noise_audio')).reshape(-1,32).to(device)
    text = torch.from_numpy(np.load(args.qwen_reference/'qwen_hidden.npy').reshape(-1,5120)).to(device,torch.bfloat16)
    # Independently patchify each normalized upstream VAE latent and augment it.
    sys.path.insert(0, str(ROOT/'tmp/video-rocm/pytorch-bench-comfy'))
    from comfy.ldm.minimax.model import PackedLayout, patchify_video
    from PIL import Image
    conditions, refs, keyframes = [], [], []
    for index in range(len(receipt['sources'])):
        size = Image.open(bundle/f'input_{index}.png').size
        shape = (1,24,1,size[1]//16,size[0]//16)
        z = torch.from_numpy(np.fromfile(bundle/f'latent_{index}.f32','<f4').reshape(shape))
        rows = patchify_video(z)
        noise = torch.randn(rows.shape, generator=torch.Generator('cpu').manual_seed(receipt['seed']))
        conditions.append(.999*rows+(1-.999)*noise)
        if receipt['variant'] == 'ref2va':
            refs.append({'kind':'image','latent_h':shape[3],'latent_w':shape[4],'latent':z})
        else:
            keyframes.append({'latent':z,'resolved_frame_index':receipt['keyframe_indices'][index]})
    patches = torch.cat(conditions).to(device)
    check('condition_patches', patches)
    layout = PackedLayout(len(text), *video.shape[:3], len(audio)//2, keyframes=keyframes, refs=refs)
    tags = np.fromfile(bundle/'text_tags.f32','<f4').astype(int)
    checkpoint = args.model/f"diffusion_models/minimax_h3_{receipt['variant']}_pruned_int8_convrot.safetensors"
    with safe_open(checkpoint, framework='pt', device='cpu') as weights:
        ref = Reference(weights, device)
        text = ref.refine(text)
        check('refined_text', text)
        angles = (layout.position_ids.float().to(device)[...,None]*ref.weight('rope.inv_freq')).flatten(1)
        rotation = angles.cos().bfloat16(), angles.sin().bfloat16()
        check('condition_rotation', torch.stack(rotation,-1))
        points = generation['sigma_grid_points']
        grid = 1-torch.arange(points,dtype=torch.float32)/(points-1)
        sv, sa = 12*grid/(1+11*grid), 3*grid/(1+2*grid)
        for step in range(points-1):
            vv, av = ref.denoise(text,video,audio,rotation,float(1-sv[step]),float(1-sa[step]),step+1,
                                 conditioning={'patches':patches,'text_tags':tags})
            video += (sv[step]-sv[step+1]).to(device)*vv
            audio += (sa[step]-sa[step+1]).to(device)*av
            check(f'latent_video_{step:03d}', video)
            check(f'latent_audio_{step:03d}', audio)
        ref.clear()
        del ref, text, rotation, vv, av, audio, patches
    torch.cuda.empty_cache()
    with safe_open(args.model/'vae/minimax_h3_video_vae_fp16.safetensors',framework='pt',device='cpu') as weights:
        ref = Reference(weights,device)
        for index, image in enumerate(ref.decode(video,generation['frames'])):
            check(f'frame_{index:03d}', image)
    args.out.write_text(json.dumps({'scope':'conditioned_diagnostic', 'comparisons':report,
        'encoders':'shared upstream PyTorch ROCm preprocessing, Qwen independently checked',
        'generation_sha256':digest(args.generation/'manifest.json'), 'reference_sha256':digest(__file__),
        'pass':all(r['pass'] for r in report.values())},indent=2)+'\n')
    if not all(r['pass'] for r in report.values()):
        raise RuntimeError('conditioned reference mismatch; see '+str(args.out))


if __name__ == '__main__':
    main()
