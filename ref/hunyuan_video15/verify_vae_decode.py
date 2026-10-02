"""Full portrait-resolution VAE decode parity from an identical saved latent.

Uses the pinned official spatial tiling recipe at the native 128-pixel tile
size, retaining every latent frame. Conditioning and denoising are excluded.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ref.hunyuan_video15.compare import compare, compare_frames
from ref.hunyuan_video15.convert_native import convert
PIN = "60783e704160023913bee78f0b47036d393d4dfa"

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load_native_latent(directory, frames):
    directory = Path(directory)
    legacy = directory / 'latent_final.shape.json'
    if legacy.is_file():
        shape = json.loads(legacy.read_text())
    else:
        meta = json.loads((directory / 'latent_final.json').read_text())
        if meta.get('dtype') != 'float32' or meta.get('layout') != 'NCTHW':
            raise ValueError('invalid canonical latent dtype/layout')
        shape = meta.get('shape')
    if shape not in ([1,32,21,53,30], [1,32,31,53,30]):
        raise ValueError('requires a canonical 480x848/81- or 121-frame latent')
    if frames not in (1, 5, 81, 121) or frames > (shape[2]-1)*4+1:
        raise ValueError('requested frames exceed the saved latent')
    source = directory / 'latent_final.f32'
    if source.stat().st_size != int(np.prod(shape))*4:
        raise ValueError('wrong latent byte count')
    latent = np.fromfile(source, dtype='<f4').reshape(shape)
    if not np.isfinite(latent).all():
        raise ValueError('non-finite latent')
    return latent

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--latent', type=Path, required=True, help='native latent_final dump directory')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frames', type=int, choices=(1,5,81,121), default=81)
    ap.add_argument('--reference-dtype', choices=('float16','float32'), default='float16')
    ap.add_argument('--actual', type=Path, help='use existing native decoded pixels instead of running the probe')
    ap.add_argument('--reference-latent-npy',type=Path,help='independently denoised reference latent for pipeline checks')
    args = ap.parse_args()
    if args.frames == 121 and not args.actual:
        raise ValueError('121-frame comparison requires --actual; the legacy decode probe supports at most 81')
    import torch
    from safetensors.torch import load_file
    upstream = args.upstream.resolve()
    if subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'], text=True).strip()!=PIN:
        raise ValueError('unexpected upstream revision')
    latent = load_native_latent(args.latent, args.frames)
    shape = latent.shape
    source = args.latent/'latent_final.f32'
    if args.reference_latent_npy:
        latent=np.load(args.reference_latent_npy,allow_pickle=False)
        if latent.shape!=shape:
            raise ValueError('independent reference latent has the wrong shape')
    if not np.isfinite(latent).all():
        raise ValueError('non-finite latent')
    latent = latent[:,:,:((args.frames-1)//4+1)].copy()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference, actual = out/'reference', out/'native'
    reference.mkdir(); actual.mkdir()
    latent.astype('<f4').tofile(out/'latent.f32')
    probe = ROOT/'cuda/hunyuan_video15/test_cuda_hunyuan_video15_vae_decode'
    library = ROOT/'tmp/hunyuan-video15-native/build/bin/libstable-diffusion.so'
    native_build = None if args.actual else {str(p.relative_to(ROOT)):sha256(p) for p in (probe,library)}
    sys.path.insert(0, str(upstream))
    spec = importlib.util.spec_from_file_location('hv15_reference_vae', upstream/'hyvideo/models/autoencoders/hunyuanvideo_15_vae.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = {k:v for k,v in json.loads(args.config.read_text()).items() if not k.startswith('_')}
    model = module.AutoencoderKLConv3D(**config)
    weights = args.model.resolve()/'split_files/vae/hunyuanvideo15_vae_fp16.safetensors'
    model.load_state_dict(load_file(str(weights)), strict=True)
    dtype = getattr(torch,args.reference_dtype)
    model.eval().requires_grad_(False).to(device='cuda',dtype=dtype)
    model.set_tile_sample_min_size(128,0.25)
    model.enable_spatial_tiling()
    completed = 0
    def tile_complete(*unused):
        nonlocal completed
        completed += 1
        print(f'REFERENCE_VAE_TILE {completed}',flush=True)
    model.decoder.register_forward_hook(tile_complete)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda',dtype=dtype,enabled=dtype==torch.float16):
        # Official pipeline unscales its FP32 scheduler output before decode.
        value = torch.from_numpy(latent).to(device='cuda')/model.scaling_factor
        if model.shift_factor is not None:
            value = value+model.shift_factor
        decoded = model.decode(value).sample
        pixels = (decoded/2+0.5).clamp(0,1).float().cpu().numpy()
    torch.cuda.synchronize()
    reference_seconds = time.monotonic()-started
    if pixels.shape!=(1,3,args.frames,848,480) or not np.isfinite(pixels).all():
        raise ValueError('invalid reference pixels')
    np.save(reference/'vae_decoded.npy',pixels)
    allocated = torch.cuda.max_memory_allocated()/1048576
    reserved = torch.cuda.max_memory_reserved()/1048576
    del model, value, decoded
    torch.cuda.empty_cache()
    started = time.monotonic()
    if args.actual:
        actual=args.actual.resolve()
        native_seconds=None
    else:
        with (out/'native.log').open('w') as log:
            subprocess.run([str(probe),str(weights),str(out/'latent.f32'),str(actual),str(args.frames)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        native_seconds = time.monotonic()-started
    convert(actual)
    results = compare(reference,actual,['vae_decoded'])
    frame_results = compare_frames(reference,actual)
    report = {'scope':'vae_decode_of_independent_pipeline_latents' if args.reference_latent_npy else 'full_spatial_vae_decode_with_identical_saved_latent',
              'upstream_revision':PIN,'reference_dtype':args.reference_dtype,
              'size':[480,848],'frames':args.frames,'tile_pixels':128,'overlap':0.25,
              'temporal_tiling':False,'reference_tiles':completed,
              'source_latent_sha256':sha256(source),'used_latent_sha256':sha256(out/'latent.f32'),
              'reference_latent_source_sha256':sha256(args.reference_latent_npy) if args.reference_latent_npy else None,
              'config_sha256':sha256(args.config),'native_build_sha256':native_build,
              'reference_seconds':reference_seconds,'native_seconds':native_seconds,
              'peak_torch_allocated_mib':allocated,'peak_torch_reserved_mib':reserved,
              'peak_native_host_rss_kib':None if args.actual else resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              'results':results,'frame_errors':frame_results,
              'frames_pass':all(v['pass'] for v in frame_results)}
    (out/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if all(v['pass'] for v in results.values()) and report['frames_pass'] else 1

if __name__=='__main__':
    raise SystemExit(main())
