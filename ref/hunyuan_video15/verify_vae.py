"""Bounded one/five-frame VAE graph comparison using the pinned official source."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ref.hunyuan_video15.compare import compare
from ref.hunyuan_video15.convert_native import convert
from cuda.hunyuan_video15.native_generate import prepare_portrait
PIN = "60783e704160023913bee78f0b47036d393d4dfa"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference-dtype", choices=("float32", "float16"), default="float32")
    ap.add_argument("--frames", type=int, choices=(1,5), default=1)
    ap.add_argument("--portrait", action="store_true", help="480x848 portrait with official 128-pixel spatial tiling")
    ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--actual", type=Path, help="compare existing native component dumps")
    args = ap.parse_args()
    upstream = args.upstream.resolve()
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != PIN:
        raise ValueError("unexpected upstream revision")
    sys.path.insert(0, str(upstream))
    spec = importlib.util.spec_from_file_location("hv15_reference_vae", upstream / "hyvideo/models/autoencoders/hunyuanvideo_15_vae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference, actual = out / "reference", out / "native"
    reference.mkdir(); actual.mkdir()
    image_path=prepare_portrait(args.image,out,480,848)[0] if args.portrait else args.image
    size=(480,848) if args.portrait else (128,128)
    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")
        rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, (96,96,96,255)), rgba).convert("RGB")
        pixels = np.asarray(rgb.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255
    # Small deterministic horizontal motion exercises causal temporal layers.
    video = np.stack([np.roll(pixels, i, axis=1) for i in range(args.frames)], axis=0)
    planar = video.transpose(3,0,1,2)
    planar.astype("<f4").tofile(out / "pixels.f32")
    dtype = getattr(torch, args.reference_dtype)
    config = json.loads(args.config.read_text())
    config = {k:v for k,v in config.items() if not k.startswith("_")}
    model = module.AutoencoderKLConv3D(**config)
    weights = args.model.resolve() / "split_files/vae/hunyuanvideo15_vae_fp16.safetensors"
    model.load_state_dict(load_file(str(weights)), strict=True)
    model = model.eval().to(device="cuda", dtype=dtype)
    if args.portrait:
        model.set_tile_sample_min_size(128,0.25)
        model.enable_spatial_tiling()
    x = torch.from_numpy(planar[None]).to(device="cuda",dtype=torch.float32 if args.portrait else dtype)
    with torch.inference_mode(), torch.autocast('cuda',dtype=dtype,enabled=args.portrait and dtype==torch.float16):
        encoded = model.encode(x * 2 - 1).latent_dist.mode()
        decoded = None if args.encode_only else model.decode(encoded).sample
    scaled=encoded*model.scaling_factor if args.portrait else encoded.float()*model.scaling_factor
    np.save(reference / "vae_encoded.npy", scaled.float().cpu().numpy())
    if decoded is not None:
        value=(decoded/2+0.5).clamp(0,1).float() if args.portrait else ((decoded.float()+1)/2).clamp(0,1)
        np.save(reference / "vae_decoded.npy", value.cpu().numpy())
        del value
    del scaled
    del model, encoded, decoded, x
    torch.cuda.empty_cache()
    if args.actual:
        actual=args.actual.resolve()
    else:
        subprocess.run([str(ROOT / "cuda/hunyuan_video15/test_cuda_hunyuan_video15_vae"),
                        str(weights), str(out / "pixels.f32"), str(actual), str(args.frames)]+
                       (['--portrait'] if args.portrait else []),check=True)
    convert(actual)
    result = compare(reference, actual, ["vae_encoded"] if args.encode_only else ["vae_encoded", "vae_decoded"])
    report = {"reference_dtype": args.reference_dtype, "size": list(size), "frames":args.frames,
              "upstream_revision":PIN, "tiling":args.portrait,"tile_pixels":128 if args.portrait else None,
              "image_sha256":hashlib.sha256(image_path.read_bytes()).hexdigest(),"results":result}
    (out / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if all(v["pass"] for v in result.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
