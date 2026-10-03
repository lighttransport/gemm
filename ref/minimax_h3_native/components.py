"""Independent H3 component checks with explicitly shared component inputs.

These isolate decoder or DiT block error; they cannot certify pipeline parity.
Run the corresponding native component_probe first. Expected values use only
PyTorch operations from reference.py and checkpoint weights.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from reference import Reference, UPSTREAM, digest
from verify import capture, comparison


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("vae", "dit-block", "dit-step"))
    p.add_argument("--model", default="/mnt/disk01/models/h3/weights")
    for name in ("input", "native", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--frames", type=int, default=39)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()
    source, native = Path(args.input), Path(args.native)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    report = {}
    if args.mode == "vae":
        video = torch.from_numpy(capture(source, "latent_video")).to(device)
        if (args.frames < 5 or (args.frames - 5) % 17 or
                len(video) != (args.frames - 5) // 17 * 5 + 2):
            raise ValueError("invalid VAE temporal geometry")
        path = Path(args.model) / "vae/minimax_h3_video_vae_fp16.safetensors"
        with safe_open(path, framework="pt", device="cpu") as weights:
            ref = Reference(weights, device)
            for index, frame in enumerate(ref.decode(video, args.frames)):
                name = f"frame_{index:03d}"
                actual = np.fromfile(native / (name + ".f32"), dtype="<f4").reshape(frame.shape)
                report[name] = comparison(actual, frame)
                print(name, report[name], flush=True)
    else:
        video = torch.from_numpy(capture(source, "noise_video")).to(device)
        audio = torch.from_numpy(capture(source, "noise_audio")).reshape(-1, 32).to(device)
        text = torch.from_numpy(capture(source, "refined_text")).reshape(-1, 5376).to(device, torch.bfloat16)
        path = Path(args.model) / "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        with safe_open(path, framework="pt", device="cpu") as weights:
            ref = Reference(weights, device)
            t, h, w, _ = video.shape
            patches = video.reshape(t, h // 2, 2, w // 2, 2, 24).permute(0, 1, 3, 5, 2, 4).reshape(-1, 96)
            x = torch.cat((text, ref.linear(audio, "audio_patch_proj", True).bfloat16(),
                           ref.linear(patches, "video_patch_proj", True).bfloat16()))
            segments = ((0, len(text), 1), (len(text), len(text) + len(audio), 5),
                        (len(text) + len(audio), len(x), 0))
            rotation = ref.dit_rope(len(text), len(audio) // 2, video.shape[:3])
            emb = ref.time_embed(0, 0)
            ref.clear()
            if args.mode == "dit-step" and len(x) > 256:
                raise ValueError("DiT tracing requires at most 256 tokens")
            for i in range(50 if args.mode == "dit-step" else 1):
                prefix = f"blocks.{i}"
                mod = ref.linear(emb, prefix + ".adaln_proj.linear", True).reshape(6, 6, 5376)
                z = ref.modulate(ref.norm(x, prefix + ".norm1"), mod, segments, 0)
                ref.gated(x, ref.attention(z, prefix + ".attn", rotation), mod, segments, 2)
                if args.mode == "dit-step":
                    name = f"dit_step_block_{i:02d}_after_attn"
                    report[name] = comparison(capture(native, name), x.float().cpu().numpy())
                    print(name, report[name], flush=True)
                z = ref.modulate(ref.norm(x, prefix + ".norm2"), mod, segments, 3)
                ref.gated(x, ref.ffn(z, prefix + ".mlp"), mod, segments, 5)
                name = f"dit_step_block_{i:02d}_after_mlp" if args.mode == "dit-step" else "dit_block_0"
                report[name] = comparison(capture(native, name), x.float().cpu().numpy())
                print(name, report[name], flush=True)
                ref.clear()
    result = {"scope": "component_shared_inputs", "mode": args.mode,
              "upstream_revision": UPSTREAM, "reference_source_sha256": digest(Path(__file__).with_name("reference.py")),
              "comparisons": report, "pass": all(item["pass"] for item in report.values())}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    if not result["pass"]:
        raise RuntimeError("H3 component parity failed")


if __name__ == "__main__":
    main()
