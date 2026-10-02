#!/usr/bin/env python3
"""Offline Torch oracle for the pinned RMBG-2.0 Swin-L backbone only.

Execute the unmodified Swin definitions from local birefnet.py, avoiding the
unrelated transformers/timm/torchvision imports and decoder initialization.
DropPath is identity in eval; all trained weights/buffers load strictly.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

SOURCE_SHA256 = "e499d75224b8819e985e68fb78b7a8e8c99316840474e74e16b5529f03ca2860"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--half-scale", action="store_true",
                    help="Use BiRefNet's bilinear half-scale input from a 2x larger normalized image")
    ap.add_argument("--trace", action="store_true", help="Dump raw stage-2 block outputs for diagnostics")
    ap.add_argument("--reference-dtype", choices=("float32", "float64"), default="float32",
                    help="Float64 is a diagnostic for accumulated FP32 roundoff")
    args = ap.parse_args()
    if not (1 <= args.height <= 1024 and 1 <= args.width <= 1024 and 1 <= args.threads <= 128):
        ap.error("dimensions must be 1..1024 and threads 1..128")
    if args.half_scale and max(args.height, args.width) > 512:
        ap.error("half-scale source dimensions must not exceed 1024")
    source = args.model / "birefnet.py"
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("Unsupported birefnet.py source: inspect before adding another revision")
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file

    torch.set_num_threads(args.threads)
    text = source.read_text()
    section = text.split("### models/backbones/swin_v1.py", 1)[1].split("### models/modules/deform_conv.py", 1)[0]
    section = section[section.index("class Mlp("):]
    ns = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, np=np,
              config=SimpleNamespace(SDPA_enabled=False),
              to_2tuple=lambda x: (x, x), trunc_normal_=torch.nn.init.trunc_normal_,
              DropPath=lambda probability: torch.nn.Identity())
    exec(compile(section, str(source) + ":swin_v1", "exec"), ns)
    with torch.device("meta"):
        model = ns["swin_v1_l"]()
    # This upstream backbone overrides train() without returning self.
    model.eval()
    weights = args.model / "model.safetensors"
    state = {k[3:]: v for k, v in load_file(str(weights)).items() if k.startswith("bb.")}
    model.load_state_dict(state, strict=True, assign=True)
    source_height = args.height * (2 if args.half_scale else 1)
    source_width = args.width * (2 if args.half_scale else 1)
    if args.image:
        # torchvision Resize(PIL) default is bilinear, then uint8 ToTensor /255.
        image = Image.open(args.image).convert("RGB").resize(
            (source_width, source_height), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(image).transpose(2, 0, 1).copy()).float() / 255
        x = (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        x = x.unsqueeze(0)
    else:
        torch.manual_seed(1234)
        x = torch.randn(1, 3, source_height, source_width)
    if args.half_scale:
        x = torch.nn.functional.interpolate(x, size=(args.height, args.width), mode="bilinear", align_corners=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.trace:
        for i, block in enumerate(model.layers[2].blocks):
            def dump(module, inputs, output, index=i):
                output.numpy().astype("<f4").tofile(args.output / f"block_{index}.f32")
            block.register_forward_hook(dump)
    x.numpy().astype("<f4").tofile(args.output / "input.f32")
    if args.reference_dtype == "float64":
        model.double()
        x = x.double()
    with torch.inference_mode():
        features = model(x)
    for i, feature in enumerate(features):
        feature.numpy().astype("<f4").tofile(args.output / f"reference_{i}.f32")
    manifest = {"kind": "rmbg2_swin_l_backbone", "height": args.height, "width": args.width,
                "shapes": [list(f.shape[1:]) for f in features], "source_sha256": SOURCE_SHA256,
                "weights": str(weights.resolve()), "weights_sha256": sha256(weights),
                "image": str(args.image.resolve()) if args.image else None, "half_scale": args.half_scale,
                "torch": torch.__version__, "reference_dtype": args.reference_dtype,
                "input_sha256": sha256(args.output / "input.f32")}
    (args.output / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
