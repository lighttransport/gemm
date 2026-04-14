#!/usr/bin/env python3
"""Compose a side-by-side comparison of reference vs HIP apple images."""
import argparse
from PIL import Image, ImageDraw, ImageFont

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref",  default="apple_ref.png")
    ap.add_argument("--hip",  default="../../rdna4/flux2/apple_hip.png")
    ap.add_argument("-o", "--out", default="apple_comparison.png")
    ap.add_argument("--label-h", type=int, default=40)
    args = ap.parse_args()

    ref = Image.open(args.ref).convert("RGB")
    hip = Image.open(args.hip).convert("RGB")

    # Match heights (resize the smaller to the larger)
    h = max(ref.height, hip.height)
    if ref.height != h:
        ref = ref.resize((int(ref.width * h / ref.height), h), Image.LANCZOS)
    if hip.height != h:
        hip = hip.resize((int(hip.width * h / hip.height), h), Image.LANCZOS)

    gap = 12
    label_h = args.label_h
    W = ref.width + hip.width + gap
    H = h + label_h

    out = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(out)

    # Labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    ref_label = "diffusers + pytorch-rocm"
    hip_label = "HIP FP8 LUT (RX 9070 XT)"

    def label_center(text, x0, x1, y):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x0 + (x1 - x0 - tw) // 2, y), text, fill=(30, 30, 30), font=font)

    label_center(ref_label, 0, ref.width, (label_h - 22) // 2)
    label_center(hip_label, ref.width + gap, ref.width + gap + hip.width, (label_h - 22) // 2)

    out.paste(ref, (0, label_h))
    out.paste(hip, (ref.width + gap, label_h))

    out.save(args.out)
    print(f"saved {args.out} ({W}×{H})")

if __name__ == "__main__":
    main()
