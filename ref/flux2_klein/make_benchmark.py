#!/usr/bin/env python3
"""Build a 2×2 comparison image: (ref 256, hip 256) / (ref 512, hip 512)
and write the timing table to stdout.
"""
import argparse
from PIL import Image, ImageDraw, ImageFont

def label(text, w, font):
    img = Image.new("RGB", (w, 28), (245, 245, 245))
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    d.text(((w - tw) // 2, 3), text, fill=(30, 30, 30), font=font)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref256", default="apple_ref_256.png")
    ap.add_argument("--hip256", default="../../rdna4/flux2/apple_256_hip.png")
    ap.add_argument("--ref512", default="apple_ref_512.png")
    ap.add_argument("--hip512", default="../../rdna4/flux2/apple_512_unpinned.png")
    ap.add_argument("-o", "--out", default="apple_benchmark.png")
    args = ap.parse_args()

    ref256 = Image.open(args.ref256).convert("RGB")
    hip256 = Image.open(args.hip256).convert("RGB")
    ref512 = Image.open(args.ref512).convert("RGB")
    hip512 = Image.open(args.hip512).convert("RGB")

    # Row 1: 256 pair. Row 2: 512 pair.
    gap = 12
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    row1_w = ref256.width + gap + hip256.width
    row2_w = ref512.width + gap + hip512.width
    W = max(row1_w, row2_w)

    label_h = 28
    r1 = Image.new("RGB", (row1_w, label_h + ref256.height), (245, 245, 245))
    r1.paste(label("diffusers 256", ref256.width, font), (0, 0))
    r1.paste(label("HIP 256", hip256.width, font), (ref256.width + gap, 0))
    r1.paste(ref256, (0, label_h))
    r1.paste(hip256, (ref256.width + gap, label_h))

    r2 = Image.new("RGB", (row2_w, label_h + ref512.height), (245, 245, 245))
    r2.paste(label("diffusers 512", ref512.width, font), (0, 0))
    r2.paste(label("HIP 512", hip512.width, font), (ref512.width + gap, 0))
    r2.paste(ref512, (0, label_h))
    r2.paste(hip512, (ref512.width + gap, label_h))

    out = Image.new("RGB", (W, r1.height + gap + r2.height), (245, 245, 245))
    out.paste(r1, ((W - row1_w) // 2, 0))
    out.paste(r2, ((W - row2_w) // 2, r1.height + gap))
    out.save(args.out)
    print(f"saved {args.out} ({out.width}x{out.height})")

if __name__ == "__main__":
    main()
