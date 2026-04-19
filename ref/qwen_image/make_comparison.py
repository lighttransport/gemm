#!/usr/bin/env python3
"""Compose a 2x2 benchmark image for Qwen-Image: diffusers vs HIP at 256/512.

Reads:
  apple_ref_256.png            (diffusers reference, 256x256)
  apple_ref_512.png            (diffusers reference, 512x512)
  ../../rdna4/qimg/apple_qimg_wmma_256.png
  ../../rdna4/qimg/apple_qimg_wmma_512.png
"""
import argparse
from PIL import Image, ImageDraw, ImageFont


def label(text, w, font, h=28):
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    d.text(((w - tw) // 2, 5), text, fill=(30, 30, 30), font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref256", default="apple_ref_256.png")
    ap.add_argument("--hip256", default="../../rdna4/qimg/apple_qimg_wmma_256.png")
    ap.add_argument("--ref512", default="apple_ref_512.png")
    ap.add_argument("--hip512", default="../../rdna4/qimg/apple_qimg_wmma_512.png")
    ap.add_argument("-o", "--out", default="apple_benchmark.png")
    args = ap.parse_args()

    ref256 = Image.open(args.ref256).convert("RGB")
    hip256 = Image.open(args.hip256).convert("RGB")
    ref512 = Image.open(args.ref512).convert("RGB")
    hip512 = Image.open(args.hip512).convert("RGB")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    gap = 12
    label_h = 28

    row1_w = ref256.width + gap + hip256.width
    row1_h = label_h + ref256.height
    row1 = Image.new("RGB", (row1_w, row1_h), (245, 245, 245))
    row1.paste(label("diffusers 256", ref256.width, font), (0, 0))
    row1.paste(label("HIP WMMA 256", hip256.width, font), (ref256.width + gap, 0))
    row1.paste(ref256, (0, label_h))
    row1.paste(hip256, (ref256.width + gap, label_h))

    row2_w = ref512.width + gap + hip512.width
    row2_h = label_h + ref512.height
    row2 = Image.new("RGB", (row2_w, row2_h), (245, 245, 245))
    row2.paste(label("diffusers 512", ref512.width, font), (0, 0))
    row2.paste(label("HIP WMMA 512", hip512.width, font), (ref512.width + gap, 0))
    row2.paste(ref512, (0, label_h))
    row2.paste(hip512, (ref512.width + gap, label_h))

    W = max(row1_w, row2_w)
    out = Image.new("RGB", (W, row1_h + gap + row2_h), (245, 245, 245))
    out.paste(row1, ((W - row1_w) // 2, 0))
    out.paste(row2, ((W - row2_w) // 2, row1_h + gap))
    out.save(args.out)
    print(f"saved {args.out} ({out.width}x{out.height})")


if __name__ == "__main__":
    main()
