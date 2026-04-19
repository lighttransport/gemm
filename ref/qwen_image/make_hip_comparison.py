#!/usr/bin/env python3
"""Side-by-side HIP LUT vs HIP WMMA Qwen-Image apples (256 + 512).
For visual confirmation that the WMMA path produces the same result as the
default tiled LUT path.
"""
from PIL import Image, ImageDraw, ImageFont


def label(text, w, font, h=28):
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    d.text(((w - tw) // 2, 5), text, fill=(30, 30, 30), font=font)
    return img


def main():
    base = "../../rdna4/qimg/"
    lut256 = Image.open(base + "apple_qimg_lut_256.png").convert("RGB")
    wmma256 = Image.open(base + "apple_qimg_wmma_256.png").convert("RGB")
    lut512 = Image.open(base + "apple_qimg_lut_512.png").convert("RGB")
    wmma512 = Image.open(base + "apple_qimg_wmma_512.png").convert("RGB")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    gap = 12
    label_h = 28

    row1_w = lut256.width + gap + wmma256.width
    row1 = Image.new("RGB", (row1_w, label_h + lut256.height), (245, 245, 245))
    row1.paste(label("HIP tiled LUT 256 (1.80 s/step)", lut256.width, font), (0, 0))
    row1.paste(label("HIP BF16 WMMA 256 (1.47 s/step)", wmma256.width, font), (lut256.width + gap, 0))
    row1.paste(lut256, (0, label_h))
    row1.paste(wmma256, (lut256.width + gap, label_h))

    row2_w = lut512.width + gap + wmma512.width
    row2 = Image.new("RGB", (row2_w, label_h + lut512.height), (245, 245, 245))
    row2.paste(label("HIP tiled LUT 512 (4.56 s/step)", lut512.width, font), (0, 0))
    row2.paste(label("HIP BF16 WMMA 512 (4.09 s/step)", wmma512.width, font), (lut512.width + gap, 0))
    row2.paste(lut512, (0, label_h))
    row2.paste(wmma512, (lut512.width + gap, label_h))

    W = max(row1_w, row2_w)
    out = Image.new("RGB", (W, row1.height + gap + row2.height), (245, 245, 245))
    out.paste(row1, ((W - row1_w) // 2, 0))
    out.paste(row2, ((W - row2_w) // 2, row1.height + gap))
    out.save("apple_hip_lut_vs_wmma.png")
    print("saved apple_hip_lut_vs_wmma.png", out.size)


if __name__ == "__main__":
    main()
