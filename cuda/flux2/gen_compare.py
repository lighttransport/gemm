#!/usr/bin/env python3
"""Generate apple + 5 gallery images with our CUDA runner AND ComfyUI,
then compose side-by-side comparison PNGs (ours | diff heatmap | comfy).

Each item is processed in isolation: our binary runs --generate (which writes
cuda_flux2_output.ppm and cuda_flux2_text_hidden.npy for that prompt), then
ComfyUI re-runs the sampling using the cached text hidden states and the same
latent seed so the two pipelines see identical inputs. ComfyUI's diffusers VAE
decode is used for the "ground truth" path.

Outputs:
    out_<idx>_ours.png    — our CUDA output
    out_<idx>_comfy.png   — ComfyUI/diffusers output
    compare_<idx>.png     — [ours | diff heatmap | comfy] composite
    compare_all.png       — 6-row stacked composite of everything

Usage:
    python gen_compare.py
"""
import sys
import os
import math
import time
import subprocess

# Gallery: (index, prompt, seed)
# Index 0 is the apple (the flagship smoke test we've used throughout the session)
GALLERY = [
    (0, "a red apple on a white table", 42),
    (1, "a golden retriever puppy sitting in a field of sunflowers", 101),
    (2, "a steaming cup of coffee on a rainy window sill", 202),
    (3, "a snow-capped mountain reflected in a calm lake at sunset", 303),
    (4, "a blue butterfly resting on a pink rose", 404),
    (5, "an astronaut floating above Earth with stars in the background", 505),
]

HEIGHT = 512
WIDTH = 512
N_STEPS = 4
COMFYUI_DIR = "/mnt/disk01/ComfyUI"
MODEL_DIR = "/mnt/disk01/models/klein2-4b"
TOK_PATH = "/mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf"
CUDA_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.12/site-packages"))
import numpy as np  # noqa: E402


def load_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    ls = data.split(b"\n", 3)
    w, h = map(int, ls[1].split())
    return np.frombuffer(ls[3], dtype=np.uint8).reshape(h, w, 3)


def our_prng_noise(seed, n):
    """Match the deterministic Box-Muller PRNG used by our binary (cuda_flux2)."""
    state = seed
    vals = np.zeros(n, dtype=np.float32)
    i = 0
    while i < n:
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u1 = (state >> 11) / (1 << 53)
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u2 = (state >> 11) / (1 << 53)
        if u1 < 1e-10:
            u1 = 1e-10
        r = math.sqrt(-2.0 * math.log(u1))
        vals[i] = r * math.cos(2.0 * math.pi * u2)
        i += 1
        if i < n:
            vals[i] = r * math.sin(2.0 * math.pi * u2)
            i += 1
    return vals


def run_ours(idx, prompt, seed):
    """Run our CUDA pipeline for this item. Produces cuda_flux2_output.ppm and
    cuda_flux2_text_hidden.npy in CUDA_DIR. Returns the output image as a uint8 array."""
    cmd = (
        f"cd {CUDA_DIR} && FLUX2_FP8_GEMM=1 FLUX2_FP8_ATTN=1 ./test_cuda_flux2 "
        f"--generate --gpu-enc --enc {MODEL_DIR}/text_encoder --tok {TOK_PATH} "
        f"--height {HEIGHT} --width {WIDTH} --steps {N_STEPS} --seed {seed} "
        f'--prompt "{prompt}"'
    )
    print(f"  [ours] running: {prompt[:60]}... (seed={seed})")
    t0 = time.time()
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
    print(f"  [ours] done in {time.time() - t0:.1f}s")
    img = load_ppm(os.path.join(CUDA_DIR, "cuda_flux2_output.ppm"))
    return img


def run_comfy(model_patcher, vae, device, prompt, seed, hidden_path):
    """Run ComfyUI sampling using the cached text hidden states, then diffusers VAE decode."""
    import comfy.samplers
    import torch

    lat_h, lat_w = HEIGHT // 8, WIDTH // 8
    ps, lc = 2, 32
    ph, pw = lat_h // ps, lat_w // ps
    n_lat = lc * lat_h * lat_w

    noise_np = our_prng_noise(seed, n_lat).reshape(lc, lat_h, lat_w)
    patchified = np.zeros((lc * ps * ps, ph, pw), dtype=np.float32)
    for c in range(lc):
        for pr in range(ps):
            for pc in range(ps):
                patchified[c * ps * ps + pr * ps + pc] = noise_np[c, pr::ps, pc::ps]
    noise = torch.tensor(patchified.reshape(1, lc * ps * ps, ph, pw), dtype=torch.float32)

    text_hidden = np.load(hidden_path)
    text_cond = torch.tensor(text_hidden, dtype=torch.float32).unsqueeze(0)
    pooled = torch.zeros(1, 3072, dtype=torch.float32)
    positive = [[text_cond, {"pooled_output": pooled}]]
    negative = [[torch.zeros_like(text_cond), {"pooled_output": pooled}]]

    sampler = comfy.samplers.KSampler(
        model=model_patcher, steps=N_STEPS, device=device,
        sampler="euler", scheduler="simple", denoise=1.0,
    )
    latent_image = torch.zeros_like(noise)
    result = sampler.sample(
        noise=noise, positive=positive, negative=negative,
        cfg=1.0, latent_image=latent_image,
        start_step=0, last_step=N_STEPS, force_full_denoise=True,
    )

    patchified_out = result[0].float().cpu().numpy()
    latent_out = np.zeros((lc, lat_h, lat_w), dtype=np.float32)
    for c in range(lc):
        for pr in range(ps):
            for pc in range(ps):
                latent_out[c, pr::ps, pc::ps] = patchified_out[c * ps * ps + pr * ps + pc]

    with torch.no_grad():
        lat_t = torch.tensor(latent_out, dtype=torch.float32, device=device).unsqueeze(0)
        rgb = vae.decode(lat_t).sample[0].cpu().numpy()
    img = ((rgb.transpose(1, 2, 0) + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
    return img


def load_comfy_model():
    """Load ComfyUI DiT + diffusers VAE. Returns (model_patcher, vae, device)."""
    sys.path.insert(0, COMFYUI_DIR)
    os.chdir(COMFYUI_DIR)
    import comfy.sd
    import comfy.model_management
    from safetensors import safe_open
    from diffusers import AutoencoderKL
    import torch

    device = comfy.model_management.get_torch_device()
    dit_path = os.path.join(MODEL_DIR, "diffusion_models", "flux-2-klein-4b-fp8.safetensors")
    print(f"Loading DiT: {dit_path}")
    model_patcher = comfy.sd.load_diffusion_model(dit_path)

    vae_path = os.path.join(MODEL_DIR, "vae", "flux2-vae.safetensors")
    vae_sd = {}
    with safe_open(vae_path, framework="pt") as f:
        for key in f.keys():
            vae_sd[key] = f.get_tensor(key)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2, norm_num_groups=32,
        latent_channels=32, scaling_factor=1.0,
    )
    vae.load_state_dict(vae_sd, strict=False)
    vae = vae.to(device, torch.float32).eval()
    return model_patcher, vae, device


def heatmap(diff_abs):
    """Map [0, 255] diff magnitude to a perceptual RGB heatmap (turbo-ish)."""
    # Normalize by a fixed cap so magnitudes are directly comparable between images.
    # 64/255 = ~25% pixel intensity is a "big" diff for visual inspection.
    x = np.clip(diff_abs / 64.0, 0.0, 1.0)
    # Poor-man's turbo lookup: 5 control points.
    # stops: 0.00 = dark navy, 0.25 = blue, 0.50 = green, 0.75 = yellow, 1.00 = red
    stops = np.array([
        [0x30, 0x12, 0x3b],  # dark purple
        [0x47, 0x36, 0x8b],  # blue
        [0x4a, 0xc1, 0x6d],  # green
        [0xfd, 0xe7, 0x25],  # yellow
        [0xfc, 0x57, 0x2f],  # red
    ], dtype=np.float32)
    n_stops = stops.shape[0]
    # Interpolate
    idx_f = x * (n_stops - 1)
    idx_lo = np.floor(idx_f).astype(np.int32)
    idx_hi = np.clip(idx_lo + 1, 0, n_stops - 1)
    t = (idx_f - idx_lo)[..., None]
    lo = stops[idx_lo]
    hi = stops[idx_hi]
    rgb = lo * (1.0 - t) + hi * t
    return rgb.clip(0, 255).astype(np.uint8)


def add_label(img, text, pad=8):
    """Burn a thin text label into the top of an image using a tiny pixel font.
    No PIL dependency — we draw each char from a 5x7 bitmap table."""
    # Super compact 5x7 font (subset — only ASCII we use).
    GLYPHS = {
        " ": "00000/00000/00000/00000/00000/00000/00000",
        "!": "00100/00100/00100/00100/00100/00000/00100",
        ",": "00000/00000/00000/00000/00100/00100/01000",
        ".": "00000/00000/00000/00000/00000/00000/00100",
        "-": "00000/00000/00000/11111/00000/00000/00000",
        "'": "00100/00100/00000/00000/00000/00000/00000",
        "0": "01110/10001/10011/10101/11001/10001/01110",
        "1": "00100/01100/00100/00100/00100/00100/01110",
        "2": "01110/10001/00001/00010/00100/01000/11111",
        "3": "11110/00001/00001/01110/00001/00001/11110",
        "4": "00010/00110/01010/10010/11111/00010/00010",
        "5": "11111/10000/11110/00001/00001/10001/01110",
        "6": "00110/01000/10000/11110/10001/10001/01110",
        "7": "11111/00001/00010/00100/01000/01000/01000",
        "8": "01110/10001/10001/01110/10001/10001/01110",
        "9": "01110/10001/10001/01111/00001/00010/01100",
        "A": "01110/10001/10001/11111/10001/10001/10001",
        "B": "11110/10001/10001/11110/10001/10001/11110",
        "C": "01110/10001/10000/10000/10000/10001/01110",
        "D": "11110/10001/10001/10001/10001/10001/11110",
        "E": "11111/10000/10000/11110/10000/10000/11111",
        "F": "11111/10000/10000/11110/10000/10000/10000",
        "G": "01110/10001/10000/10111/10001/10001/01110",
        "H": "10001/10001/10001/11111/10001/10001/10001",
        "I": "01110/00100/00100/00100/00100/00100/01110",
        "J": "11111/00010/00010/00010/00010/10010/01100",
        "K": "10001/10010/10100/11000/10100/10010/10001",
        "L": "10000/10000/10000/10000/10000/10000/11111",
        "M": "10001/11011/10101/10101/10001/10001/10001",
        "N": "10001/10001/11001/10101/10011/10001/10001",
        "O": "01110/10001/10001/10001/10001/10001/01110",
        "P": "11110/10001/10001/11110/10000/10000/10000",
        "Q": "01110/10001/10001/10001/10101/10010/01101",
        "R": "11110/10001/10001/11110/10100/10010/10001",
        "S": "01111/10000/10000/01110/00001/00001/11110",
        "T": "11111/00100/00100/00100/00100/00100/00100",
        "U": "10001/10001/10001/10001/10001/10001/01110",
        "V": "10001/10001/10001/10001/10001/01010/00100",
        "W": "10001/10001/10001/10101/10101/10101/01010",
        "X": "10001/10001/01010/00100/01010/10001/10001",
        "Y": "10001/10001/01010/00100/00100/00100/00100",
        "Z": "11111/00001/00010/00100/01000/10000/11111",
    }
    scale = 2
    ch_w, ch_h = 5 * scale, 7 * scale
    gap = 1 * scale
    text = text.upper()
    text_w = len(text) * (ch_w + gap)
    text_h = ch_h
    label_h = text_h + 2 * pad
    H, W, _ = img.shape
    out = np.zeros((H + label_h, W, 3), dtype=np.uint8)
    out[label_h:, :, :] = img
    # Dark bar for label bg
    out[:label_h, :, :] = 30
    # Draw text centered
    x0 = max(0, (W - text_w) // 2)
    y0 = pad
    for i, c in enumerate(text):
        glyph = GLYPHS.get(c, GLYPHS[" "])
        rows = glyph.split("/")
        for ry, row in enumerate(rows):
            for rx, px in enumerate(row):
                if px == "1":
                    x = x0 + i * (ch_w + gap) + rx * scale
                    y = y0 + ry * scale
                    out[y:y + scale, x:x + scale, :] = 255
    return out


def compose_row(ours, comfy, label):
    """Build a [ours | heatmap | comfy] strip with a label bar above."""
    diff = np.abs(ours.astype(np.int16) - comfy.astype(np.int16)).max(axis=2).astype(np.uint8)
    hm = heatmap(diff)
    gap = np.full((ours.shape[0], 8, 3), 40, dtype=np.uint8)
    row = np.concatenate([ours, gap, hm, gap, comfy], axis=1)
    stats = f"MAX {int(diff.max())} MEAN {float(diff.mean()):.1f}"
    labeled = add_label(row, f"{label}    {stats}")
    return labeled, diff


def save_png(img, path):
    """Save RGB uint8 array as PNG via stdlib only (no PIL)."""
    import struct
    import zlib
    H, W, _ = img.shape
    raw = bytearray()
    for y in range(H):
        raw.append(0)  # filter: None
        raw.extend(img[y].tobytes())
    comp = zlib.compress(bytes(raw), 9)

    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return out + struct.pack(">I", crc)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", comp)
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def main():
    # Step 1: run our CUDA pipeline for all 6 items FIRST.
    # This produces cuda_flux2_output.ppm (renamed per-item) and
    # cuda_flux2_text_hidden.npy (renamed per-item) in CUDA_DIR.
    print("=" * 70)
    print("Phase 1: our CUDA runs")
    print("=" * 70)
    ours_imgs = {}
    hidden_paths = {}
    for idx, prompt, seed in GALLERY:
        img = run_ours(idx, prompt, seed)
        ours_imgs[idx] = img
        # Cache the text hidden states for ComfyUI reuse
        src_hidden = os.path.join(CUDA_DIR, "cuda_flux2_text_hidden.npy")
        dst_hidden = os.path.join(CUDA_DIR, f"hidden_{idx}.npy")
        subprocess.run(f"cp {src_hidden} {dst_hidden}", shell=True, check=True)
        hidden_paths[idx] = dst_hidden
        # Also save our PNG right away
        save_png(img, os.path.join(CUDA_DIR, f"out_{idx}_ours.png"))
        print(f"  saved out_{idx}_ours.png")

    # Step 2: load ComfyUI model once, run all 6 items.
    print()
    print("=" * 70)
    print("Phase 2: ComfyUI runs")
    print("=" * 70)
    model_patcher, vae, device = load_comfy_model()
    comfy_imgs = {}
    for idx, prompt, seed in GALLERY:
        print(f"  [comfy] {prompt[:60]}... (seed={seed})")
        t0 = time.time()
        img = run_comfy(model_patcher, vae, device, prompt, seed, hidden_paths[idx])
        print(f"  [comfy] done in {time.time() - t0:.1f}s")
        comfy_imgs[idx] = img
        save_png(img, os.path.join(CUDA_DIR, f"out_{idx}_comfy.png"))
        print(f"  saved out_{idx}_comfy.png")

    # Step 3: build composite images.
    print()
    print("=" * 70)
    print("Phase 3: building comparison composites")
    print("=" * 70)
    rows = []
    for idx, prompt, seed in GALLERY:
        label = f"[{idx}] {prompt[:46]}"
        row, diff = compose_row(ours_imgs[idx], comfy_imgs[idx], label)
        save_png(row, os.path.join(CUDA_DIR, f"compare_{idx}.png"))
        print(f"  compare_{idx}.png  max_diff={int(diff.max())} mean_diff={float(diff.mean()):.2f}")
        rows.append(row)

    # Pad rows to the same width before stacking
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad_w = max_w - r.shape[1]
            pad = np.full((r.shape[0], pad_w, 3), 30, dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    # Insert gap rows between items
    gap_row = np.full((12, max_w, 3), 30, dtype=np.uint8)
    stacked = padded[0]
    for r in padded[1:]:
        stacked = np.concatenate([stacked, gap_row, r], axis=0)
    save_png(stacked, os.path.join(CUDA_DIR, "compare_all.png"))
    print(f"  compare_all.png ({stacked.shape[1]}×{stacked.shape[0]})")
    print()
    print("All done. Composite: compare_all.png")


if __name__ == "__main__":
    main()
