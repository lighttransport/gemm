#!/usr/bin/env bash
# setup_ref.sh - Set up PyTorch reference environment and download model weights
#
# Usage:
#   cd cuda/hy3d && bash setup_ref.sh [--models-dir /path/to/models]
#
# What this does:
#   1. Creates ref/ directory with uv Python venv + PyTorch + deps
#   2. Downloads Hunyuan3D-2.1 model weights (conditioner, DiT, VAE)
#   3. Exports .ckpt to .safetensors for the C runner
#   4. Copies reference scripts (dump_dinov2, dump_vae, dump_dit, compare)
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/getting-started/installation/)
#   - ~8GB disk for model weights
#
# SPDX-License-Identifier: MIT
# Copyright 2025 - Present, Light Transport Entertainment Inc.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REF_DIR="${SCRIPT_DIR}/ref"
MODELS_DIR="${1:-/mnt/nvme02/models/Hunyuan3D-2.1}"

# Parse --models-dir flag
for arg in "$@"; do
    case $arg in
        --models-dir=*) MODELS_DIR="${arg#*=}" ;;
        --models-dir)   shift; MODELS_DIR="${1:-$MODELS_DIR}" ;;
    esac
done

echo "=== Hunyuan3D-2.1 Reference Environment Setup ==="
echo "  Ref dir:    ${REF_DIR}"
echo "  Models dir: ${MODELS_DIR}"
echo ""

# ---- Step 0: Check uv ----
if ! command -v uv &>/dev/null; then
    echo "ERROR: 'uv' not found. Install from https://docs.astral.sh/uv/"
    exit 1
fi
echo "[0/4] uv found: $(uv --version)"

# ---- Step 1: Create ref directory + Python venv ----
echo "[1/4] Setting up Python environment..."
mkdir -p "${REF_DIR}"

cat > "${REF_DIR}/pyproject.toml" << 'PYPROJECT'
[project]
name = "hy3d-reference"
version = "0.1.0"
description = "PyTorch reference for Hunyuan3D-2.1 verification"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.3",
    "torchvision",
    "safetensors",
    "numpy",
    "Pillow",
    "transformers",
    "pyyaml",
    "trimesh",
    "einops",
    "diffusers",
    "scikit-image",
]
PYPROJECT

cat > "${REF_DIR}/.gitignore" << 'GITIGNORE'
.venv/
__pycache__/
output/
*.npy
*.npz
uv.lock
GITIGNORE

cd "${REF_DIR}"
uv sync
echo "  Python venv ready at ${REF_DIR}/.venv"

# ---- Step 2: Download model weights ----
echo "[2/4] Checking model weights..."
mkdir -p "${MODELS_DIR}"

DIT_DIR="${MODELS_DIR}/hunyuan3d-dit-v2-1"
VAE_DIR="${MODELS_DIR}/hunyuan3d-vae-v2-1"

# Download from HuggingFace if not present
if [ ! -f "${DIT_DIR}/model.fp16.ckpt" ]; then
    echo "  Downloading DiT weights from HuggingFace..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download tencent/Hunyuan3D-2 \
            hunyuan3d-dit-v2-1/model.fp16.ckpt \
            hunyuan3d-dit-v2-1/config.yaml \
            --local-dir "${MODELS_DIR}" --local-dir-use-symlinks False
    else
        echo "  WARN: huggingface-cli not found. Install with: pip install huggingface_hub[cli]"
        echo "  Manual download: https://huggingface.co/tencent/Hunyuan3D-2"
        echo "  Place files in ${DIT_DIR}/"
    fi
else
    echo "  DiT weights found: ${DIT_DIR}/model.fp16.ckpt"
fi

if [ ! -f "${VAE_DIR}/model.fp16.safetensors" ]; then
    echo "  Downloading VAE weights from HuggingFace..."
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download tencent/Hunyuan3D-2 \
            hunyuan3d-vae-v2-1/model.fp16.safetensors \
            hunyuan3d-vae-v2-1/config.yaml \
            --local-dir "${MODELS_DIR}" --local-dir-use-symlinks False
    else
        echo "  WARN: huggingface-cli not found."
        echo "  Place files in ${VAE_DIR}/"
    fi
else
    echo "  VAE weights found: ${VAE_DIR}/model.fp16.safetensors"
fi

# ---- Step 3: Export .ckpt to .safetensors ----
echo "[3/4] Exporting safetensors for C runner..."

cat > "${REF_DIR}/export_safetensors.py" << 'EXPORT_PY'
"""Export model components from combined .ckpt to individual .safetensors files."""
import argparse, os, torch
from safetensors.torch import save_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)

    # Conditioner (DINOv2)
    cond_path = os.path.join(args.outdir, "conditioner.safetensors")
    if not os.path.exists(cond_path):
        sd = {k.replace("conditioner.", "", 1) if k.startswith("conditioner.") else k: v
              for k, v in ckpt["conditioner"].items()}
        print(f"  Saving conditioner ({len(sd)} tensors) -> {cond_path}")
        save_file(sd, cond_path)
    else:
        print(f"  Conditioner exists: {cond_path}")

    # DiT model
    model_path = os.path.join(args.outdir, "model.safetensors")
    if not os.path.exists(model_path):
        print(f"  Saving DiT ({len(ckpt['model'])} tensors) -> {model_path}")
        save_file(ckpt["model"], model_path)
    else:
        print(f"  DiT exists: {model_path}")

    print("Done.")

if __name__ == "__main__":
    main()
EXPORT_PY

if [ -f "${DIT_DIR}/model.fp16.ckpt" ]; then
    if [ ! -f "${MODELS_DIR}/conditioner.safetensors" ] || [ ! -f "${MODELS_DIR}/model.safetensors" ]; then
        cd "${REF_DIR}"
        uv run python export_safetensors.py \
            --ckpt "${DIT_DIR}/model.fp16.ckpt" \
            --outdir "${MODELS_DIR}"
    else
        echo "  Safetensors already exported"
    fi
else
    echo "  SKIP: ${DIT_DIR}/model.fp16.ckpt not found"
fi

# ---- Step 4: Write reference scripts ----
echo "[4/4] Writing reference scripts..."

# --- dump_dinov2.py ---
cat > "${REF_DIR}/dump_dinov2.py" << 'DUMP_DINO'
"""Dump DINOv2 encoder outputs for verification.
Usage: uv run python dump_dinov2.py --ckpt <path> [--image path.png] [--outdir output/]"""
import argparse, os, numpy as np, torch
from PIL import Image

def preprocess_image(img_path, size=518):
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    return ((img - mean) / std).transpose(2, 0, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="output")
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cond_sd = {k.replace("conditioner.", "", 1) if k.startswith("conditioner.") else k: v
               for k, v in ckpt["conditioner"].items()}

    from transformers import Dinov2Model, Dinov2Config
    config = Dinov2Config(hidden_size=1024, num_attention_heads=16, num_hidden_layers=24,
                          intermediate_size=4096, patch_size=14, image_size=518,
                          layer_norm_eps=1e-6, qkv_bias=True, layerscale_value=1.0)
    model = Dinov2Model(config)
    prefix = "main_image_encoder.model."
    new_sd = {k[len(prefix):]: v for k, v in cond_sd.items() if k.startswith(prefix)}
    model.load_state_dict(new_sd, strict=False)
    model = model.float().eval()

    if args.image:
        img = preprocess_image(args.image)
    else:
        np.random.seed(42)
        img = np.random.randn(3, 518, 518).astype(np.float32) * 0.1
    np.save(os.path.join(args.outdir, "dinov2_input.npy"), img)

    with torch.no_grad():
        out = model(torch.from_numpy(img).unsqueeze(0).float(), output_hidden_states=True)
    result = out.last_hidden_state.numpy()[0]
    np.save(os.path.join(args.outdir, "dinov2_output.npy"), result)
    print(f"  Output: {result.shape}, mean={result.mean():.6f}, std={result.std():.6f}")
    if out.hidden_states:
        for i, hs in enumerate(out.hidden_states):
            if i in [0, 12, 23, 24]:
                np.save(os.path.join(args.outdir, f"dinov2_hidden_{i}.npy"), hs.numpy()[0])
    print(f"Saved to {args.outdir}/")

if __name__ == "__main__":
    main()
DUMP_DINO

# --- dump_vae.py ---
cat > "${REF_DIR}/dump_vae.py" << 'DUMP_VAE'
"""Dump ShapeVAE decoder outputs for verification.
Usage: uv run python dump_vae.py --vae-path <safetensors> [--grid-res 8] [--outdir output/]"""
import argparse, os, numpy as np, torch, torch.nn.functional as F
from safetensors.torch import load_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="output")
    parser.add_argument("--vae-path", type=str, required=True)
    parser.add_argument("--grid-res", type=int, default=32)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sd = load_file(args.vae_path)
    W, H, HD = 1024, 16, 64

    torch.manual_seed(42)
    latents = torch.randn(1, 4096, 64, dtype=torch.float32)
    np.save(os.path.join(args.outdir, "vae_input_latents.npy"), latents.numpy()[0])

    # Post-KL
    x = F.linear(latents, sd["post_kl.weight"].float(), sd["post_kl.bias"].float())
    np.save(os.path.join(args.outdir, "vae_post_kl.npy"), x.numpy()[0])

    # 16 transformer blocks
    for bi in range(16):
        p = f"transformer.resblocks.{bi}."
        normed = F.layer_norm(x, [W], sd[p+"ln_1.weight"].float(), sd[p+"ln_1.bias"].float(), eps=1e-6)
        qkv = F.linear(normed, sd[p+"attn.c_qkv.weight"].float())
        B, N, _ = qkv.shape
        qkv = qkv.reshape(B, N, H, 3, HD)
        q, k, v = qkv.unbind(dim=3)
        q = F.layer_norm(q, [HD], sd[p+"attn.attention.q_norm.weight"].float(),
                         sd[p+"attn.attention.q_norm.bias"].float(), eps=1e-6)
        k = F.layer_norm(k, [HD], sd[p+"attn.attention.k_norm.weight"].float(),
                         sd[p+"attn.attention.k_norm.bias"].float(), eps=1e-6)
        attn = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2),
                                              v.transpose(1,2)).transpose(1,2).reshape(B, N, W)
        attn = F.linear(attn, sd[p+"attn.c_proj.weight"].float(), sd[p+"attn.c_proj.bias"].float())
        x = x + attn
        n2 = F.layer_norm(x, [W], sd[p+"ln_2.weight"].float(), sd[p+"ln_2.bias"].float(), eps=1e-6)
        h = F.gelu(F.linear(n2, sd[p+"mlp.c_fc.weight"].float(), sd[p+"mlp.c_fc.bias"].float()))
        x = x + F.linear(h, sd[p+"mlp.c_proj.weight"].float(), sd[p+"mlp.c_proj.bias"].float())
        if bi in [0, 8, 15]:
            np.save(os.path.join(args.outdir, f"vae_block_{bi}.npy"), x.numpy()[0])
            print(f"  Block {bi}: mean={x.mean():.6f} std={x.std():.6f}")

    np.save(os.path.join(args.outdir, "vae_decoded_latents.npy"), x.numpy()[0])

    # SDF query
    gr = args.grid_res
    c1d = torch.linspace(-1, 1, gr)
    gx, gy, gz = torch.meshgrid(c1d, c1d, c1d, indexing='ij')
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    # Fourier embed (include_pi=false)
    freqs = 2.0 ** torch.arange(8, dtype=torch.float32)
    embed = coords.unsqueeze(-1) * freqs
    embed = embed.reshape(coords.shape[0], -1)
    qe = torch.cat([coords, torch.sin(embed), torch.cos(embed)], dim=-1)

    cp = "geo_decoder."
    qp = F.linear(qe, sd[cp+"query_proj.weight"].float(), sd[cp+"query_proj.bias"].float())
    qp = qp.unsqueeze(0)

    cap = cp + "cross_attn_decoder."
    ql = F.layer_norm(qp, [W], sd[cap+"ln_1.weight"].float(), sd[cap+"ln_1.bias"].float(), eps=1e-6)
    kl = F.layer_norm(x, [W], sd[cap+"ln_2.weight"].float(), sd[cap+"ln_2.bias"].float(), eps=1e-6)
    Q = F.linear(ql, sd[cap+"attn.c_q.weight"].float())
    KV = F.linear(kl, sd[cap+"attn.c_kv.weight"].float())
    Nq, Nkv = qp.shape[1], 4096
    KV = KV.reshape(1, Nkv, H, 2, HD); K, V = KV.unbind(dim=3)
    Q = Q.reshape(1, Nq, H, HD)
    Q = F.layer_norm(Q, [HD], sd[cap+"attn.attention.q_norm.weight"].float(),
                     sd[cap+"attn.attention.q_norm.bias"].float(), eps=1e-6)
    K = F.layer_norm(K, [HD], sd[cap+"attn.attention.k_norm.weight"].float(),
                     sd[cap+"attn.attention.k_norm.bias"].float(), eps=1e-6)
    cs = 8192
    outs = []
    for i in range(0, Nq, cs):
        outs.append(F.scaled_dot_product_attention(
            Q[:,i:i+cs].transpose(1,2), K.transpose(1,2), V.transpose(1,2)).transpose(1,2))
    ao = torch.cat(outs, dim=1).reshape(1, Nq, W)
    ao = F.linear(ao, sd[cap+"attn.c_proj.weight"].float(), sd[cap+"attn.c_proj.bias"].float())
    qo = qp + ao
    m = F.layer_norm(qo, [W], sd[cap+"ln_3.weight"].float(), sd[cap+"ln_3.bias"].float(), eps=1e-6)
    m = F.gelu(F.linear(m, sd[cap+"mlp.c_fc.weight"].float(), sd[cap+"mlp.c_fc.bias"].float()))
    qo = qo + F.linear(m, sd[cap+"mlp.c_proj.weight"].float(), sd[cap+"mlp.c_proj.bias"].float())
    qo = F.layer_norm(qo, [W], sd[cp+"ln_post.weight"].float(), sd[cp+"ln_post.bias"].float(), eps=1e-6)
    sdf = F.linear(qo, sd[cp+"output_proj.weight"].float(), sd[cp+"output_proj.bias"].float())
    sdf = sdf.squeeze(0).squeeze(-1).numpy().reshape(gr, gr, gr)
    np.save(os.path.join(args.outdir, "vae_sdf_grid.npy"), sdf)
    print(f"  SDF: {sdf.shape}, min={sdf.min():.6f} max={sdf.max():.6f}")
    print(f"Saved to {args.outdir}/")

if __name__ == "__main__":
    main()
DUMP_VAE

# --- dump_dit_single_step.py ---
cat > "${REF_DIR}/dump_dit_single_step.py" << 'DUMP_DIT'
"""Dump DiT single-step output for verification.
Usage: uv run python dump_dit_single_step.py --ckpt <path> [--outdir output/]"""
import argparse, os, sys, math, importlib.util, numpy as np, torch, torch.nn as nn

def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

HY3D_REPO = os.environ.get("HY3D_REPO",
    "/mnt/nvme02/work/vision-language.cpp/main/verif/3d/text-to-mesh/hunyuan3d-repo")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="output")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--hy3d-repo", type=str, default=HY3D_REPO)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = ckpt["model"]

    # Direct-import DiT class (avoid transitive deps)
    moe_path = os.path.join(args.hy3d_repo, "hy3dgen/shapegen/models/denoisers/moe_layers.py")
    dit_path = os.path.join(args.hy3d_repo, "hy3dgen/shapegen/models/denoisers/hunyuandit.py")
    if not os.path.exists(dit_path):
        print(f"ERROR: {dit_path} not found. Set --hy3d-repo or HY3D_REPO env var.")
        sys.exit(1)

    moe = import_file("hy3dgen.shapegen.models.denoisers.moe_layers", moe_path)
    for pkg in ["hy3dgen", "hy3dgen.shapegen", "hy3dgen.shapegen.models",
                "hy3dgen.shapegen.models.denoisers"]:
        sys.modules.setdefault(pkg, type(sys)(pkg))
    sys.modules["hy3dgen.shapegen.models.denoisers.moe_layers"] = moe
    dit_mod = import_file("hy3dgen.shapegen.models.denoisers.hunyuandit", dit_path)

    model = dit_mod.HunYuanDiTPlain(
        input_size=4096, in_channels=64, hidden_size=2048, context_dim=1024,
        depth=21, num_heads=16, qk_norm=True, text_len=1370,
        with_decoupled_ca=False, use_attention_pooling=False,
        qk_norm_type='rms', qkv_bias=False, use_pos_emb=False,
        num_moe_layers=6, num_experts=8, moe_top_k=2)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.float().eval()  # CPU (DiT too large for 8GB GPU with activations)
    torch.manual_seed(42)
    latents = torch.randn(1, 4096, 64)
    context = torch.randn(1, 1370, 1024)
    t = torch.tensor([0.5])

    np.save(os.path.join(args.outdir, "dit_input_latents.npy"), latents.numpy()[0])
    np.save(os.path.join(args.outdir, "dit_input_context.npy"), context.numpy()[0])

    with torch.no_grad():
        output = model(latents, t, {"main": context})
    out_np = output.float().cpu().numpy()[0]
    np.save(os.path.join(args.outdir, "dit_output.npy"), out_np)
    print(f"  Output: {out_np.shape}, mean={out_np.mean():.6f}, std={out_np.std():.6f}")

    # Timestep embedding reference
    class Timesteps(nn.Module):
        def __init__(self, dim, max_period=10000):
            super().__init__()
            self.dim = dim; self.max_period = max_period
        def forward(self, t):
            half = self.dim // 2
            exp = -math.log(self.max_period) * torch.arange(0, half, dtype=torch.float32) / half
            emb = t[:, None].float() * torch.exp(exp)[None, :]
            return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    t_emb = Timesteps(2048)(torch.tensor([0.5])).numpy()[0]
    np.save(os.path.join(args.outdir, "dit_timestep_embed.npy"), t_emb)
    print(f"Saved to {args.outdir}/")

if __name__ == "__main__":
    main()
DUMP_DIT

# --- compare.py ---
cat > "${REF_DIR}/compare.py" << 'COMPARE'
"""Compare reference .npy outputs against CUDA runner outputs.
Usage: uv run python compare.py <ref_dir> <test_dir> [rtol] [atol]"""
import sys, os, numpy as np

def compare(name, ref, test, rtol=1e-3, atol=1e-4):
    if ref.shape != test.shape:
        print(f"  {name}: SHAPE MISMATCH ref={ref.shape} test={test.shape}")
        return False
    diff = np.abs(ref - test)
    ok = np.allclose(ref, test, rtol=rtol, atol=atol)
    status = "OK" if ok else "FAIL"
    print(f"  {name}: {status}  max={diff.max():.2e} mean={diff.mean():.2e} shape={ref.shape}")
    if not ok:
        idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"    worst@{idx}: ref={ref[idx]:.6f} test={test[idx]:.6f}")
    return ok

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ref_dir> <test_dir> [rtol] [atol]"); sys.exit(1)
    ref_dir, test_dir = sys.argv[1], sys.argv[2]
    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
    ref_files = {f for f in os.listdir(ref_dir) if f.endswith(".npy")}
    test_files = {f for f in os.listdir(test_dir) if f.endswith(".npy")}
    common = sorted(ref_files & test_files)
    ok = fail = 0
    for f in common:
        if compare(f, np.load(os.path.join(ref_dir, f)), np.load(os.path.join(test_dir, f)),
                   rtol=rtol, atol=atol):
            ok += 1
        else:
            fail += 1
    print(f"\n{ok} OK, {fail} FAIL / {len(common)} compared")

if __name__ == "__main__":
    main()
COMPARE

echo ""
echo "=== Setup complete ==="
echo ""
echo "Model weight files:"
for f in "${MODELS_DIR}/conditioner.safetensors" \
         "${MODELS_DIR}/model.safetensors" \
         "${MODELS_DIR}/hunyuan3d-vae-v2-1/model.fp16.safetensors"; do
    if [ -f "$f" ]; then
        echo "  OK  $f ($(du -sh "$f" | cut -f1))"
    else
        echo "  MISSING  $f"
    fi
done

echo ""
echo "Generate reference outputs:"
echo "  cd ${REF_DIR}"
echo "  uv run python dump_dinov2.py --ckpt ${DIT_DIR}/model.fp16.ckpt"
echo "  uv run python dump_vae.py --vae-path ${VAE_DIR}/model.fp16.safetensors --grid-res 8"
echo "  uv run python dump_dit_single_step.py --ckpt ${DIT_DIR}/model.fp16.ckpt"
echo ""
echo "Compare CUDA vs reference:"
echo "  uv run python compare.py output/ <cuda_output_dir>/"
echo ""
echo "Run CUDA runner:"
echo "  cd ${SCRIPT_DIR} && make"
echo "  ./test_cuda_hy3d ${MODELS_DIR}/conditioner.safetensors \\"
echo "                   ${MODELS_DIR}/model.safetensors \\"
echo "                   ${MODELS_DIR}/hunyuan3d-vae-v2-1/model.fp16.safetensors \\"
echo "                   -i test.ppm -o output.obj"
