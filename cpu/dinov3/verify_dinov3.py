#!/usr/bin/env python3
"""
verify_dinov3.py - Generate DINOv3 reference output and compare with C implementation.

Usage:
  # Step 1: Generate reference output from PyTorch
  python verify_dinov3.py --generate --model facebook/dinov3-vitl16-pretrain-lvd1689m \
      [--image test.png] [--output ref_output.npy]

  # Step 2: Compare C output with reference
  python verify_dinov3.py --compare ref_output.npy c_output.npy

  # Step 3: Dump safetensors weight info
  python verify_dinov3.py --inspect model.safetensors

Dependencies:
  pip install torch transformers pillow numpy safetensors
"""

import argparse
import numpy as np
import sys
import os


def generate_gradient_image(size=512):
    """Generate the same synthetic gradient image as the C test."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            img[y, x, 0] = int(255 * x / (size - 1))  # R
            img[y, x, 1] = int(255 * y / (size - 1))  # G
            img[y, x, 2] = 128                          # B
    return img


def generate_reference(model_name, image_path=None, output_path="ref_output.npy",
                       save_weights=False, weights_dir="weights"):
    """Run DINOv3 through PyTorch and save reference output."""
    import torch
    from PIL import Image

    print(f"Loading model: {model_name}")

    # Try loading with transformers
    try:
        from transformers import AutoModel, AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.eval()

        # Prepare image
        if image_path:
            img = Image.open(image_path).convert("RGB")
        else:
            img_np = generate_gradient_image(512)
            img = Image.fromarray(img_np)

        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # Try different output formats
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[0].numpy()
        elif isinstance(outputs, tuple):
            features = outputs[0][0].numpy() if outputs[0].dim() == 3 else outputs[0].numpy()
        else:
            features = outputs.numpy()

        print(f"Output shape: {features.shape}")
        print(f"Output stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.6f}")
        print(f"CLS token [0:8]: {features[0, :8]}")

        np.save(output_path, features)
        print(f"Saved reference output to {output_path}")
        return features

    except Exception as e:
        print(f"transformers loading failed: {e}")
        print("Trying manual loading...")

    # Manual loading fallback for DINOv3
    try:
        import torch
        from safetensors.torch import load_file
        from torchvision import transforms

        # Load weights
        st_path = model_name
        if not os.path.exists(st_path):
            from huggingface_hub import hf_hub_download
            st_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")

        state_dict = load_file(st_path)

        print(f"\nLoaded {len(state_dict)} tensors:")
        for k, v in sorted(state_dict.items())[:20]:
            print(f"  {k}: {v.shape} {v.dtype}")
        if len(state_dict) > 20:
            print(f"  ... ({len(state_dict) - 20} more)")

        # Auto-detect architecture
        embed_dim = None
        patch_size = None
        n_blocks = 0
        n_storage = 0

        for k, v in state_dict.items():
            if "patch_embed" in k and "weight" in k and v.dim() == 4:
                embed_dim = v.shape[0]
                patch_size = v.shape[2]
            if "register_tokens" in k or "storage_tokens" in k:
                n_storage = v.shape[1] if v.dim() == 3 else v.shape[0]
            if "blocks." in k:
                L = int(k.split("blocks.")[1].split(".")[0])
                n_blocks = max(n_blocks, L + 1)

        print(f"\nDetected: embed_dim={embed_dim}, patch_size={patch_size}, "
              f"n_blocks={n_blocks}, n_storage={n_storage}")

        if save_weights:
            os.makedirs(weights_dir, exist_ok=True)
            for k, v in state_dict.items():
                safe_name = k.replace("/", "_")
                np.save(f"{weights_dir}/{safe_name}.npy", v.numpy())
            print(f"Saved weights to {weights_dir}/")

        np.save(output_path, np.zeros((1, embed_dim or 1024)))
        print(f"Note: Manual forward pass not implemented yet. "
              f"Use transformers or implement manually.")
        return None

    except Exception as e:
        print(f"Manual loading also failed: {e}")
        return None


def compare_outputs(ref_path, c_path, tolerance=0.01):
    """Compare reference and C implementation outputs."""
    ref = np.load(ref_path)
    c_out = np.load(c_path)

    print(f"Reference shape: {ref.shape}")
    print(f"C output shape:  {c_out.shape}")

    if ref.shape != c_out.shape:
        print(f"ERROR: Shape mismatch!")
        return False

    # Per-token correlation
    n_tokens = ref.shape[0]
    correlations = []
    for t in range(n_tokens):
        r = np.corrcoef(ref[t], c_out[t])[0, 1]
        correlations.append(r)

    correlations = np.array(correlations)
    mean_corr = np.nanmean(correlations)
    min_corr = np.nanmin(correlations)

    print(f"\nPer-token correlation:")
    print(f"  Mean: {mean_corr:.6f}")
    print(f"  Min:  {min_corr:.6f}")
    print(f"  Tokens with r < 0.999: {np.sum(correlations < 0.999)}/{n_tokens}")
    print(f"  Tokens with r < 0.99:  {np.sum(correlations < 0.99)}/{n_tokens}")

    # Overall stats
    abs_diff = np.abs(ref - c_out)
    rel_diff = abs_diff / (np.abs(ref) + 1e-8)
    print(f"\nAbsolute difference:")
    print(f"  Max:  {abs_diff.max():.6f}")
    print(f"  Mean: {abs_diff.mean():.6f}")
    print(f"\nRelative difference:")
    print(f"  Max:  {rel_diff.max():.6f}")
    print(f"  Mean: {rel_diff.mean():.6f}")

    # CLS token comparison
    print(f"\nCLS token [0:8]:")
    print(f"  Ref: {ref[0, :8]}")
    print(f"  C:   {c_out[0, :8]}")

    # Summary
    passed = min_corr > 0.99
    print(f"\n{'PASS' if passed else 'FAIL'}: min correlation = {min_corr:.6f} "
          f"(threshold: 0.99)")
    return passed


def inspect_safetensors(path):
    """Print all tensor names and shapes from a safetensors file."""
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as f:
        keys = f.keys()
        print(f"Safetensors file: {path}")
        print(f"Total tensors: {len(keys)}")
        print()

        # Group by prefix
        groups = {}
        for k in sorted(keys):
            tensor = f.get_tensor(k)
            prefix = k.split(".")[0] if "." in k else k
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((k, tensor.shape, tensor.dtype))

        for prefix in sorted(groups):
            print(f"[{prefix}]")
            for name, shape, dtype in groups[prefix]:
                print(f"  {name}: {list(shape)} ({dtype})")
            print()


def main():
    parser = argparse.ArgumentParser(description="DINOv3 verification tool")
    parser.add_argument("--generate", action="store_true",
                        help="Generate reference output from PyTorch")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "C_OUTPUT"),
                        help="Compare reference and C output .npy files")
    parser.add_argument("--inspect", type=str,
                        help="Inspect safetensors file")
    parser.add_argument("--model", type=str,
                        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                        help="HuggingFace model name or local path")
    parser.add_argument("--image", type=str, default=None,
                        help="Input image path (default: synthetic gradient)")
    parser.add_argument("--output", type=str, default="ref_output.npy",
                        help="Output .npy file path")
    parser.add_argument("--save-weights", action="store_true",
                        help="Save individual weight tensors as .npy files")

    args = parser.parse_args()

    if args.inspect:
        inspect_safetensors(args.inspect)
    elif args.generate:
        generate_reference(args.model, args.image, args.output,
                           save_weights=args.save_weights)
    elif args.compare:
        success = compare_outputs(args.compare[0], args.compare[1])
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
