#!/usr/bin/env python3
"""
Quantize TRELLIS.2 Stage-1 DiT weights to FP8 E4M3 with per-tensor scales.

Quantizes only 2D weight matrices that the BF16-act × FP8-weight WMMA kernel
in rdna4/trellis2 can dispatch on (n_out % 64 == 0 AND n_in % 16 == 0).
All other tensors (norms, biases, modulation, small projections like
input_layer / out_layer) are passed through unchanged.

Convention (matching rdna4/flux2/hip_flux2_runner.c gpu_upload_st_wt):
  <name>.weight        : F8_E4M3 raw bytes, shape [n_out, n_in]
  <name>.weight.scale  : F32 scalar = max_abs / 448

Reconstruction in-kernel: f32 = lut[fp8_byte] * scale.

Usage:
  python quantize_dit_fp8.py <input.safetensors> <output.safetensors>
"""
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file


FP8_MAX = 448.0  # E4M3 max representable absolute value
KERNEL_M_ALIGN = 64
KERNEL_K_ALIGN = 16

# Tensors consumed by custom kernels (not via the BF16xFP8 GEMM dispatcher) must
# stay in F32/BF16 even if their shape is kernel-eligible.
EXCLUDE_NAMES = (
    'adaLN_modulation',  # consumed by fn_modulation, expects F32
    't_embedder',        # always called with n_tok=1; WMMA path needs n_tok>=16
)


def is_fp8_eligible(name, shape):
    if any(ex in name for ex in EXCLUDE_NAMES):
        return False
    if len(shape) != 2:
        return False
    n_out, n_in = shape
    return (n_out >= KERNEL_M_ALIGN and n_in >= KERNEL_K_ALIGN
            and n_out % KERNEL_M_ALIGN == 0 and n_in % KERNEL_K_ALIGN == 0)


def quantize_to_e4m3(t_bf16):
    """t_bf16: BF16 tensor. Returns (fp8_bytes_uint8, scale_f32)."""
    t_f32 = t_bf16.to(torch.float32)
    max_abs = t_f32.abs().max().item()
    if max_abs <= 1e-12:
        scale = 1.0
    else:
        scale = max_abs / FP8_MAX
    scaled = t_f32 / scale
    # Use torch's native E4M3 cast (PyTorch 2.1+).
    fp8 = scaled.to(torch.float8_e4m3fn)
    # Reinterpret as uint8 for safetensors (which doesn't yet have F8_E4M3 in
    # all toolchains; the C side reads raw bytes via dtype tag).
    return fp8, torch.tensor(scale, dtype=torch.float32)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src_path, dst_path = sys.argv[1], sys.argv[2]

    out = {}
    n_quantized = 0
    n_passthrough = 0
    src_bytes_quant = 0
    dst_bytes_quant = 0

    with safe_open(src_path, framework='pt') as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            shape = list(t.shape)
            if k.endswith('.weight') and is_fp8_eligible(k, shape):
                fp8, scale = quantize_to_e4m3(t)
                out[k] = fp8
                out[k + '.scale'] = scale
                n_quantized += 1
                src_bytes_quant += t.numel() * 2  # BF16 = 2 bytes
                dst_bytes_quant += fp8.numel() * 1  # E4M3 = 1 byte
            else:
                out[k] = t
                n_passthrough += 1

    # safetensors metadata: record convention so a future loader can sanity-check.
    metadata = {
        'fp8_format': 'E4M3',
        'fp8_max': str(FP8_MAX),
        'fp8_eligible_align_m': str(KERNEL_M_ALIGN),
        'fp8_eligible_align_k': str(KERNEL_K_ALIGN),
        'scale_convention': 'sibling_tensor_dot_scale',
    }
    save_file(out, dst_path, metadata=metadata)

    print(f'Quantized {n_quantized} weight matrices to FP8 E4M3.')
    print(f'Pass-through (norms / biases / small / 1D weights): {n_passthrough}.')
    print(f'Quantized payload: {src_bytes_quant/1e6:.1f} MB BF16  ->  '
          f'{dst_bytes_quant/1e6:.1f} MB FP8  '
          f'({100.0 * dst_bytes_quant / max(src_bytes_quant, 1):.1f}%)')
    print(f'Wrote {dst_path}')


if __name__ == '__main__':
    main()
