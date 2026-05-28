#!/usr/bin/env python3
"""
nunchaku_dump_reference.py  —  RUN THIS ON A CUDA / NVIDIA BOX where Nunchaku actually works.

Why this exists
---------------
We are porting the Nunchaku INT4 (SVDQuant) Qwen-Image checkpoint to RDNA4. The
weight *layout* swizzle is already solved and bit-exact-proven offline. What is
NOT yet pinned is the SVDQuant *numeric semantics* — the exact composition of:
  - the smoothing diagonal (smooth_factor): is the activation divided or multiplied by it?
  - the rank-128 low-rank branch: does proj_down see the smoothed or the raw activation?
  - the per-group weight/activation scale association.
Those conventions live inside Nunchaku's fused CUDA kernels (svdq_*_w4a4_cuda) and
cannot be read from the pure-Python shim. So instead we capture a small ground-truth
sample from the REAL kernels for ONE linear layer; offline (on the AMD box) we then
tune our CPU decode until it reproduces this sample — turning guesswork into a
deterministic fit. The FP8 checkpoint was too loose a proxy; this dump is the true oracle.

What it captures (for one layer, default the block-0 MLP first linear / fc1)
---------------------------------------------------------------------------
  x            [N, in]    input activation fed to the real module (deterministic, fixed seed)
  y            [N, out]   the module's full forward output      <-- operational oracle
  quantized_x  [N, in/2]  uint8  int4-quantized (smoothed) activation  ) the `quantize()`
  ascales      [...]             per-group activation scales            ) intermediates —
  lora_act_out [N, rank]  fp32   low-rank activation projection         ) reveal smoothing
                                                                        ) direction + low-rank basis
  qweight, wscales, proj_down, proj_up, smooth_factor, smooth_factor_orig, bias
               the layer's LIVE post-load tensors (so any load-time transform is captured too)

Plus a JSON metadata blob: in/out features, rank, group_size, precision, seed, and any
scalar module attrs (wtscale/alpha, act_unsigned, presence of wcscales — these distinguish
the int4 path from the nvfp4 two-level-scale path).

Output: a single self-contained `nunchaku_ref_dump.safetensors`. Copy it back to the AMD
box — a few MB beyond the one layer's tensors.

Usage
-----
  pip install nunchaku  # plus its prerequisites (torch+cuda, diffusers); see nunchaku.tech docs
  python nunchaku_dump_reference.py \
      --ckpt nunchaku-ai/nunchaku-qwen-image  \   # HF repo id OR a local path to the .safetensors
      --variant svdq-int4_r128-qwen-image.safetensors \
      --layer transformer_blocks.0.img_mlp.net.0.proj \
      --out  nunchaku_ref_dump.safetensors

The ONLY part likely to need tweaking for your installed Nunchaku version is the
"load the transformer" section — everything after `layer = _resolve(model, args.layer)`
uses the documented SVDQW4A4Linear .quantize()/.forward() interface and is version-stable.
If the load API differs, hand the docstring of this file to an agent on the CUDA box and
have it adapt only that section (the dump spec below is the contract).
"""
import argparse, json
import torch


def _resolve(root, dotted):
    """Navigate a dotted module path, handling Sequential indices like 'net.0'."""
    obj = root
    for part in dotted.split('.'):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='HF repo id (nunchaku-ai/nunchaku-qwen-image) or local dir/file')
    ap.add_argument('--variant', default='svdq-int4_r128-qwen-image.safetensors',
                    help='which safetensors variant inside the repo (the base int4_r128, NOT a lightning one)')
    ap.add_argument('--layer', default='transformer_blocks.0.img_mlp.net.0.proj',
                    help='dotted module path of the SVDQW4A4Linear to probe')
    ap.add_argument('--out', default='nunchaku_ref_dump.safetensors')
    ap.add_argument('--n', type=int, default=256, help='batch rows (256 == Nunchaku pad_size; avoids batch padding)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    # ---- load the Nunchaku Qwen-Image transformer (the API-dependent part) ----
    # Most likely current API; adjust to your installed version if it differs.
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
    model = NunchakuQwenImageTransformer2DModel.from_pretrained(args.ckpt)  # may take a local file path too
    model = model.to('cuda').eval()

    layer = _resolve(model, args.layer)
    print('resolved layer:', args.layer, '->', type(layer).__name__)
    print('layer methods present:', [m for m in ('quantize', 'forward_quant', 'forward') if hasattr(layer, m)])

    in_features  = getattr(layer, 'in_features')
    out_features = getattr(layer, 'out_features')
    dtype = layer.proj_up.dtype          # bf16 for the int4 checkpoint
    print(f'in={in_features} out={out_features} dtype={dtype}')

    # ---- deterministic input (well-conditioned so the activation quantizer behaves normally) ----
    g = torch.Generator(device='cuda').manual_seed(args.seed)
    x2d = torch.randn(args.n, in_features, generator=g, device='cuda', dtype=dtype)   # (N, in)
    x3d = x2d.reshape(1, args.n, in_features)                                          # forward wants (B, S, in)

    with torch.no_grad():
        # operational oracle: the real fused forward
        y = layer(x3d).reshape(args.n, out_features)
        # mechanistic oracle: the quantize() intermediates that expose smoothing + low-rank basis
        quantized_x, ascales, lora_act_out = layer.quantize(x2d)

    # ---- collect the layer's live (post-load) tensors so any load-time transform is captured ----
    live = {}
    for name, t in list(layer.named_parameters()) + list(layer.named_buffers()):
        if t is not None:
            live['weight.' + name] = t.detach().to('cpu').contiguous()

    tensors = {
        'x': x2d.to('cpu').contiguous(),
        'y': y.to('cpu').contiguous(),
        'quantized_x': quantized_x.to('cpu').contiguous(),
        'ascales': ascales.to('cpu').contiguous(),
        'lora_act_out': lora_act_out.float().to('cpu').contiguous(),
        **live,
    }

    # scalar / non-tensor attrs that disambiguate int4 vs nvfp4 conventions
    def _scalar(attr):
        v = getattr(layer, attr, None)
        if v is None: return None
        if torch.is_tensor(v): return v.float().flatten()[:8].tolist() + (['...'] if v.numel() > 8 else [])
        return v
    meta = {
        'layer': args.layer, 'in_features': str(in_features), 'out_features': str(out_features),
        'dtype': str(dtype), 'seed': str(args.seed), 'n': str(args.n),
        'attrs': json.dumps({a: _scalar(a) for a in
                             ('rank', 'group_size', 'precision', 'wtscale', 'act_unsigned', 'wcscales', 'bias')}),
        'tensor_shapes': json.dumps({k: list(v.shape) + [str(v.dtype)] for k, v in tensors.items()}),
        'note': 'Nunchaku INT4 SVDQuant ground-truth dump for offline numeric-convention fitting.',
    }
    print('metadata:', meta['attrs'])
    print('shapes  :', meta['tensor_shapes'])

    from safetensors.torch import save_file
    # safetensors stores only string metadata; tensors must be contiguous on CPU.
    save_file(tensors, args.out, metadata=meta)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
