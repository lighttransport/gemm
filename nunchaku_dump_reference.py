#!/usr/bin/env python3
r"""
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
import argparse, json, os
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
    ap.add_argument('--layers', default=None,
                    help='comma-separated dotted module paths; dumps each to its own file from ONE model load. '
                         'Overrides --layer. With >1 layer, --out is used as a base stem '
                         '(<stem>.<layer>.safetensors per layer).')
    ap.add_argument('--all', action='store_true',
                    help='dump EVERY SVDQW4A4Linear in the model into ONE consolidated --out file '
                         '(keys namespaced <layer>.<tensor>); one model load, one file to copy at once.')
    ap.add_argument('--probe', action='store_true',
                    help='over-determined (x,y) probe mode for offline dense effective-weight (W_eff) recovery '
                         'via least squares (see rdna4/qimg/nunchaku.md). Saves only x, y (+ y_at_zero bias ref) '
                         'per layer, N=probe_mult*in_features rows. Uses --layers (or the default 3-layer set).')
    ap.add_argument('--probe-mult', type=int, default=2,
                    help='N = probe_mult * in_features (>=2 gives a comfortably well-conditioned X)')
    ap.add_argument('--out', default='nunchaku_ref_dump.safetensors')
    ap.add_argument('--n', type=int, default=256, help='batch rows (256 == Nunchaku pad_size; avoids batch padding)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--precision', default='int4', choices=['int4', 'nvfp4'],
                    help='force the layer layout (Blackwell GPUs otherwise auto-pick nvfp4, mismatching the int4 ckpt)')
    args = ap.parse_args()

    # ---- load the Nunchaku Qwen-Image transformer (adapted to installed nunchaku v1.2.x) ----
    # In v1.2.x, from_pretrained() takes a local *.safetensors file path directly (or an HF repo id),
    # plus a `device` kwarg (default "cpu"). dtype is fixed at load (bf16) and the model cannot be
    # re-cast afterwards, so we pass device="cuda" here instead of a post-hoc .to('cuda').
    import nunchaku.models.transformers.transformer_qwenimage as _tq
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
    # from_pretrained() picks the layer layout via get_precision(): on Blackwell (sm_120/121) that auto-
    # returns "fp4" and builds NVFP4 layers (group_size=16), which then fails to load THIS int4_r128
    # checkpoint (group_size=64 -> wscales [48,*] vs the nvfp4 [192,*]). Force int4 so the constructed
    # layout matches the checkpoint we are probing. (Override may be set via --precision.)
    _tq.get_precision = lambda *a, **k: args.precision
    ckpt = args.ckpt
    if os.path.isdir(ckpt):                       # repo/dir layout: pick the requested variant file
        ckpt = os.path.join(ckpt, args.variant)
    model = NunchakuQwenImageTransformer2DModel.from_pretrained(ckpt, device="cuda")
    model = model.eval()

    # ---- over-determined (x,y) probe dump for offline W_eff recovery (rdna4/qimg/nunchaku.md) ----
    if args.probe:
        if args.layers:
            probe_paths = [s.strip() for s in args.layers.split(',') if s.strip()]
        else:                                    # default = the doc's in=3072 set (fc1 + qkv + deep fc1)
            probe_paths = ['transformer_blocks.0.img_mlp.net.0.proj',
                           'transformer_blocks.0.attn.to_qkv',
                           'transformer_blocks.30.img_mlp.net.0.proj']
        outdir = os.path.dirname(args.out) or '.'
        os.makedirs(outdir, exist_ok=True)
        for lp in probe_paths:
            probe_layer(model, lp, os.path.join(outdir, f'nunchaku_probe.{lp}.safetensors'), args)
        return

    # ---- all-layers consolidated dump (single model load, single file) ----
    if args.all:
        dump_all(model, args.out, args)
        return

    # ---- resolve the layer list and per-layer output paths (single model load for all) ----
    if args.layers:
        layer_paths = [s.strip() for s in args.layers.split(',') if s.strip()]
    else:
        layer_paths = [args.layer]
    if len(layer_paths) == 1:
        out_paths = [args.out]
    else:
        stem = args.out[:-len('.safetensors')] if args.out.endswith('.safetensors') else args.out
        out_paths = [f'{stem}.{lp}.safetensors' for lp in layer_paths]

    for lp, op in zip(layer_paths, out_paths):
        dump_layer(model, lp, op, args)


def build_layer_dump(model, layer_path, args):
    """Run the fixed per-layer dump contract for ONE SVDQW4A4Linear; return (tensors, meta)."""
    layer = _resolve(model, layer_path)
    print('\nresolved layer:', layer_path, '->', type(layer).__name__)

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
        'layer': layer_path, 'in_features': str(in_features), 'out_features': str(out_features),
        'dtype': str(dtype), 'seed': str(args.seed), 'n': str(args.n),
        'attrs': json.dumps({a: _scalar(a) for a in
                             ('rank', 'group_size', 'precision', 'wtscale', 'act_unsigned', 'wcscales', 'bias')}),
        'tensor_shapes': json.dumps({k: list(v.shape) + [str(v.dtype)] for k, v in tensors.items()}),
        'note': 'Nunchaku INT4 SVDQuant ground-truth dump for offline numeric-convention fitting.',
    }
    print('metadata:', meta['attrs'])
    return tensors, meta


def dump_layer(model, layer_path, out_path, args):
    """Capture ONE SVDQW4A4Linear to its own self-contained safetensors file."""
    tensors, meta = build_layer_dump(model, layer_path, args)
    from safetensors.torch import save_file
    # safetensors stores only string metadata; tensors must be contiguous on CPU.
    save_file(tensors, out_path, metadata=meta)
    print('wrote', out_path)


def probe_layer(model, layer_path, out_path, args):
    """Over-determined (x,y) probe for offline dense effective-weight (W_eff) recovery.

    Feeds N = probe_mult * in_features unit-Gaussian rows through the REAL fused forward and saves
    only (x, y) in bf16 plus a free bias reference y_at_zero = forward(0). Offline, augmented least
    squares [W_eff^T; b^T] = ([X|1])^+ Y recovers the logical-basis operator, sidestepping every
    CUDA-internal layout (activation swizzle, dual-scale k-tiling) and pinning the wscales
    group<->channel alignment. See rdna4/qimg/nunchaku.md."""
    layer = _resolve(model, layer_path)
    K = getattr(layer, 'in_features')
    out_features = getattr(layer, 'out_features')
    dtype = layer.proj_up.dtype
    N = args.probe_mult * K
    print(f'\nprobe {layer_path}: K={K} out={out_features} N={N} (mult={args.probe_mult}) dtype={dtype}')

    # unit-Gaussian, over-determined batch (doc recipe: global manual_seed for regenerability)
    torch.manual_seed(args.seed)
    x = torch.randn(N, K, dtype=dtype, device='cuda')          # keeps the per-token int4 quantizer in-regime
    with torch.no_grad():
        y = layer(x.reshape(1, N, K)).reshape(N, out_features)              # the real quantized forward
        y_at_zero = layer(torch.zeros(1, 1, K, dtype=dtype, device='cuda')).reshape(out_features)  # == bias

    tensors = {
        'x': x.to('cpu').contiguous(),
        'y': y.to('cpu').contiguous(),
        'y_at_zero': y_at_zero.to('cpu').contiguous(),
    }
    meta = {
        'layer': layer_path, 'N': str(N), 'K': str(K), 'out_features': str(out_features),
        'seed': str(args.seed), 'probe_mult': str(args.probe_mult), 'dtype': str(dtype),
        'note': 'Nunchaku INT4 over-determined (x,y) probe for offline dense effective-weight (W_eff) '
                'least-squares recovery; see rdna4/qimg/nunchaku.md.',
    }
    del x, y, y_at_zero                  # free GPU copies before the (large-N) finite/std check
    torch.cuda.empty_cache()
    from safetensors.torch import save_file
    save_file(tensors, out_path, metadata=meta)
    xf, yf = tensors['x'].float(), tensors['y'].float()        # check on CPU copies, not fresh GPU floats
    print(f'  x finite={torch.isfinite(xf).all().item()} y finite={torch.isfinite(yf).all().item()} '
          f"x.std={xf.std():.3f} y.std={yf.std():.3f} |y_at_zero|max={tensors['y_at_zero'].float().abs().max():.4f}")
    print('  wrote', out_path)


def find_svdquant_layers(model):
    """Discover every SVDQW4A4Linear module path, in module-declaration order."""
    return [name for name, m in model.named_modules() if type(m).__name__ == 'SVDQW4A4Linear']


def dump_all(model, out_path, args):
    """Dump EVERY SVDQW4A4Linear into ONE consolidated file (keys namespaced '<layer>.<tensor>').

    Per-layer scalar attrs/shapes are stored in the file's string metadata under 'layers' (a JSON
    map keyed by layer path). One model load, one file -> copy to the AMD box at once."""
    layer_paths = find_svdquant_layers(model)
    print(f'discovered {len(layer_paths)} SVDQW4A4Linear layers')
    all_tensors = {}
    layers_meta = {}
    for i, lp in enumerate(layer_paths):
        print(f'[{i+1}/{len(layer_paths)}]', end=' ')
        tensors, meta = build_layer_dump(model, lp, args)
        for tname, t in tensors.items():
            all_tensors[f'{lp}.{tname}'] = t          # e.g. transformer_blocks.0.attn.to_qkv.weight.qweight
        layers_meta[lp] = {'in_features': meta['in_features'], 'out_features': meta['out_features'],
                           'dtype': meta['dtype'], 'attrs': meta['attrs'], 'tensor_shapes': meta['tensor_shapes']}
    file_meta = {
        'seed': str(args.seed), 'n': str(args.n), 'precision': args.precision,
        'num_layers': str(len(layer_paths)),
        'layer_order': json.dumps(layer_paths),
        'layers': json.dumps(layers_meta),
        'key_format': '<layer_path>.<tensor>  where <tensor> in '
                      '{x,y,quantized_x,ascales,lora_act_out,weight.qweight,weight.wscales,'
                      'weight.proj_down,weight.proj_up,weight.smooth_factor,weight.smooth_factor_orig,weight.bias}',
        'note': 'Nunchaku INT4 SVDQuant ALL-LAYERS ground-truth dump for offline numeric-convention fitting.',
    }
    from safetensors.torch import save_file
    print(f'\nwriting {len(all_tensors)} tensors -> {out_path} ...')
    save_file(all_tensors, out_path, metadata=file_meta)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
