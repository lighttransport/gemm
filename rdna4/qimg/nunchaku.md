# NV-box procedure: dense effective-weight probe for the Nunchaku INT4 port

**Purpose.** Produce the one external artifact that unblocks the RDNA4 Nunchaku INT4 (SVDQuant)
Qwen-Image port — a *full-rank* set of input→output pairs from the real Nunchaku kernels, from which
we recover each linear's **dense effective weight** offline. This is the basis-invariant ground-truth
operator; it sidesteps every CUDA-internal layout (the activation swizzle and the GEMM's dual-scale
k-tiling) and pins the one remaining unknown — the wscales group↔channel *alignment* — that has the
offline decode stuck. Run on an NVIDIA box where Nunchaku actually works.

## Background (state of the port)

The on-disk format is fully reverse-engineered offline: the weight byte-swizzle is bit-exact-proven,
the low-rank factor packing is inverted with Nunchaku's own `unpack_lowrank_weight`, and the SVDQuant
numeric definitions are read from deepcompressor. What is *not* solved is how the per-group `wscales`
align to weight channels in the kernel's GEMM — its definitions are confirmed but the layout
correspondence lives inside the fused CUDA kernels (not pure-Python, not invertible from source).
Diagnosis: the reconstructed int4-main term has ~3× the norm of the true output, the signature of a
per-group-scale↔channel misalignment that no diagonal correction fixes. The earlier all-layers dump
captured `(x, y)` pairs but with only **N=256 inputs against a 3072-wide input dimension** — rank-deficient,
so the operator was unrecoverable. The fix is purely to make the probe over-determined. (Full trail in
the project memory note; companion script: `tools/nunchaku_dump_reference.py`.)

## The idea

The Nunchaku quantized linear is, as a black box, approximately a linear map `y ≈ W_eff·x + b` (the int4
activation quantization is a small, zero-mean-ish perturbation that averages out over many inputs). With a
*full-rank* batch of probe inputs, ordinary least squares recovers `W_eff` in the original/logical basis —
exactly the ground-truth operator, with no need to know the activation swizzle, the k-tiling, or even the
smoothing direction (it is all baked into the recovered operator). Offline, diffing `W_eff` against the
de-swizzled `int4×wscale + low-rank` reconstruction exposes the group/channel relabeling that is the fault.

## Procedure

Reuse the already-working `tools/nunchaku_dump_reference.py` (its model-load path is correct on your box).
Change only the probe input to be over-determined, and dump the `(x, y)` pair.

1. Resolve the target layer as before (default `transformer_blocks.0.img_mlp.net.0.proj`), read its
   `in_features = K` and `out_features`, and the module dtype (bf16).

2. Replace the fixed N=256 input with a deterministic, over-determined Gaussian batch and run the *real*
   quantized forward:

   ```python
   torch.manual_seed(0)
   K = layer.in_features
   N = 2 * K                                   # over-determined & well-conditioned; fc1 K=3072 -> N=6144
   x = torch.randn(N, K, dtype=dtype, device='cuda')        # unit Gaussian: keeps the activation quantizer in-regime
   with torch.no_grad():
       y = layer(x.reshape(1, N, K)).reshape(N, layer.out_features)
   ```

3. Save `x` and `y` (bf16) to one safetensors per probed layer. The layer's stored int4/scale/low-rank
   tensors are *not* needed again (already captured in the first dump). Per-layer size is small (fc1 ≈ 190 MB:
   6144×3072 + 6144×12288 in bf16). Include simple metadata: layer path, N, K, out_features, seed.

4. Optionally also record the forward output at `x = 0` (it equals the bias) — a free bias reference. Not
   required; the bias is otherwise recovered as the regression intercept.

5. Run it per the layer table below and copy the file(s) back to `/mnt/disk1/models/qwen-image/nunchaku/`.

## Layers to probe

All have `in_features = 3072`, so `N = 6144` covers them. Minimum viable is just the first one.

| Layer (dotted module path)                          | role                         | why probe it |
|-----------------------------------------------------|------------------------------|--------------|
| `transformer_blocks.0.img_mlp.net.0.proj`           | MLP fc1 (the canonical one)  | the layer the offline alignment fit is anchored on — **do this at minimum** |
| `transformer_blocks.0.attn.to_qkv`                  | fused QKV projection         | confirms the fused-QKV / attention-projection pattern |
| `transformer_blocks.30.img_mlp.net.0.proj`          | deep-block fc1               | checks the alignment fix generalizes across depth |
| `transformer_blocks.0.img_mlp.net.2` (optional)     | MLP fc2, `in_features=12288` | needs the larger `N≈24576`; defer until the in=3072 case is cracked |

## Sanity notes for whoever runs it

- Keep the probe input unit-Gaussian — not tiny, not huge — so the per-token activation int4 quantizer
  operates in its production regime; its noise then averages out across the N rows in the least-squares fit.
- This is deliberately black-box: it touches only the real `forward`. No access to dequantized tensors,
  internal CUDA layouts, or kernel source is required — which is exactly why it works where source-reading hit
  the CUDA wall.
- `N ≥ 2·in_features` gives a comfortably well-conditioned `X`; more rows further suppress the quantization
  residual. Determinism (fixed seed) keeps the artifact reproducible.

## What happens offline with the artifact

Recover the dense effective operator per layer by augmented least squares: `[W_effᵀ ; bᵀ] = ([X | 1])⁺ Y`.
The regression residual norm is a built-in confidence check (small ⇒ the linear approximation is good ⇒
`W_eff` is trustworthy). Then compare `W_eff` to the de-swizzled `int4×wscale + unpacked-low-rank`
reconstruction; the wscales group↔channel misalignment surfaces as an explicit relabeling pattern, which fixes
the offline dequantizer. With the numeric semantics pinned in the logical domain, Phase 1 of the port — the
INT4-resident, block-streaming-eliminating GPU path — is unblocked.

Minimal deliverable: **the fc1 layer alone at N=6144** is enough to crack the alignment; the other layers are
confirmation that the fix generalizes.

## Results from the NV-box run (2026-05-23) — and a correction to the N=2K recommendation

Ran on the Blackwell box (RTX 5060 Ti, sm_120, nunchaku 1.2.1, INT4). `X` is well-conditioned at every N
(`cond(X)=5.8`, ideal for unit-Gaussian). **But N=2K is marginal**, because the per-token int4 *activation*
quantizer makes the map `y = W_eff·quant(x) + lowrank(x) + b`: the quantization noise is an **irreducible
~20% per-row residual** that does *not* shrink with more rows. What more rows buy is a tighter `W_eff`
*estimate* (lower variance), not a lower residual. Measured on `transformer_blocks.0.img_mlp.net.0.proj`
(K=3072, out=12288), fitting `W_eff` on nested subsets vs the N=8K fit:

| N        | γ=K/N | in-sample resid | held-out pred err | ‖W(N)−W(8K)‖/‖W‖ |
|----------|-------|-----------------|-------------------|-------------------|
| 6144 (2K)| 0.50  | 0.150           | **0.300**         | **0.200**         |
| 12288(4K)| 0.25  | 0.183           | 0.245             | 0.094             |
| 18432(6K)| 0.17  | 0.193           | 0.214             | 0.052             |
| 24576(8K)| 0.125 | 0.198           | 0.198             | 0 (ref)           |

At N=2K the held-out error (0.30) is 2× the in-sample (0.15) — the overfit gap — and `W_eff` is **~20% off**
the converged operator, same order as the noise floor and only ~10× below the 3× (200%) misalignment being
hunted. Usable to *see* the relabeling, but noisy. By N≥4K `W_eff` is within ~9%; by N=6–8K within ~5%, with
in-sample and held-out both at the ~20% noise floor (no more overfit gap). **Recommendation: use N=8K
(`--probe-mult 8`).** That is what was dumped. Offline, treat the ~20% in-sample residual as the expected
floor (it's the activation-quantization noise, not a bug), and use the held-out-prediction-error−vs−in-sample
gap (not the in-sample residual alone) as the convergence check.

Artifacts (one safetensors per layer; keys `x`,`y` bf16 + `y_at_zero`=bias ref; meta has N/K/out/seed):
`nunchaku_probe.transformer_blocks.0.img_mlp.net.0.proj` (N=24576, ~755 MB),
`…0.attn.to_qkv` (N=24576), `…30.img_mlp.net.0.proj` (N=24576). NB: this box stores them under
`/mnt/disk01/models/qwen-image/nunchaku/` (the doc's `/mnt/disk1/…` is the AMD-box destination — copy across).
The optional fc2 (`…0.img_mlp.net.2`, in=12288) would want N=8·12288=98304 (~4.8 GB); still deferred.
