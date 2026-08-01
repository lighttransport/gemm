# rdna4/flux2 — FLUX.2 Klein 4B on RDNA4 / RX 9070 XT (gfx1201)

HIP/ROCm runner for the FLUX.2 Klein 4B image-generation transformer. Implements all four
quantization paths shipped on top of the same `flux2_hip_wt` struct, with one set of
kernels per format and a uniform dispatcher (`op_gemm_wt` → int4/int8/fp8/f32).

## Quality / speed snapshot

Image PSNR vs bf16 reference @256, 4-step Klein render (seed 42, "a red apple on a white
table"). DiT-step time on RX 9070 XT (gfx1201). All numbers reproducible from the commit
log; see `cmp_phase_final_256.png` for the visual montage.

| path | model | PSNR | cos | DiT/step | on-disk |
|---|---|---|---|---|---|
| bf16 (ref) | flux-2-klein-4b.safetensors | — | — | 0.61 s | 7.75 GB |
| fp8 (orig, ModelOpt) | flux-2-klein-4b-fp8.safetensors | 37.15 dB | 0.99983 | 0.05 s | 4.07 GB |
| fp8 + Phase B K=2 | (same model, env opt-in) | 38.50 dB | 0.99988 | 0.06 s | 4.07 GB |
| fp8 + Phase B K=5 + smooth | flux-2-klein-4b-fp8-smooth.safetensors + env | 40.27 dB | 0.99992 | 0.08 s | 4.07 GB |
| int8 (orig, ModelOpt) | flux-2-klein-4b-int8.safetensors | 38.58 dB | 0.99992 | 0.04 s | 4.07 GB |
| **int8 + smooth (recommended)** | **flux-2-klein-4b-int8-smooth.safetensors** | **42.73 dB** | **0.99996** | **0.04 s** | **4.08 GB** |
| int4 SVDQ (r=32 g16) | flux-2-klein-4b-int4-svdq-r32-g16.safetensors | 32.59 dB | 0.99951 | 0.25 s | 2.78 GB |
| int4 RTN g16 | flux-2-klein-4b-int4-g16.safetensors | 30.13 dB | 0.99914 | 0.19 s | 2.69 GB |

bf16-F32 OOMs at 512; int8/fp8/int4 all render at 1024.

## Running

```bash
make test_hip_flux2 verify_dit
./test_hip_flux2 --generate \
  --dit  <model.safetensors> \
  --vae  <vae.safetensors> \
  --enc  <text_encoder.safetensors> \
  --tok  <tokenizer.gguf> \
  --prompt "..." --steps 4 --size 512 --seed 42 \
  -o out.ppm
```

The runner **auto-detects** the quant path from the safetensors keys:
- `.qint4` keys → INT4 path (W4A16 RTN g16 or g64; SVDQuant LoRA optional via `.lora_up/down`).
- `.weight` + I8 dtype → INT8 path (W8A8 per-row).
- `.weight` + F8_E4M3 dtype + per-tensor `.weight_scale` → FP8 path (pipe32 default).
- Else: BF16/F32 dequant path.

Presence of `<base>.smooth` keys enables the SmoothQuant path on int8/fp8/int4.

## Environment variables

| env | effect | default |
|---|---|---|
| `FLUX2_FP8_GEMM=0` | Force F32 dequant (debug; slow) | 1 |
| `FLUX2_FP8_WMMA=0/1/2` | 0=LUT, 1=BF16×FP8 (~0.999 cos), 2=pipe32 FP8×FP8 (~0.985 cos) | 2 |
| `FLUX2_FP8_OPT=0` | Disable 128×128 LUT tiling | 1 |
| **`FLUX2_FP8_BF16_BLOCKS=K`** | **Phase B**: route flat block indices `[0, K)` to BF16-act×FP8-wt for quality (cost +30%/+block). Layers with `flux-2-klein-4b-fp8-smooth.safetensors`. | 0 |
| `FLUX2_INT8_DP4A=1` | Force scalar int8 dp4a fallback (debug) | 0 |
| `FLUX2_INT8_DEBUG=1` | One-shot per-linear cos(gpu,host) self-test for INT8 GEMM | 0 |
| `FLUX2_ATTN_WMMA=0` | Disable BF16 WMMA self-attention | 1 |
| `FLUX2_CALIB_DUMP=path` | Dump per-linear activation amax during a run — feeds `tools/{int8,svdq_int4,fp8}_smooth_from_bf16.py --calib`. | — |
| `FLUX2_VAE_WMMA=0` | Disable VAE WMMA path | 1 |

## Tools (`tools/`)

| tool | purpose |
|---|---|
| `rtn_int4_g16_from_bf16.py` | Plain RTN int4 g16 quantizer (no SVDQuant). Smallest VRAM. |
| `svdq_int4_g16_from_bf16.py` | SVDQuant int4 (LoRA r=32/128) + optional `--calib` smooth + optional `--group-size 64`. |
| **`int8_smooth_from_bf16.py`** | **Phase D**: bf16 → int8 + SmoothQuant lambda (lam ≥ 1). +4.15 dB PSNR free. |
| `fp8_smooth_from_bf16.py` | Phase F: bf16 → fp8 + SmoothQuant. +0.87 dB free (layers with `FLUX2_FP8_BF16_BLOCKS`). |
| `probe_hipblaslt_fp8.cpp` | Phase C probe: confirmed hipBLASLt FP8 works on ROCm 7.2.3 but is 40-60% slower than the hand-written pipe32. Negative finding documented; not wired into the runner. |

## Calibration recipe (smooth paths)

```bash
# 1. Dump per-linear activation amax during a bf16 forward (any 4-step render is enough):
FLUX2_FP8_GEMM=0 FLUX2_CALIB_DUMP=/mnt/disk1/models/klein2-4b/calib/flux2_klein_4b_calib.safetensors \
  ./test_hip_flux2 --generate --dit <bf16> --vae <vae> --enc <enc> --tok <tok> \
  --prompt "..." --steps 4 --size 256 --seed 42 -o /tmp/calib.ppm

# 2. Quantize bf16 → int8 + smooth (or fp8 / int4 svdq):
python tools/int8_smooth_from_bf16.py \
  --bf16  /mnt/disk1/models/klein2-4b/bf16/flux-2-klein-4b.safetensors \
  --calib /mnt/disk1/models/klein2-4b/calib/flux2_klein_4b_calib.safetensors \
  --out   /mnt/disk1/models/klein2-4b/int8_smooth/flux-2-klein-4b-int8-smooth.safetensors
```

The same calib dump is reused across the int8 / fp8 / int4-svdq smooth quantizers.

## Comparison montages

- `cmp_phase_final_256.png` — bf16 / fp8 / fp8+sm K=0 / fp8+sm K=5 / int8 / int8+sm.
- `cmp_int8_smooth_phaseD_256.png` — int8 vs int8+smooth (the +4.15 dB win).
- `cmp_fp8_perblock_K_256.png` — fp8 quality routing K sweep.
- `cmp_int8_fp8_bf16_256.png` — original int8 vs fp8 vs bf16 baselines.
- `cmp_int4_int8_fp8_bf16_256.png` — full 4-way including int4.
- `cmp_int4svdq_int4_int8_fp8_bf16_256.png` — int4 RTN vs SVDQ.
- `cmp_int4_g64smooth_all_256.png` — int4 g64 smooth probe (negative on klein).

See the commit log for the per-phase rationale and stop conditions.
