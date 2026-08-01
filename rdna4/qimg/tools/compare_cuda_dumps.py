#!/usr/bin/env python3
"""Layer-by-layer comparison of the qimg HIP runner against the CUDA fp8 reference dumps.

The CUDA reference (cuda/qimg) dumps a full set of per-step (DiT) and per-stage (VAE)
intermediate tensors at 256x256, fp8, for the prompt "a red apple on a white table".
This tool feeds the *same* inputs into the HIP runner and compares each stage, reporting
cosine / PSNR / max-abs / mean-abs and the first point of divergence so the wobble can be
localized to a specific denoise step or VAE stage.

Dump locations (default --cuda-dir): /mnt/disk1/models/qwen-image/dumps
  cuda_perstep_NN.bin   raw f32 [16,32,32], latent BEFORE Euler step N (NN=00..19). _00 = init noise.
  cuda_latent_prenorm.npy  (1,16,1,32,32) final latent BEFORE Wan21 denorm  (== HIP DiT output)
  cuda_latent.bin       raw f32 [16,32,32] final latent AFTER denorm (== VAE input)
  cuda_vae_<stage>.npy  per-stage VAE activations [C,H,W]

HIP side:
  DiT:  test_hip_qimg --generate ... --dump-steps-prefix <P> --dump-final <F>
        -> <P>_stepNNN.bin  raw f32 [n_img=256, in_ch=64] PACKED, latent AFTER step NNN.
  VAE:  QIMG_VAE_DUMP_PREFIX=<V> test_hip_qimg --test-vae --latent-bin cuda_latent.bin ...
        -> <V>_<stage>.bin  raw f32 [C,H,W].

Layout/timing reconciliation (verified against test_hip_qimg.c):
  * Patchify: token tok = py*16+px, inner 64 = [c=16][dy=2][dx=2] -> latent[c, py*2+dy, px*2+dx].
  * Off-by-one: HIP _step{N} (after step N) == CUDA cuda_perstep_{N+1} (before step N+1).
                HIP final == cuda_latent_prenorm.
"""
import argparse
import os
import sys
import numpy as np

LAT_C, LAT_H, LAT_W = 16, 32, 32
PS = 2
WP = LAT_W // PS   # 16 patches across
HP = LAT_H // PS   # 16 patches down
N_IMG = HP * WP    # 256
IN_CH = LAT_C * PS * PS  # 64

DEFAULT_CUDA_DIR = "/mnt/disk1/models/qwen-image/dumps"


# ---------- IO ----------

def load_bin_f32(path, shape=None):
    a = np.fromfile(path, dtype="<f4")
    if shape is not None:
        a = a.reshape(shape)
    return a


def load_npy(path):
    a = np.load(path)
    return np.ascontiguousarray(a, dtype=np.float64).astype(np.float32)


# ---------- layout ----------

def unpatchify(packed):
    """[256,64] packed -> [16,32,32] unpacked latent (inverse of test_hip_qimg.c:649-657)."""
    p = packed.reshape(HP, WP, LAT_C, PS, PS)        # [py, px, c, dy, dx]
    lat = p.transpose(2, 0, 3, 1, 4)                 # [c, py, dy, px, dx]
    return np.ascontiguousarray(lat).reshape(LAT_C, LAT_H, LAT_W)


def patchify(lat):
    """[16,32,32] -> [256,64] packed (forward of test_hip_qimg.c:649-657)."""
    l = lat.reshape(LAT_C, HP, PS, WP, PS)           # [c, py, dy, px, dx]
    p = l.transpose(1, 3, 0, 2, 4)                   # [py, px, c, dy, dx]
    return np.ascontiguousarray(p).reshape(N_IMG, IN_CH)


# ---------- metrics ----------

def metrics(ref, got):
    ref = ref.astype(np.float64).ravel()
    got = got.astype(np.float64).ravel()
    n = ref.size
    n_nan = int(np.count_nonzero(~np.isfinite(got)))
    if n_nan:
        # keep metrics meaningful: count NaNs separately, compute on finite mask
        mask = np.isfinite(got) & np.isfinite(ref)
        ref_m, got_m = ref[mask], got[mask]
    else:
        ref_m, got_m = ref, got
    diff = np.abs(ref_m - got_m)
    max_abs = float(diff.max()) if diff.size else float("nan")
    mean_abs = float(diff.mean()) if diff.size else float("nan")
    denom = (np.linalg.norm(ref_m) * np.linalg.norm(got_m)) + 1e-30
    cosine = float(np.dot(ref_m, got_m) / denom) if diff.size else float("nan")
    peak = float(np.abs(ref_m).max()) if diff.size else 0.0
    mse = float((diff ** 2).mean()) if diff.size else float("nan")
    if mse > 0 and peak > 0:
        psnr = 20.0 * np.log10(peak) - 10.0 * np.log10(mse)
    elif mse == 0:
        psnr = float("inf")
    else:
        psnr = float("nan")
    return dict(n=n, n_nan=n_nan, cosine=cosine, psnr=psnr,
                max_abs=max_abs, mean_abs=mean_abs)


def fmt_row(label, m, flag=""):
    psnr = m["psnr"]
    psnr_s = "  inf  " if psnr == float("inf") else f"{psnr:7.2f}"
    nan_s = f" NaN={m['n_nan']}" if m["n_nan"] else ""
    return (f"  {label:<16} cos={m['cosine']:.6f}  psnr={psnr_s} dB  "
            f"max={m['max_abs']:.4e}  mean={m['mean_abs']:.4e}{nan_s}  {flag}")


# ---------- subcommands ----------

def cmd_prep_init(args):
    src = os.path.join(args.cuda_dir, "cuda_perstep_00.bin")
    lat = load_bin_f32(src, (LAT_C, LAT_H, LAT_W))
    packed = patchify(lat)
    packed.astype("<f4").tofile(args.out)
    print(f"wrote {args.out}  [{N_IMG},{IN_CH}] from {src}  "
          f"(min={lat.min():.3f} max={lat.max():.3f} std={lat.std():.3f})")


def cmd_dit(args):
    cos_thresh = args.cos_thresh
    print(f"DiT per-step comparison  (HIP _step{{N}} vs CUDA cuda_perstep_{{N+1}})")
    print(f"  cuda_dir={args.cuda_dir}  hip_prefix={args.hip_prefix}")
    first_bad = None
    rows = []
    for n in range(0, args.steps - 1):  # HIP _step0.._step18 -> cuda_perstep_01..19
        hip_path = f"{args.hip_prefix}_step{n:03d}.bin"
        cuda_path = os.path.join(args.cuda_dir, f"cuda_perstep_{n+1:02d}.bin")
        if not os.path.exists(hip_path):
            print(f"  [missing] {hip_path}"); break
        if not os.path.exists(cuda_path):
            print(f"  [missing] {cuda_path}"); continue
        hip = unpatchify(load_bin_f32(hip_path, (N_IMG, IN_CH)))
        cuda = load_bin_f32(cuda_path, (LAT_C, LAT_H, LAT_W))
        m = metrics(cuda, hip)
        flag = ""
        if m["n_nan"] or m["cosine"] < cos_thresh:
            flag = "<-- DIVERGENCE"
            if first_bad is None:
                first_bad = n
        rows.append(fmt_row(f"step{n:02d}->{n+1:02d}", m, flag))

    # final latent (HIP dump-final, packed) vs cuda_latent_prenorm.npy
    if args.hip_final and os.path.exists(args.hip_final):
        cuda_prenorm = os.path.join(args.cuda_dir, "cuda_latent_prenorm.npy")
        if os.path.exists(cuda_prenorm):
            hip = unpatchify(load_bin_f32(args.hip_final, (N_IMG, IN_CH)))
            cuda = load_npy(cuda_prenorm).reshape(LAT_C, LAT_H, LAT_W)
            m = metrics(cuda, hip)
            flag = "<-- DIVERGENCE" if (m["n_nan"] or m["cosine"] < cos_thresh) else ""
            rows.append(fmt_row("final(prenorm)", m, flag))

    print("\n".join(rows))
    if first_bad is not None:
        print(f"\nFIRST DIVERGENCE at HIP step {first_bad} "
              f"(cosine < {cos_thresh}); inspect that step's DiT blocks / FP8 path.")
    else:
        print(f"\nAll steps cosine >= {cos_thresh}: DiT trajectory matches CUDA.")


# VAE stage names in pipeline order, matching cuda_vae_<name>.npy and hip_vae_<name>.bin
VAE_STAGES = (["post_quant", "conv1", "middle_0", "middle_1", "middle_2"]
              + [f"upsample_{i}" for i in range(15)])


def cmd_vae(args):
    cos_thresh = args.cos_thresh
    print(f"VAE per-stage comparison")
    print(f"  cuda_dir={args.cuda_dir}  hip_prefix={args.hip_prefix}")
    first_bad = None
    rows = []
    for name in VAE_STAGES:
        hip_path = f"{args.hip_prefix}_{name}.bin"
        cuda_path = os.path.join(args.cuda_dir, f"cuda_vae_{name}.npy")
        if not os.path.exists(cuda_path):
            continue
        if not os.path.exists(hip_path):
            print(f"  [missing] {hip_path}")
            continue
        cuda = load_npy(cuda_path)
        hip = load_bin_f32(hip_path)
        if hip.size != cuda.size:
            rows.append(f"  {name:<16} SIZE MISMATCH hip={hip.size} cuda={cuda.size}")
            if first_bad is None:
                first_bad = name
            continue
        hip = hip.reshape(cuda.shape)
        m = metrics(cuda, hip)
        flag = ""
        if m["n_nan"] or m["cosine"] < cos_thresh:
            flag = "<-- DIVERGENCE"
            if first_bad is None:
                first_bad = name
        rows.append(fmt_row(name, m, flag) + f"  shape={tuple(cuda.shape)}")
    print("\n".join(rows))
    if first_bad is not None:
        print(f"\nFIRST DIVERGENCE at VAE stage '{first_bad}' (cosine < {cos_thresh}).")
    else:
        print(f"\nAll VAE stages cosine >= {cos_thresh}: VAE decode matches CUDA.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep-init", help="patchify cuda_perstep_00 -> packed --init-bin")
    p.add_argument("--cuda-dir", default=DEFAULT_CUDA_DIR)
    p.add_argument("--out", default="init_cuda_packed.bin")
    p.set_defaults(fn=cmd_prep_init)

    p = sub.add_parser("dit", help="compare HIP per-step latents vs CUDA perstep dumps")
    p.add_argument("--cuda-dir", default=DEFAULT_CUDA_DIR)
    p.add_argument("--hip-prefix", required=True, help="e.g. hip_step (for hip_step_stepNNN.bin)")
    p.add_argument("--hip-final", default=None, help="HIP --dump-final output (packed)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cos-thresh", type=float, default=0.999)
    p.set_defaults(fn=cmd_dit)

    p = sub.add_parser("vae", help="compare HIP per-stage VAE dumps vs CUDA vae dumps")
    p.add_argument("--cuda-dir", default=DEFAULT_CUDA_DIR)
    p.add_argument("--hip-prefix", required=True, help="e.g. hip_vae (for hip_vae_<stage>.bin)")
    p.add_argument("--cos-thresh", type=float, default=0.999)
    p.set_defaults(fn=cmd_vae)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
