#!/usr/bin/env python3
"""Serial FP8xFP8 quality sweeps for rdna4/qimg.

Runs one `test_hip_qimg` process per candidate selector so ROCm memory state
does not get polluted by concurrent 20 GB checkpoint loads.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
QIMG = Path(__file__).resolve().parents[1]
DIFFUSION_REF = Path("/mnt/disk1/work/gemm/diffusion/ref/qwen_image")

DEFAULT_LABELS = [
    "img_in",
    "img_q",
    "img_k",
    "img_v",
    "img_attn_out",
    "img_mlp_fc1",
    "img_mlp_fc2",
]

FINAL_RE = re.compile(
    r"Final packed latent: .*?cos=([0-9.eE+-]+).*?psnr_peak=([0-9.eE+-]+) dB"
)
SUMMARY_RE = re.compile(r"hip_qimg: GEMM path summary:(.*)")
DEN_RE = re.compile(r"Denoising done in ([0-9.eE+-]+)s")


def existing_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_summary(text: str) -> dict[str, int]:
    match = SUMMARY_RE.search(text)
    if not match:
        return {}
    out: dict[str, int] = {}
    for item in match.group(1).strip().split():
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            pass
    return out


def run_case(args: argparse.Namespace, label: str | None) -> dict[str, object]:
    cmd = [
        str(args.exe),
        "--generate",
        "--dit",
        str(args.dit),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--steps",
        str(args.steps),
        "--init-bin",
        str(args.init_bin),
        "--txt-bin",
        str(args.txt_bin),
        "--sigmas-bin",
        str(args.sigmas_bin),
        "--ref-final",
        str(args.ref_final),
        "--path-stats",
        "--fp8-quality-target-db",
        str(args.target_db),
        "-o",
        str(args.output_dir / f"qimg_{label or 'none'}.ppm"),
    ]
    if label:
        cmd += ["--fast", "fp8_matrix_mult"]
        cmd += ["--fp8-fp8-allow", label]
    if args.block_min is not None:
        if "--fast" not in cmd:
            cmd += ["--fast", "fp8_matrix_mult"]
        cmd += ["--fp8-fp8-block-min", str(args.block_min)]
    if args.block_max is not None:
        if "--fast" not in cmd:
            cmd += ["--fast", "fp8_matrix_mult"]
        cmd += ["--fp8-fp8-block-max", str(args.block_max)]

    env = os.environ.copy()
    if args.extra_env:
        for item in args.extra_env:
            k, _, v = item.partition("=")
            if not k or not _:
                raise ValueError(f"bad --env item {item!r}; expected KEY=VALUE")
            env[k] = v

    proc = subprocess.run(
        cmd,
        cwd=args.cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
    )
    text = proc.stdout
    (args.output_dir / f"{label or 'none'}.log").write_text(text)

    final = FINAL_RE.search(text)
    den = DEN_RE.search(text)
    summary = parse_summary(text)
    psnr = float(final.group(2)) if final else None
    cos = float(final.group(1)) if final else None
    denoise_s = float(den.group(1)) if den else None
    passed = proc.returncode == 0 and psnr is not None and psnr >= args.target_db
    return {
        "reference_mode": args.reference_mode,
        "label": label or "",
        "returncode": proc.returncode,
        "passed": passed,
        "psnr_peak_db": psnr,
        "cosine": cos,
        "denoise_s": denoise_s,
        "fp8xfp8_wmma": summary.get("fp8xfp8_wmma", 0),
        "bf16xfp8_wmma": summary.get("bf16xfp8_wmma", 0),
        "fp8_scalar": summary.get("fp8_scalar", 0),
        "log": str(args.output_dir / f"{label or 'none'}.log"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=QIMG / "test_hip_qimg")
    ap.add_argument(
        "--dit",
        type=Path,
        default=Path("/mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"),
    )
    ap.add_argument("--init-bin", type=Path, default=existing_path(ROOT / "ref/qwen_image/init_latent_256.bin", DIFFUSION_REF / "init_latent_256.bin"))
    ap.add_argument("--txt-bin", type=Path, default=existing_path(ROOT / "ref/qwen_image/apple_text_256.bin", DIFFUSION_REF / "apple_text_256.bin"))
    ap.add_argument("--sigmas-bin", type=Path, default=existing_path(ROOT / "ref/qwen_image/sigmas_256.bin", DIFFUSION_REF / "sigmas_256.bin"))
    ap.add_argument("--ref-final", type=Path, default=existing_path(ROOT / "ref/qwen_image/final_latent_packed_256.bin", DIFFUSION_REF / "final_latent_packed_256.bin"))
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="comma-separated allow labels to test")
    ap.add_argument("--include-none", action="store_true", help="also test no allowlist baseline")
    ap.add_argument(
        "--reference-mode",
        default="qimg-quality-gated",
        choices=["comfy-default", "comfy-fast-fp8", "qimg-quality-gated"],
        help="reference execution mode label written to summary outputs",
    )
    ap.add_argument("--target-db", type=float, default=50.0)
    ap.add_argument("--block-min", type=int)
    ap.add_argument("--block-max", type=int)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--output-dir", type=Path, default=Path("/tmp/qimg_fp8fp8_sweep"))
    ap.add_argument("--cwd", type=Path, default=ROOT)
    ap.add_argument("--env", dest="extra_env", action="append", help="extra environment KEY=VALUE")
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels: list[str | None] = [s for s in args.labels.split(",") if s]
    if args.include_none:
        labels.insert(0, None)

    results = []
    for label in labels:
        print(f"== sweep {label or '<no allowlist>'} ==", flush=True)
        try:
            row = run_case(args, label)
        except subprocess.TimeoutExpired as exc:
            row = {
                "reference_mode": args.reference_mode,
                "label": label or "",
                "returncode": "timeout",
                "passed": False,
                "psnr_peak_db": None,
                "cosine": None,
                "denoise_s": None,
                "fp8xfp8_wmma": 0,
                "bf16xfp8_wmma": 0,
                "fp8_scalar": 0,
                "log": str(args.output_dir / f"{label or 'none'}.log"),
            }
            Path(row["log"]).write_text(exc.stdout or "")
        print(row, flush=True)
        results.append(row)

    json_path = args.output_dir / "summary.json"
    csv_path = args.output_dir / "summary.csv"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")

    return 1 if any(not r["passed"] for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
