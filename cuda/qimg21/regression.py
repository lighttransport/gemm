#!/usr/bin/env python3
"""Run deterministic Qwen-Image 2.1 reference/runner parity cases."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Case:
    height: int
    width: int
    steps: int
    seed: int

    @property
    def name(self) -> str:
        return f"{self.height}x{self.width}-s{self.steps}-seed{self.seed}"


DEFAULT_CASES = (
    Case(256, 256, 2, 42),
    Case(256, 512, 2, 123),
    Case(512, 512, 4, 42),
)
FULL_CASE = Case(1024, 1024, 40, 42)


def _load_timesteps(reference: Path, steps: int) -> list[float]:
    """Require captured model inputs, never substitute unrounded sigmas."""
    expected = [f"timestep_{step:03d}.npy" for step in range(steps)]
    if sorted(path.name for path in reference.glob("timestep_*.npy")) != expected:
        raise ValueError("reference timestep fixture set does not match requested steps")
    values = []
    for name in expected:
        value = np.load(reference / name, allow_pickle=False)
        if value.size != 1 or not np.isfinite(value).all() or not 0 <= float(value.reshape(-1)[0]) <= 1:
            raise ValueError(f"invalid captured model timestep: {name}")
        values.append(float(value.reshape(-1)[0]))
    return values


def _parse_case(value: str) -> Case:
    try:
        geometry, steps_text, seed_text = value.split(":", 2)
        height_text, width_text = geometry.lower().split("x", 1)
        case = Case(int(height_text), int(width_text), int(steps_text), int(seed_text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid case {value!r}; expected HEIGHTxWIDTH:STEPS:SEED"
        ) from exc
    if min(case.height, case.width, case.steps) <= 0:
        raise argparse.ArgumentTypeError("case dimensions and steps must be positive")
    if case.height % 32 or case.width % 32:
        raise argparse.ArgumentTypeError("case dimensions must be divisible by 32")
    return case


def _run(command: list[str], root: Path) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--image", help="optional condition image for Python editing cases")
    ap.add_argument("--negative-prompt")
    ap.add_argument("--true-cfg-scale", type=float, default=1.0)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--reference-sdpa-backend", choices=("default", "efficient"), default="default")
    ap.add_argument(
        "--case",
        dest="cases",
        action="append",
        type=_parse_case,
        help="repeatable HEIGHTxWIDTH:STEPS:SEED override (default: three smoke cases)",
    )
    ap.add_argument(
        "--include-full",
        action="store_true",
        help="also run the 1024x1024/40-step acceptance case",
    )
    ap.add_argument("--work-dir", default="tmp/qimg21-regression")
    ap.add_argument(
        "--native",
        action="store_true",
        help="run the native C/NVRTC denoiser against the PyTorch reference checkpoints",
    )
    ap.add_argument("--native-bin", default="cuda/qimg21/test_cuda_qimg21_native")
    ap.add_argument("--native-attention", choices=("math", "reverse64", "mma64", "mma64-flash", "mma64-mixed", "mma64-forward-flash", "mma128-efficient", "cutlass-efficient"), default="math")
    ap.add_argument("--native-normalization", choices=("default", "vector4"), default="default")
    ap.add_argument("--native-rope", choices=("default", "host-table", "host-table-vector4", "host-table-exact"), default="default")
    ap.add_argument("--quantized", action="store_true")
    ap.add_argument("--quantized-transformer", type=Path, help="Optional native row-INT8 package")
    ap.add_argument("--quantize-on-load", choices=("int8-row",), help="Quantize each matrix without a disk export")
    ap.add_argument("--int8-tensor-core", action="store_true",
                    help="dynamic W8A8 custom tensor-core execution (requires package)")
    ap.add_argument("--cosine-threshold", type=float)
    args = ap.parse_args()
    if args.native_attention != "math" and not args.native:
        ap.error("--native-attention requires --native")
    if not args.native and (args.native_normalization != "default" or args.native_rope != "default"):
        ap.error("native normalization/rope options require --native")
    if args.quantized_transformer and args.quantize_on_load:
        ap.error("choose a package or quantize-on-load, not both")
    if args.quantized_transformer or args.quantize_on_load:
        if not args.native:
            ap.error("quantized weights require --native")
        args.quantized = True
    elif args.quantized:
        ap.error("--quantized requires --quantized-transformer or --quantize-on-load")
    if args.int8_tensor_core and not args.quantized_transformer:
        ap.error("--int8-tensor-core requires --quantized-transformer")

    root = Path(__file__).resolve().parents[2]
    model = Path(args.model).resolve()
    if not model.is_dir():
        raise SystemExit(f"model directory does not exist: {model}")
    if args.native and args.image:
        raise SystemExit("--native image conditioning is not implemented yet")
    if args.native and ((args.negative_prompt is not None) != (args.true_cfg_scale > 1.0)):
        raise SystemExit("native true CFG requires --negative-prompt and --true-cfg-scale > 1 together")
    if args.native and args.dtype != "bf16":
        raise SystemExit("the native runner currently implements BF16 activation boundaries only")
    work = Path(args.work_dir)
    if not work.is_absolute():
        work = root / work
    cases = list(args.cases or DEFAULT_CASES)
    if args.include_full:
        cases.append(FULL_CASE)

    reference = root / "cuda/qimg21/reference.py"
    runner = root / "cuda/qimg21/test_cuda_qimg21.py"
    compare = root / "cuda/qimg21/compare.py"
    native_bin = Path(args.native_bin)
    if not native_bin.is_absolute():
        native_bin = root / native_bin
    if args.native and not native_bin.exists():
        raise SystemExit(f"native executable not found: {native_bin}; run `make -C cuda/qimg21 native` first")
    failed: list[str] = []
    for case in cases:
        case_dir = work / case.name
        ref_dir = case_dir / "reference"
        run_dir = case_dir / ("native" if args.native else "runner")
        out_path = run_dir / "runner.png"
        common = [
            "--model",
            str(model),
            "--prompt",
            args.prompt,
            "--height",
            str(case.height),
            "--width",
            str(case.width),
            "--steps",
            str(case.steps),
            "--seed",
            str(case.seed),
        ]
        if args.image:
            common.extend(["--image", str(Path(args.image).resolve())])
        if args.negative_prompt is not None:
            common.extend(["--negative-prompt", args.negative_prompt])
        if args.true_cfg_scale != 1.0:
            common.extend(["--true-cfg-scale", str(args.true_cfg_scale)])
        print(f"\n=== {case.name} ===", flush=True)
        try:
            reference_command = [
                sys.executable,
                str(reference),
                *common,
                "--dtype",
                args.dtype,
                "--sdpa-backend",
                args.reference_sdpa_backend,
                "--dump-initial-latents",
            ]
            if args.native:
                reference_command.extend(["--dump-pred-dir", str(ref_dir)])
            reference_command.extend(["--dump-dir", str(ref_dir)])
            _run(reference_command, root)
            if args.native:
                run_dir.mkdir(parents=True, exist_ok=True)
                guidance = ["--attention", args.native_attention, "--normalization", args.native_normalization,
                            "--rope", args.native_rope]
                if args.negative_prompt is not None:
                    guidance.extend(["--negative-prompt-embeds", str(ref_dir / "negative_prompt_embeds.npy"),
                                     "--guidance-scale", str(args.true_cfg_scale)])
                if args.quantized_transformer:
                    guidance.extend(["--quantized-transformer", str(args.quantized_transformer.resolve())])
                if args.int8_tensor_core:
                    guidance.append("--int8-tensor-core")
                if args.quantize_on_load:
                    guidance.extend(["--quantize-on-load", args.quantize_on_load])
                timesteps = _load_timesteps(ref_dir, case.steps)
                (run_dir / "native_config.json").write_text(json.dumps({
                    "attention": args.native_attention, "model": str(model),
                    "normalization": args.native_normalization, "rope": args.native_rope,
                    "quantized_transformer": str(args.quantized_transformer.resolve()) if args.quantized_transformer else None,
                    "quantize_on_load": args.quantize_on_load,
                    "int8_tensor_core": args.int8_tensor_core,
                    "dtype": args.dtype, "timesteps": timesteps,
                    "reference_sdpa_backend": args.reference_sdpa_backend,
                    "case": case.name, "prompt": args.prompt,
                    "negative_prompt": args.negative_prompt, "true_cfg_scale": args.true_cfg_scale,
                }, indent=2) + "\n")
                for step, sigma in enumerate(timesteps):
                    step_pred_dir = run_dir / f"pred-step-{step:03d}"
                    _run(
                        [
                            str(native_bin),
                            *guidance,
                            "--model",
                            str(model),
                            "--prompt-embeds",
                            str(ref_dir / "prompt_embeds.npy"),
                            "--latents",
                            str(ref_dir / f"input_{step:03d}.npy"),
                            "--height-tokens",
                            str(case.height // 16),
                            "--width-tokens",
                            str(case.width // 16),
                            "--steps",
                            "1",
                            "--timestep",
                            f"{sigma:.10f}",
                            "--pred-dir",
                            str(step_pred_dir),
                            "--out",
                            str(run_dir / f"native_step_{step:03d}.npy"),
                        ],
                        root,
                    )
                    shutil.copyfile(
                        step_pred_dir / "pred_000.npy", run_dir / f"pred_{step:03d}.npy"
                    )
                # Free-running Euler trajectory: only the initial latent is
                # shared; every later state comes from the native runner.
                trajectory_dir = run_dir / "trajectory"
                trajectory_dir.mkdir(parents=True, exist_ok=True)
                _run([
                    str(native_bin), *guidance, "--model", str(model),
                    "--prompt-embeds", str(ref_dir / "prompt_embeds.npy"),
                    "--latents", str(ref_dir / "initial_latents.npy"),
                    "--height-tokens", str(case.height // 16),
                    "--width-tokens", str(case.width // 16),
                    "--steps", str(case.steps), "--dump-dir", str(trajectory_dir),
                    "--out", str(trajectory_dir / "final_latents.npy"),
                ], root)
            else:
                _run(
                    [
                        sys.executable,
                        str(runner),
                        "--generate",
                        *common,
                        "--dtype",
                        args.dtype,
                        "--dump-initial-latents",
                        "--dump-dir",
                        str(run_dir),
                        "--out",
                        str(out_path),
                    ],
                    root,
                )
            compare_command = [sys.executable, str(compare), "--reference-dir", str(ref_dir), "--runner-dir", str(run_dir)]
            if args.native:
                compare_command.append("--denoiser-only")
            if args.quantized:
                compare_command.append("--quantized")
            if args.int8_tensor_core:
                compare_command.extend(["--mre-threshold", "0.25"])
            if args.cosine_threshold is not None:
                compare_command.extend(["--cosine-threshold", str(args.cosine_threshold)])
            commands = [compare_command]
            if args.native:
                trajectory_compare = [sys.executable, str(compare), "--steps-only",
                                      "--reference-dir", str(ref_dir),
                                      "--runner-dir", str(trajectory_dir)]
                if args.quantized:
                    trajectory_compare.append("--quantized")
                if args.int8_tensor_core:
                    trajectory_compare.extend(["--mre-threshold", "0.25"])
                if args.cosine_threshold is not None:
                    trajectory_compare.extend(["--cosine-threshold", str(args.cosine_threshold)])
                commands.append(trajectory_compare)
            comparison_failed = False
            for command in commands:
                try:
                    _run(command, root)
                except subprocess.CalledProcessError:
                    comparison_failed = True
            if comparison_failed:
                failed.append(case.name)
                print(f"FAIL: {case.name}", file=sys.stderr)
        except (subprocess.CalledProcessError, ValueError, OSError) as exc:
            failed.append(case.name)
            print(f"FAIL: {case.name}: {exc}", file=sys.stderr)

    if failed:
        print("\nREGRESSION FAIL: " + ", ".join(failed), file=sys.stderr)
        return 1
    print(f"\nREGRESSION PASS: {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
