#!/usr/bin/env python3
"""Run deterministic Qwen-Image 2.1 reference/runner parity cases."""

from __future__ import annotations

import argparse
import math
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


def _flow_sigmas(steps: int, image_tokens: int) -> list[float]:
    """Mirror FlowMatchEulerDiscreteScheduler's dynamic-shift schedule."""
    base_seq, max_seq = 256.0, 8192.0
    base_shift, max_shift = 0.5, 0.9
    mu = image_tokens * (max_shift - base_shift) / (max_seq - base_seq)
    mu += base_shift - (max_shift - base_shift) / (max_seq - base_seq) * base_seq
    emu = math.exp(mu)
    sigmas = []
    for i in range(steps):
        u = 1.0 - i / steps
        sigmas.append(emu / (emu + (1.0 / u - 1.0)))
    if steps == 1:
        return [1.0]
    scale = (1.0 - sigmas[-1]) / (1.0 - 0.02)
    return [1.0 - (1.0 - sigma) / scale for sigma in sigmas]


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
    ap.add_argument("--quantized", action="store_true")
    ap.add_argument("--cosine-threshold", type=float)
    args = ap.parse_args()

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
                "--dump-initial-latents",
            ]
            if args.native:
                reference_command.extend(["--dump-pred-dir", str(ref_dir)])
            reference_command.extend(["--dump-dir", str(ref_dir)])
            _run(reference_command, root)
            if args.native:
                run_dir.mkdir(parents=True, exist_ok=True)
                guidance = []
                if args.negative_prompt is not None:
                    guidance = ["--negative-prompt-embeds", str(ref_dir / "negative_prompt_embeds.npy"),
                                "--guidance-scale", str(args.true_cfg_scale)]
                fallback_sigmas = _flow_sigmas(case.steps, (case.height // 16) * (case.width // 16))
                timesteps = []
                for step, fallback in enumerate(fallback_sigmas):
                    timestep_path = ref_dir / f"timestep_{step:03d}.npy"
                    if timestep_path.exists():
                        timesteps.append(float(np.load(timestep_path).reshape(-1)[0]))
                    else:
                        timesteps.append(fallback)
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
            if args.cosine_threshold is not None:
                compare_command.extend(["--cosine-threshold", str(args.cosine_threshold)])
            commands = [compare_command]
            if args.native:
                trajectory_compare = [sys.executable, str(compare), "--steps-only",
                                      "--reference-dir", str(ref_dir),
                                      "--runner-dir", str(trajectory_dir)]
                if args.quantized:
                    trajectory_compare.append("--quantized")
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
        except subprocess.CalledProcessError:
            failed.append(case.name)
            print(f"FAIL: {case.name}", file=sys.stderr)

    if failed:
        print("\nREGRESSION FAIL: " + ", ".join(failed), file=sys.stderr)
        return 1
    print(f"\nREGRESSION PASS: {len(cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
