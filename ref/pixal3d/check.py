#!/usr/bin/env python3
"""Tiered, reproducible Pixal3D regression and CUDA quality runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "ref/pixal3d"
ESTABLISHED_GLB_SHA256 = (
    "6d8c267b006df8cf60e3c5ea71d470e89c3cb734f61b91b5e1c622e1f711a8b7")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Runner:
    def __init__(self, output: Path, dry_run: bool):
        self.output = output
        self.logs = output / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.records: list[dict] = []
        self.environment = os.environ.copy()
        self.environment["TMPDIR"] = str(ROOT / "tmp/pixal3d")

    def run(self, name: str, command: list[str], timeout: float | None = None) -> dict:
        safe_name = name.replace("/", "-")
        log = self.logs / f"{safe_name}.log"
        print(f"[{name}] {' '.join(command)}", flush=True)
        started = time.monotonic()
        if self.dry_run:
            record = {"name": name, "command": command, "status": "dry-run"}
            self.records.append(record)
            return record
        with log.open("w") as handle:
            process = subprocess.run(
                command, cwd=ROOT, env=self.environment, stdout=handle,
                stderr=subprocess.STDOUT, text=True, timeout=timeout)
        record = {"name": name, "command": command, "returncode": process.returncode,
                  "seconds": time.monotonic() - started,
                  "log": str(log.relative_to(ROOT))}
        self.records.append(record)
        if process.returncode:
            tail = log.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"{name} failed ({process.returncode}):\n{tail}")
        return record


def quick(runner: Runner) -> None:
    runner.run("cpu-build-and-unit", ["make", "-C", "cpu/pixal3d", "test", "validation"])
    runner.run("input-preparation-unit",
               [sys.executable, str(REF / "test_prepare_input.py")])
    runner.run("mesh-component-unit",
               [sys.executable, str(REF / "test_mesh_components.py")])
    runner.run("server-unit",
               [sys.executable, "-m", "unittest", "server.pixal3d.test_app"])
    runner.run("validation-record",
               [sys.executable, str(REF / "validate_results_record.py")])


def web(runner: Runner) -> None:
    runner.run("browser-integration",
               [sys.executable, str(ROOT / "server/pixal3d/test_browser.py")])


def cuda(runner: Runner, args: argparse.Namespace) -> None:
    runner.run("cuda-build", ["make", "-C", "cuda/pixal3d"])
    runner.run("cuda-kernel-reference", [
        sys.executable, str(REF / "validate.py"), "--backend", "cuda",
        "--gpu-execution", "resident", "--gpu-kernels", "auto",
        "--gpu-flow-precision", "mixed"])
    runner.run("cuda-resident", [
        sys.executable, str(REF / "validate_resident.py"), "--backend", "cuda"])
    if args.cuda_dump_dir and args.multiview_runs:
        runner.run("cuda-reliability", [
            sys.executable, str(REF / "validate_cuda_reliability.py"),
            "--dump-dir", str(args.cuda_dump_dir), "--multiview-runs",
            *(str(item) for item in args.multiview_runs),
            "--output", str(runner.output / "cuda-reliability.json")])
    else:
        runner.records.append({
            "name": "cuda-reliability", "status": "skipped",
            "reason": "pass --cuda-dump-dir and two --multiview-runs artifacts"})


def verify_assets(manifest: dict) -> dict[str, Path]:
    directory = ROOT / manifest["source"]["directory"]
    assets = {}
    for item in manifest["assets"]:
        path = directory / item["file"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"missing or changed quality asset: {path}")
        digest = sha256(path)
        if digest != item["sha256"]:
            raise RuntimeError(f"quality asset SHA-256 changed: {path}: {digest}")
        assets[item["id"]] = path
    return assets


def parse_validation(log: Path) -> dict:
    for line in log.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "triangles" in value:
            return value
    raise RuntimeError(f"GLB validation JSON missing from {log}")


def quality(runner: Runner, args: argparse.Namespace) -> None:
    manifest = json.loads((REF / "quality-corpus.json").read_text())
    assets = verify_assets(manifest)
    defaults = manifest["defaults"]
    selected = set(args.runs or ())
    runs = [item for item in manifest["native_runs"]
            if not selected or item["name"] in selected]
    unknown = selected - {item["name"] for item in manifest["native_runs"]}
    if unknown:
        raise ValueError(f"unknown quality runs: {sorted(unknown)}")
    quality_root = runner.output / "quality"
    quality_root.mkdir(parents=True, exist_ok=True)
    native_records = {}
    for item in runs:
        directory = quality_root / item["name"]
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "native.glb"
        profile = directory / "profile.json"
        record_path = directory / "result.json"
        if args.resume and output.is_file() and record_path.is_file():
            cached = json.loads(record_path.read_text())
            if cached.get("artifact", {}).get("sha256") == sha256(output):
                cached["status"] = "resumed"
                native_records[item["name"]] = cached
                runner.records.append({"name": f"quality-{item['name']}",
                                       "status": "resumed"})
                continue
        command = [
            str(ROOT / "cpu/pixal3d/pixal3d"), "--backend", "cuda",
            "--input", str(assets[item["asset"]]), "--output", str(output),
            "--fov", str(defaults["fov"]), "--mesh-scale",
            str(defaults["mesh_scale"]), "--seed", str(item["seed"]),
            "--model-dir", str(args.model_dir), "--dinov3", str(args.dinov3),
            "--naf", str(args.naf), "--gpu-execution",
            defaults["gpu_execution"], "--gpu-kernels", defaults["gpu_kernels"],
            "--gpu-flow-precision", defaults["gpu_flow_precision"],
            "--vram-budget-mib", str(args.vram_budget_mib),
            "--texture-size", str(item["texture_size"]),
            "--triangle-target", str(item["triangle_target"]),
            "--profile-json", str(profile)]
        timing = runner.run(f"quality-{item['name']}", command)
        if runner.dry_run:
            continue
        validation = runner.run(
            f"quality-{item['name']}-validate",
            [sys.executable, str(REF / "validate_glb.py"), str(output)])
        record = {
            **item, "asset_path": str(assets[item["asset"]].relative_to(ROOT)),
            "artifact": {"path": str(output.relative_to(ROOT)),
                         "bytes": output.stat().st_size, "sha256": sha256(output)},
            "seconds": timing["seconds"],
            "validation": parse_validation(ROOT / validation["log"]),
            "profile": json.loads(profile.read_text()) if profile.is_file() else {},
        }
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        native_records[item["name"]] = record

    if runner.dry_run:
        return
    thresholds = manifest["thresholds"]
    pair_records = {}
    for pair in manifest["reference_pairs"]:
        if selected and pair["native_run"] not in selected:
            continue
        native = native_records.get(pair["native_run"])
        if native is None:
            raise RuntimeError(
                f"reference pair requires native run {pair['native_run']}")
        item = next(value for value in manifest["native_runs"]
                    if value["name"] == pair["native_run"])
        directory = quality_root / f"pair-{pair['name']}"
        directory.mkdir(parents=True, exist_ok=True)
        reference = directory / "reference.glb"
        runner.run(f"quality-{pair['name']}-reference", [
            str(REF / "run_reference_cuda310.sh"), "cuda",
            str(REF / "run_reference_sv.py"), "--image", str(assets[item["asset"]]),
            "--output", str(reference), "--seed", str(item["seed"]),
            "--fov", str(defaults["fov"]), "--model_path", str(args.model_dir),
            "--low_vram", "--resolution", "1024"])
        runner.run(f"quality-{pair['name']}-reference-validate", [
            sys.executable, str(REF / "validate_glb.py"), str(reference),
            "--allow-material-red"])
        native_renders = directory / "native-renders"
        reference_renders = directory / "reference-renders"
        for label, source, destination in (
                ("native", ROOT / native["artifact"]["path"], native_renders),
                ("reference", reference, reference_renders)):
            runner.run(f"quality-{pair['name']}-{label}-renders", [
                sys.executable, str(REF / "preview_glb.py"), str(source),
                "--output-dir", str(destination)])
        comparison_path = directory / "comparison.json"
        runner.run(f"quality-{pair['name']}-compare", [
            sys.executable, str(REF / "compare_outputs.py"),
            str(ROOT / native["artifact"]["path"]), str(reference),
            "--samples", "100000", "--native-renders", str(native_renders),
            "--reference-renders", str(reference_renders),
            "--output", str(comparison_path)])
        comparison = json.loads(comparison_path.read_text())
        pair_records[pair["name"]] = {
            "native_run": pair["native_run"],
            "reference": {
                "path": str(reference.relative_to(ROOT)),
                "bytes": reference.stat().st_size,
                "sha256": sha256(reference),
            },
            "comparison": comparison,
        }
        geometry = comparison["geometry"]
        if geometry["symmetric_chamfer_rms"] >= thresholds["symmetric_chamfer_rms_max"]:
            raise RuntimeError(f"{pair['name']} Chamfer RMS exceeds threshold")
        for direction in ("native_to_reference", "reference_to_native"):
            if (geometry[direction]["normal_abs_cosine_mean"] <=
                    thresholds["normal_abs_cosine_mean_min"]):
                raise RuntimeError(f"{pair['name']} normal agreement is below threshold")
        for rendered in comparison["renders"]:
            if (rendered["rgb_psnr"] is not None and
                    rendered["rgb_psnr"] <= thresholds["render_psnr_min"]):
                raise RuntimeError(f"{pair['name']} render PSNR is below threshold")
            if (rendered["silhouette_iou"] is not None and
                    rendered["silhouette_iou"] <= thresholds["silhouette_iou_min"]):
                raise RuntimeError(f"{pair['name']} silhouette IoU is below threshold")

    summary = {
        "manifest": str((REF / "quality-corpus.json").relative_to(ROOT)),
        "native_runs": native_records,
        "reference_pairs": pair_records,
        "thresholds": thresholds,
    }
    (quality_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def exact_postprocess(runner: Runner, dump_dir: Path) -> None:
    output = runner.output / "postprocess-exact.glb"
    profile = runner.output / "postprocess-exact-profile.json"
    runner.run("postprocess-exact", [
        sys.executable, str(REF / "replay_postprocess.py"),
        "--dump-dir", str(dump_dir), "--output", str(output),
        "--profile-json", str(profile), "--texture-size", "4096",
        "--triangle-target", "1000000"])
    if not runner.dry_run:
        digest = sha256(output)
        if digest != ESTABLISHED_GLB_SHA256:
            raise RuntimeError(
                f"postprocess GLB hash changed: {digest} != {ESTABLISHED_GLB_SHA256}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("quick", "web", "cuda", "quality", "all"),
                        default="quick")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "tmp/pixal3d/checks")
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/mnt/disk2/models/Pixal3D"))
    parser.add_argument("--dinov3", type=Path,
                        default=Path("/mnt/disk2/models/dinov3-vitl16/model.safetensors"))
    parser.add_argument("--naf", type=Path,
                        default=REF / "weights/naf_release.safetensors")
    parser.add_argument("--vram-budget-mib", type=int, default=12288)
    parser.add_argument("--runs", nargs="*",
                        help="quality native run names; default is the full corpus")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cuda-dump-dir", type=Path)
    parser.add_argument("--multiview-runs", nargs=2, type=Path)
    parser.add_argument("--postprocess-dump", type=Path)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.output_dir, args.dry_run)
    started = time.monotonic()
    status = "failed"
    try:
        if args.tier in ("quick", "web", "cuda", "all"):
            quick(runner)
        if args.tier in ("web", "cuda", "all"):
            web(runner)
        if args.tier in ("cuda", "all"):
            cuda(runner, args)
        if args.tier in ("quality", "all"):
            quality(runner, args)
        if args.tier == "all":
            if args.postprocess_dump is None:
                raise ValueError("--tier all requires --postprocess-dump")
            exact_postprocess(runner, args.postprocess_dump)
        status = "pass"
    finally:
        summary = {"tier": args.tier, "status": status,
                   "seconds": time.monotonic() - started,
                   "commands": runner.records}
        temporary = args.output_dir / ".summary.json.partial"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        os.replace(temporary, args.output_dir / "summary.json")
    print(f"Pixal3D {args.tier} checks: PASS")


if __name__ == "__main__":
    main()
