"""Compare a fresh H3 reference as it runs, pruning receipt-verified intermediate arrays.

The independent reference implementation is unchanged. Every array is checked
against its original receipt before pruning; the final report binds that receipt,
both array hashes, the comparison result, and the validation sources. Native
captures, reference boundary frames and final latents remain available.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ref.minimax_h3_native import captures, verify


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


def validate_receipt(name, receipt, storage_hash, shape, generation_hash, sources):
    if receipt.get("storage") not in (name + ".npy", name + ".npy.gz"):
        raise ValueError("noncanonical reference storage: " + name)
    if type(receipt.get("shared_input")) is not bool:
        raise ValueError("reference shared-input flag must be boolean: " + name)
    required = {
        "sha256": storage_hash, "generation_sha256": generation_hash,
        "upstream_revision": verify.UPSTREAM, "shared_input": name.startswith("noise_"),
        "reference_source_sha256": sources["reference_source_sha256"],
        "capture_reader_sha256": sources["capture_reader_sha256"], "shape": shape,
    }
    for key, value in required.items():
        if receipt.get(key) != value:
            raise ValueError("independent reference receipt mismatch: " + name + "/" + key)


def run(args):
    manifest, native, reference = map(Path, (args.manifest, args.native, args.reference))
    generation = json.loads(manifest.read_text())
    if generation.get("backend") != "minimax_h3_rocm_experimental" or generation.get("parity") != "unverified":
        raise ValueError("expected a complete native H3 generation")
    required = {"width": 1344, "height": 768, "frames": 124, "fps": 24,
                "sigma_grid_points": 40, "euler_updates": 39, "seed": 42,
                "video_shift": 12, "audio_shift": 3}
    if any(generation.get(key) != value for key, value in required.items()):
        raise ValueError("streaming acceptance requires the full default geometry and schedule")
    if generation.get("metrics", {}).get("memory_fit") != "pass":
        raise ValueError("measured native process VRAM must fit its budget")
    if reference.exists() or Path(args.out).exists():
        raise FileExistsError("fresh reference and report paths are required")
    generation_hash = verify.digest(manifest)
    source_paths = {"reference_source_sha256": Path(__file__).with_name("reference.py"),
                    "capture_reader_sha256": Path(captures.__file__),
                    "comparison_source_sha256": Path(verify.__file__),
                    "streaming_source_sha256": Path(__file__)}
    sources = {key: verify.digest(path) for key, path in source_paths.items()}
    names = ["qwen_hidden", "refined_text", "noise_video", "noise_audio"]
    for step in range(39):
        names += [f"latent_video_{step:03d}", f"latent_audio_{step:03d}"]
    names += [f"frame_{index:03d}" for index in range(124)]
    keep = {"qwen_hidden", "refined_text", "noise_video", "noise_audio",
            "latent_video_038", "latent_audio_038", "frame_000", "frame_123"}
    command = [sys.executable, str(source_paths["reference_source_sha256"]),
               "--manifest", str(manifest), "--native", str(native),
               "--qwen-reference", args.qwen_reference, "--out", str(reference),
               "--compress-output"]
    process = subprocess.Popen(command)
    witness, first_failed = {}, False
    witness_path = Path(args.out).with_suffix(".witness.json")
    try:
        while len(witness) < len(names):
            try:
                receipts = json.loads((reference / "receipts.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                receipts = {}
            for name in names:
                if name in witness or name not in receipts:
                    continue
                receipt = receipts[name]
                storage = receipt.get("storage")
                if storage not in (name + ".npy", name + ".npy.gz"):
                    raise ValueError("noncanonical reference storage: " + name)
                path = reference / storage
                expected = captures.read_npy(path)
                if name.startswith("frame_"):
                    actual = captures.read_f32(native, name, 768 * 1344 * 3).reshape(768, 1344, 3)
                else:
                    actual = verify.capture(native, name)
                if actual.shape != expected.shape:
                    raise ValueError("reference shape mismatch: " + name)
                storage_hash = verify.digest(path)
                validate_receipt(name, receipt, storage_hash, list(expected.shape), generation_hash, sources)
                native_path = native / (name + ".f32")
                if not native_path.exists():
                    native_path = native_path.with_suffix(".f32.gz")
                result = verify.comparison(actual, expected)
                retained = name in keep or (not result["pass"] and not first_failed)
                first_failed = first_failed or not result["pass"]
                witness[name] = {"receipt": receipt, "comparison": result,
                                 "native_storage": native_path.name,
                                 "native_sha256": verify.digest(native_path), "retained": retained}
                # Publish the witness before unlinking independently verified bytes.
                atomic_json(witness_path, {"complete": False, "generation_sha256": generation_hash,
                                          "sources": sources, "arrays": witness})
                if not retained:
                    path.unlink()
                print("COMPARED", name, json.dumps(result), "retained=" + str(retained), flush=True)
            status = process.poll()
            if status is not None:
                if status != 0:
                    raise RuntimeError("independent reference failed: " + str(status))
                if len(witness) != len(names):
                    raise RuntimeError("independent reference omitted required arrays")
                break
            time.sleep(.5)
        if process.wait() != 0:
            raise RuntimeError("independent reference failed")
        receipts = json.loads((reference / "receipts.json").read_text())
        if verify.digest(manifest) != generation_hash or any(verify.digest(path) != sources[key] for key, path in source_paths.items()):
            raise ValueError("generation or validation sources changed during verification")
        for name, item in witness.items():
            if receipts.get(name) != item["receipt"] or verify.digest(native / item["native_storage"]) != item["native_sha256"]:
                raise ValueError("receipt or native capture changed: " + name)
            if item["retained"] and verify.digest(reference / item["receipt"]["storage"]) != item["receipt"]["sha256"]:
                raise ValueError("retained reference capture changed: " + name)
        atomic_json(witness_path, {"complete": True, "generation_sha256": generation_hash,
                                  "sources": sources, "arrays": witness})
        report = {"scope": "full_pipeline_streamed_reference", "generation_sha256": generation_hash,
                  "reference_receipts_sha256": verify.digest(reference / "receipts.json"),
                  "witness_sha256": verify.digest(witness_path), "sources": sources,
                  "storage_policy": "all_native_arrays; receipt-verified reference intermediates pruned",
                  "comparisons": {name: witness[name]["comparison"] for name in names},
                  "pass": all(item["comparison"]["pass"] for item in witness.values())}
        atomic_json(args.out, report)
        if not report["pass"]:
            raise RuntimeError("full H3 streaming pipeline parity failed")
        print("FULL_DEFAULT_H3_STREAMING_ACCEPTANCE_PASSED", len(witness), flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "native", "reference", "qwen-reference", "out"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
