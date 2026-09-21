#!/usr/bin/env python3
"""Strict native Qwen3-VL text parity at full and cropped output boundaries."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare import NONQUANTIZED_COSINE_THRESHOLD, _cosine_error


def load_fixture(reference: Path, branch: str):
    folder = reference / f"text_{branch}"
    metadata = json.loads((folder / "capture.json").read_text())
    if (metadata.get("hidden_boundary") != "before_final_rmsnorm" or
            metadata.get("calls") != 1 or metadata.get("norm_calls") != 1):
        raise ValueError("reference does not identify a single pre-final-RMSNorm capture")
    ids = np.load(folder / "input_ids.npy", allow_pickle=False)
    mask = np.load(folder / "attention_mask.npy", allow_pickle=False)
    hidden = np.load(folder / "hidden_prenorm.npy", allow_pickle=False)
    name = "prompt_embeds.npy" if branch == "positive" else "negative_prompt_embeds.npy"
    cropped = np.load(reference / name, allow_pickle=False)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype.kind not in "iu":
        raise ValueError("expected batch-one integer token IDs")
    if mask.shape != ids.shape or not np.all(mask == 1):
        raise ValueError("native text encoder currently requires unpadded input")
    if ids.shape[1] > 4096 or np.any(ids < 0) or np.any(ids >= 151936):
        raise ValueError("token IDs/sequence length outside native bounds")
    if np.isin(ids, [151652, 151653, 151655, 151656]).any():
        raise ValueError("native text parity does not support vision inputs")
    drop = metadata["drop_idx"]
    if not isinstance(drop, int) or not 0 <= drop < ids.shape[1]:
        raise ValueError("invalid system-prefix crop index")
    if hidden.shape != (*ids.shape, 4096) or cropped.shape != (1, ids.shape[1] - drop, 4096):
        raise ValueError("reference hidden/embedding shape mismatch")
    if not np.isfinite(hidden).all() or not np.isfinite(cropped).all():
        raise ValueError("nonfinite reference embeddings")
    if not np.array_equal(hidden[:, drop:], cropped):
        raise ValueError("cropped prompt fixture is not the captured pre-norm output")
    return ids[0], hidden[0], cropped[0], drop


def compare_hidden(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError(f"hidden shape mismatch: {reference.shape} vs {candidate.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("nonfinite hidden state")
    cosine, relative_l2 = _cosine_error(reference, candidate)
    return {"cosine": cosine, "relative_l2": relative_l2,
            "mae": float(np.mean(np.abs(reference.astype(np.float64) - candidate))),
            "threshold": NONQUANTIZED_COSINE_THRESHOLD,
            "passed": bool(cosine >= NONQUANTIZED_COSINE_THRESHOLD)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--prompt", default="a red apple on a white table")
    ap.add_argument("--negative-prompt")
    ap.add_argument("--work-dir", type=Path, default=Path("tmp/qimg21-text-regression"))
    ap.add_argument("--reference-dir", type=Path, help="Reuse an explicit existing text-stage capture")
    ap.add_argument("--native-attention", choices=("custom", "cutlass-efficient", "flash-exact"),
                    default="flash-exact")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    reference = args.reference_dir.resolve() if args.reference_dir else work / "reference"
    if args.reference_dir is None:
        # Reject stale captures instead of combining fixtures from two runs.
        reference.mkdir()
        command = [sys.executable, str(root / "cuda/qimg21/test_cuda_qimg21.py"),
                   "--test-text", "--dump-text-stages", "--model", str(args.model.resolve()),
                   "--prompt", args.prompt, "--dtype", "bf16", "--dump-dir", str(reference)]
        if args.negative_prompt is not None:
            command.extend(["--negative-prompt", args.negative_prompt])
        subprocess.run(command, check=True)
    results = {}
    branches = ["positive"] + (["negative"] if args.negative_prompt is not None else [])
    for branch in branches:
        ids, hidden, cropped, drop = load_fixture(reference, branch)
        candidate = work / f"native_{branch}"
        candidate.mkdir()  # Never accept a stale candidate after a failed launch.
        token_path = candidate / "tokens.txt"
        token_path.write_text("\n".join(str(int(v)) for v in ids) + "\n")
        hidden_path = candidate / "hidden.npy"
        subprocess.run([str(root / "cuda/qimg21/test_cuda_qimg21_text"),
                        "--model", str(args.model.resolve()), "--tokens", str(token_path),
                        "--attention", args.native_attention, "--out", str(hidden_path)], check=True)
        got = np.load(hidden_path, allow_pickle=False)
        results[branch] = {"full_prenorm": compare_hidden(hidden, got),
                           "cropped_prompt": compare_hidden(cropped, got[drop:])}
        # Persist each branch before starting another GPU job.
        (work / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({branch: results[branch]}, indent=2), flush=True)
    return 0 if all(metric["passed"] for branch in results.values() for metric in branch.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
