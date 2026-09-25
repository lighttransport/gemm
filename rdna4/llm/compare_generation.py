#!/usr/bin/env python3
"""Compare complete generation traces, including the selected EOS and raw bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np


def read_trace(prefix, repeat):
    stem = Path(f"{prefix}.{repeat}")
    tokens = [int(s) for s in Path(f"{stem}.tokens").read_text().splitlines()]
    prompt = [int(s) for s in Path(f"{prefix}.prompt.tokens").read_text().splitlines()]
    body = Path(f"{stem}.output").read_bytes()
    log = Path(f"{prefix}.log").read_text(errors="replace")
    summaries = re.findall(r"GENERATION finish=(\w+) selected=(\d+) emitted=(\d+) synthetic=(\d+)", log)
    if repeat >= len(summaries) or "Result: FAIL" in log or "Result: PASS" not in log:
        raise ValueError(f"{prefix}: successful complete generation missing")
    reason, selected, emitted, synthetic = summaries[repeat]
    if reason not in ("length", "eos") or int(synthetic) or int(selected) != len(tokens):
        raise ValueError(f"{prefix}: incomplete, synthetic, or inconsistent trace")
    if len(tokens) != int(emitted) + (reason == "eos"):
        raise ValueError(f"{prefix}: emitted/stop token accounting mismatch")
    logits = np.fromfile(f"{stem}.logits.f32", dtype="<f4")
    if not tokens or not logits.size or logits.size % len(tokens):
        raise ValueError(f"{prefix}: logits rows do not match selected tokens")
    logits = logits.reshape(len(tokens), -1)
    if not np.isfinite(logits).all():
        raise ValueError(f"{prefix}: non-finite model logits")
    return prompt, tokens, body, reason, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ours", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--require-logits", action="store_true")
    args = parser.parse_args()
    ap, at, ab, ar, al = read_trace(args.ours, args.repeat)
    bp, bt, bb, br, bl = read_trace(args.reference, args.repeat)
    first_token = next((i for i, pair in enumerate(zip(at, bt)) if pair[0] != pair[1]), None)
    if first_token is None and len(at) != len(bt):
        first_token = min(len(at), len(bt))
    rows = min(len(at), len(bt), first_token + 1 if first_token is not None else len(at))
    if al.shape[1] != bl.shape[1]:
        raise ValueError("vocabulary sizes differ")
    equal = al[:rows].view("u4") == bl[:rows].view("u4")
    different = np.argwhere(~equal)
    first_logit = different[0].tolist() if len(different) else None
    # Only compare logits whose preceding input token sequences are identical.
    diff = al[:rows].astype("f8") - bl[:rows]
    result = {
        "prompt_tokens_identical": ap == bp, "tokens_identical": at == bt,
        "bytes_identical": ab == bb, "finish_identical": ar == br,
        "first_token_mismatch": first_token, "first_logit_mismatch": first_logit,
        "teacher_forced_rows": rows,
        "logits_identical": al.shape == bl.shape and bool(equal.all()) and at == bt,
        "logit_max_abs": float(np.abs(diff).max()) if rows else None,
        "logit_relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(bl[:rows].astype("f8")), 1e-30)),
        "ours_sha256": hashlib.sha256(ab).hexdigest(),
        "reference_sha256": hashlib.sha256(bb).hexdigest(),
        "selected_tokens": [len(at), len(bt)], "finish": [ar, br],
    }
    print(json.dumps(result, indent=2))
    success = all(result[k] for k in ("prompt_tokens_identical", "tokens_identical", "bytes_identical", "finish_identical"))
    return 0 if success and (not args.require_logits or result["logits_identical"]) else 1


if __name__ == "__main__":
    sys.exit(main())
