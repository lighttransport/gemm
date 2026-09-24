#!/usr/bin/env python3
"""Gate held-out quality against NVFP4 using a common BF16-logit reference.

This does not certify greedy trace equality or throughput; those are separate
checks. Inputs are JSON stdout from qwen38_lowbit_eval on the same token file.
"""
import argparse
import json
import math
from pathlib import Path


def compare(baseline, candidate):
    errors = []
    if baseline.get("format") != "nvfp4" or baseline.get("arithmetic") != 0:
        errors.append("baseline must use source NVFP4 and FP32 activations")
    for label, record in (("baseline", baseline), ("candidate", candidate)):
        if record.get("has_reference") is not True:
            errors.append(label + ": BF16 reference required")
        if not isinstance(record.get("tokens"), int) or record["tokens"] < 2:
            errors.append(label + ": incomplete token sequence")
        for field in ("nll", "perplexity", "relative_l2", "max_abs", "kl"):
            value = record.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < -1e-12:
                errors.append(label + ": invalid " + field)
        if not record.get("token_hash"):
            errors.append(label + ": missing token identity")
        # An evaluator can compare arbitrary logit files. Only a recorded BF16
        # reference is eligible for this quality gate.
        if record.get("reference_format") != 3 or record.get("reference_arithmetic") != 0:
            errors.append(label + ": reference is not unquantized BF16")
        if not record.get("reference_token_hash"):
            errors.append(label + ": missing reference identity")
        if not record.get("reference_hash"):
            errors.append(label + ": missing reference payload identity")
    for field in ("tokens", "token_hash", "reference_token_hash", "reference_hash"):
        if baseline.get(field) != candidate.get(field):
            errors.append("mismatched " + field)
    if not errors:
        for field in ("nll", "relative_l2", "max_abs", "kl"):
            if candidate[field] > baseline[field]:
                errors.append(field + " exceeds NVFP4 baseline")
    return {"passed": not errors, "errors": errors,
            "greedy_trace_validated": False, "throughput_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    result = compare(json.loads(Path(args.baseline).read_text()),
                     json.loads(Path(args.candidate).read_text()))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
