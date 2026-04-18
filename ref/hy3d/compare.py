"""Compare reference .npy outputs against CUDA runner outputs.

Examples:
    uv run python compare.py output ../../cuda/hy3d/cuda_output
    uv run python compare.py output ../../cuda/hy3d/cuda_output --manifest verify_manifest.json --profile f16
"""
import argparse
import json
import os
import sys

import numpy as np


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _shape_tuple(shape):
    return tuple(int(x) for x in shape)


def compare(name, ref, test, rtol=1e-3, atol=1e-4, max_abs_limit=None, mean_abs_limit=None):
    result = {
        "name": name,
        "shape_ref": _shape_tuple(ref.shape),
        "shape_test": _shape_tuple(test.shape),
        "shape_match": True,
    }
    if ref.shape != test.shape:
        result["shape_match"] = False
        result["ok"] = False
        result["status"] = "SHAPE_MISMATCH"
        return result

    diff = np.abs(ref - test)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    idx = tuple(int(x) for x in np.unravel_index(int(diff.argmax()), diff.shape))

    result.update(
        {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "worst_index": idx,
            "worst_ref": float(ref[idx]),
            "worst_test": float(test[idx]),
        }
    )

    if max_abs_limit is not None or mean_abs_limit is not None:
        ok_max = True if max_abs_limit is None else max_abs <= float(max_abs_limit)
        ok_mean = True if mean_abs_limit is None else mean_abs <= float(mean_abs_limit)
        result["ok"] = bool(ok_max and ok_mean)
        result["status"] = "OK" if result["ok"] else "FAIL"
        result["limits"] = {"max_abs": max_abs_limit, "mean_abs": mean_abs_limit}
    else:
        ok = bool(np.allclose(ref, test, rtol=rtol, atol=atol))
        result["ok"] = ok
        result["status"] = "OK" if ok else "FAIL"
        result["limits"] = {"rtol": rtol, "atol": atol}

    return result


def _print_result(res):
    if not res["shape_match"]:
        print(f"  {res['name']}: SHAPE MISMATCH ref={res['shape_ref']} test={res['shape_test']}")
        return
    shape = tuple(res["shape_ref"])
    print(f"  {res['name']}: {res['status']}  max={res['max_abs']:.2e} mean={res['mean_abs']:.2e} shape={shape}")
    if res["status"] != "OK":
        print(
            f"    worst@{tuple(res['worst_index'])}: "
            f"ref={res['worst_ref']:.6f} test={res['worst_test']:.6f}"
        )


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("ref_dir")
    p.add_argument("test_dir")
    p.add_argument("rtol", nargs="?", type=float, default=1e-3)
    p.add_argument("atol", nargs="?", type=float, default=1e-4)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--profile", type=str, default="f16")
    p.add_argument("--json-out", type=str, default=None)
    return p.parse_args()


def _manifest_entries(manifest, profile):
    profiles = manifest.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"profile '{profile}' missing in manifest")
    entries = profiles[profile]
    if not isinstance(entries, dict):
        raise ValueError(f"profile '{profile}' must be a dict")
    return entries


def main():
    args = _parse_args()
    ref_dir, test_dir = args.ref_dir, args.test_dir
    results = []
    missing_required = []
    missing_optional = []

    if args.manifest:
        manifest = _load_manifest(args.manifest)
        entries = _manifest_entries(manifest, args.profile)
        compared = 0
        for fname, cfg in sorted(entries.items()):
            required = bool(cfg.get("required", True))
            ref_path = os.path.join(ref_dir, fname)
            test_path = os.path.join(test_dir, fname)
            if not os.path.exists(ref_path) or not os.path.exists(test_path):
                if required:
                    missing_required.append(fname)
                else:
                    missing_optional.append(fname)
                continue

            ref = np.load(ref_path)
            test = np.load(test_path)
            res = compare(
                fname,
                ref,
                test,
                max_abs_limit=cfg.get("max_abs"),
                mean_abs_limit=cfg.get("mean_abs"),
            )
            res["required"] = required
            results.append(res)
            compared += 1
            _print_result(res)

        ok_count = sum(1 for r in results if r["ok"])
        fail_count = sum(1 for r in results if not r["ok"])
        fail_required = sum(1 for r in results if not r["ok"] and r.get("required", True))
        if missing_required:
            print(f"  missing required: {', '.join(sorted(missing_required))}")
        if missing_optional:
            print(f"  missing optional: {', '.join(sorted(missing_optional))}")
        print(
            f"\n{ok_count} OK, {fail_count} FAIL / {compared} compared "
            f"(missing required={len(missing_required)}, optional={len(missing_optional)})"
        )
        exit_code = 0 if (fail_required == 0 and not missing_required) else 1
    else:
        ref_files = {f for f in os.listdir(ref_dir) if f.endswith(".npy")}
        test_files = {f for f in os.listdir(test_dir) if f.endswith(".npy")}
        common = sorted(ref_files & test_files)
        for fname in common:
            ref = np.load(os.path.join(ref_dir, fname))
            test = np.load(os.path.join(test_dir, fname))
            res = compare(fname, ref, test, rtol=args.rtol, atol=args.atol)
            results.append(res)
            _print_result(res)
        ok_count = sum(1 for r in results if r["ok"])
        fail_count = sum(1 for r in results if not r["ok"])
        print(f"\n{ok_count} OK, {fail_count} FAIL / {len(common)} compared")
        exit_code = 0 if fail_count == 0 else 1

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        payload = {
            "ref_dir": ref_dir,
            "test_dir": test_dir,
            "profile": args.profile if args.manifest else None,
            "manifest": args.manifest,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "results": results,
            "exit_code": exit_code,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
