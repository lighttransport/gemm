#!/usr/bin/env python3
"""Compare E0M3 decode tiles with the bulk winner using GLM5.3-Flash shapes.

Dimensions follow Strata/include/strata/artifact/glm_native.hpp's validated
tensor schema (K,N below). This measures synthetic resident W4A4 projections,
not model inference, routing, SwiGLU, quantization or weight transfers.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import subprocess

SHAPES = {
    "expert_gate_up": (4096,2048),
    "expert_gate_up_fused": (4096,4096),
    "expert_down": (2048,4096),
    "dense_gate_up": (4096,12288),
    "dense_down": (12288,4096),
    "kda_qkv_projection": (4096,8192),
    "kda_output": (8192,4096),
    "mla_q_a": (4096,1536),
    "mla_q_b": (1536,16384),
    "mla_kv_a": (4096,512),
    "mla_output": (16384,4096),
    "indexer_q_b": (1536,4096),
    "lm_head": (4096,154880),
}


def write_dispatch(out):
    """Keep bulk unless decode is at least 2% faster than the paired run."""
    groups = {}
    with (out / "summary.csv").open() as f:
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in ("projection","m","n","k","cache"))
            groups.setdefault(key,{})[row["library"]] = row
    fields = ["projection","m","n","k","cache","library","candidate","median_us","speedup","status"]
    with (out / "dispatch.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        for pair in groups.values():
            bulk, decode = pair["bulk"], pair["decode"]
            winner = decode if float(decode["median_us"]) < 0.98 * float(bulk["median_us"]) else bulk
            writer.writerow(winner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",choices=("experts","glm"),default="glm")
    parser.add_argument("--shapes",help="Comma-separated subset of the named shapes")
    parser.add_argument("--batches",default="1,16,64")
    parser.add_argument("--cache",choices=("warm","cold","both"),default="both")
    parser.add_argument("--iters",type=int,default=100)
    parser.add_argument("--rounds",type=int,default=5)
    parser.add_argument("--output",type=Path,default=Path("build/decode"))
    parser.add_argument("--summarize-only",action="store_true",help="Regenerate dispatch.csv from an existing summary")
    parser.add_argument("--resume",action="store_true",help="Reuse completed rows in an interrupted sweep")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = (root / args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:
        write_dispatch(out)
        return
    shapes = args.shapes.split(",") if args.shapes else list(SHAPES)[:3] if args.suite == "experts" else list(SHAPES)
    batches = [int(m) for m in args.batches.split(",")]
    if any(m < 1 or m > 16384 for m in batches):
        parser.error("batches must be between 1 and 16384")
    if any(name not in SHAPES for name in shapes):
        parser.error("unknown shape")
    caches = ("warm","cold") if args.cache == "both" else (args.cache,)
    libraries = {"bulk": root / "build/libcutlass_e0m3.so", "decode": root / "build/libdecode_e0m3.so"}
    if args.resume:
        expected = dict(line.split("  ",1)[::-1] for line in (out / "libraries.sha256").read_text().splitlines())
        for path in libraries.values():
            if expected.get(path.name) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"Cannot reuse rows from a different GPU library: {path.name}")
    (out / ("libraries.resume.sha256" if args.resume else "libraries.sha256")).write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in [*libraries.values(),root.parent / "fp4/cublas_fp4_gemm"]))
    fields = ["projection","m","n","k","cache","library","median_us","tops","candidate","speedup","status"]
    previous = {}
    if args.resume:
        with (out / "summary.csv").open() as f:
            for row in csv.DictReader(f):
                if row["status"] != "PASS":
                    raise ValueError("Cannot resume an unverified row")
                previous[(row["projection"],int(row["m"]),row["cache"],row["library"])] = row
    with (out / "summary.csv").open("a" if args.resume else "w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        if not args.resume:
            writer.writeheader()
        for name in shapes:
            k,n = SHAPES[name]
            for m in batches:
                for cache in caches:
                    baseline = None
                    for label,path in libraries.items():
                        stem = f"{name}_m{m}_{cache}_{label}"
                        raw = out / f"{stem}.csv"
                        prior = previous.get((name,m,cache,label))
                        if prior:
                            with raw.open() as data:
                                saved, = list(csv.DictReader(data))
                            required = {"m":m,"n":n,"k":k,"iters":args.iters,"rounds":args.rounds,
                                "flush_mb":64 if cache == "cold" else 0}
                            if saved["status"] != "PASS" or saved["timing"] != "graph" or any(
                                    int(saved[key]) != value for key,value in required.items()):
                                raise ValueError(f"Cannot resume different benchmark settings: {stem}")
                            if label == "bulk":
                                baseline = float(prior["median_us"]) / 1000
                            continue
                        command = [str(root.parent / "fp4/cublas_fp4_gemm"),"--format","e0m3","--output","bf16",
                            "--cutlass",str(path),"-m",str(m),"-n",str(n),"-k",str(k),"--graph",
                            "--compare-cublas","--raster","2","--swizzle","4","--iters",str(args.iters),
                            "--rounds",str(args.rounds),"--csv",str(raw)]
                        if cache == "cold":
                            command += ["--flush-mb","64"]
                        if k * n > 1 << 28:
                            command += ["--pilot-iters","4"]
                        try:
                            result = subprocess.run(command,cwd=root,text=True,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,timeout=120)
                        except subprocess.TimeoutExpired as error:
                            partial = error.stdout or b""
                            (out / f"{stem}.log").write_text(partial.decode(errors="replace") if isinstance(partial,bytes) else partial)
                            raise
                        (out / f"{stem}.log").write_text(result.stdout)
                        if result.returncode:
                            raise RuntimeError(f"{stem} failed ({result.returncode}); see its log")
                        with raw.open() as data:
                            row, = list(csv.DictReader(data))
                        if row["status"] != "PASS":
                            raise ValueError(f"Unverified benchmark result: {stem}")
                        ms = float(row["median_ms"])
                        if label == "bulk":
                            baseline = ms
                        writer.writerow(dict(projection=name,m=m,n=n,k=k,cache=cache,library=label,
                            median_us=f"{ms * 1000:.4f}",tops=row["tops"],candidate=row["algo_index"],
                            speedup=f"{baseline / ms:.4f}",status=row["status"]))
                        f.flush()
                        print(f"{name} M={m} {cache} {label}: {ms * 1000:.3f} us, {baseline / ms:.2f}x",flush=True)
    write_dispatch(out)


if __name__ == "__main__":
    main()
