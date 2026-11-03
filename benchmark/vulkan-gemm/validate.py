#!/usr/bin/env python3
"""Run correctness/argument checks against one or more already-built executables."""
import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executables", nargs="+")
    args = parser.parse_args()
    cases = [
        (["--help"], 0, "Usage:"),
        (["--info"], 0, "heap["),
        (["--m", "37", "--n", "41", "--k", "53", "--warmup", "0", "--iterations", "1"], 0, "correctness PASS"),
        (["--m", "1", "--n", "1", "--k", "1", "--warmup", "0", "--iterations", "1"], 0, "correctness PASS"),
        (["--m", "17", "--n", "3", "--k", "31", "--iterations", "1"], 0, "correctness PASS"),
        (["--m", "0"], 2, "Invalid arguments"),
        (["--k", "-1"], 2, "Invalid arguments"),
        (["--n", "4294967296"], 2, "Invalid arguments"),
        (["--n", "abc"], 2, "Invalid arguments"),
        (["--type", "unknown"], 2, "Invalid arguments"),
        (["--iterations", "0"], 2, "Invalid arguments"),
        (["--warmup", "100001"], 2, "Invalid arguments"),
        (["--m"], 2, "Invalid arguments"),
        (["--unknown", "1"], 2, "Invalid arguments"),
        (["--vram-test", "nan"], 2, "Invalid arguments"),
        (["--vram-test", "0"], 2, "Invalid arguments"),
        (["--vram-test", "--info"], 2, "Invalid arguments"),
        (["--device", "4294967280"], 1, "Invalid device index"),
        (["--type", "fp32", "--m", "100000", "--n", "100000"], 1, "dimensions exceed"),
        (["--type", "fp64", "--m", "20000", "--n", "20000", "--k", "1"], 1, "buffer exceeds"),
        (["--type", "int32", "--m", "1", "--n", "1", "--k", "8388608"], 1, "accumulation limits"),
        (["--vram-test", "0.03125"], 0, "VRAM PASS"),
        (["--vram-test", "--vram-reserve-mib", "65536"], 1, "Insufficient VRAM budget"),
    ]
    for rows, cols, kstep, pack in [(128,128,16,0), (128,128,16,1), (64,64,8,0),
                                      (64,128,16,0), (128,64,32,0)]:
        cases.append((["--type", "fp32", "--m", str(rows), "--n", str(cols),
                       "--k", "32", "--fp32-rows", str(rows), "--fp32-tile", str(cols),
                       "--fp32-kstep", str(kstep), "--fp32-pack-a", str(pack),
                       "--warmup", "0", "--iterations", "1"], 0, "all outputs"))
    cases.append((["--type", "fp32", "--m", "137", "--n", "131", "--k", "19",
                   "--iterations", "1"], 0, "all outputs"))
    cases.append((["--type", "fp32", "--fp32-kernel", "baseline", "--m", "128", "--n", "128",
                   "--k", "32", "--iterations", "1"], 0, "all outputs"))
    cases.append((["--fp32-kstep", "12"], 2, "Invalid arguments"))
    for m, n, k, pack, prefetch in [
            (129,132,20,0,1), (129,132,20,0,0), (129,132,4,0,1),
            (1,4,4,0,1), (65,68,8,0,1), (129,132,28,0,1),
            (132,132,19,1,1), (132,132,19,1,0), (4,4,1,1,1),
            (128,128,4,0,1), (128,128,20,0,1)]:
        cases.append((["--type", "fp32", "--m", str(m), "--n", str(n), "--k", str(k),
                       "--fp32-rows", "128", "--fp32-tile", "128",
                       "--fp32-pack-a", str(pack), "--fp32-prefetch", str(prefetch),
                       "--warmup", "0", "--iterations", "2"], 0, "all outputs"))
    for kstep in [8,32]:
        cases.append((["--type", "fp32", "--m", "65", "--n", "68", "--k", "12",
                       "--fp32-rows", "64", "--fp32-tile", "64", "--fp32-kstep", str(kstep),
                       "--warmup", "0", "--iterations", "2"], 0, "all outputs"))
    for m, n, tile in [(512,512,"128x64"), (256,256,"64x64"), (512,1024,"128x128"),
                       (64,1024,"64x64")]:
        cases.append((["--type", "fp32", "--m", str(m), "--n", str(n), "--k", "4",
                       "--warmup", "0", "--iterations", "1"], 0, "tile="+tile))
    cases.append((["--type", "fp32", "--m", "512", "--n", "512", "--k", "4",
                   "--fp32-tile", "128", "--warmup", "0", "--iterations", "1"], 0, "tile=128x128"))
    cases.append((["--type", "fp32", "--m", "512", "--n", "512", "--k", "4",
                   "--fp32-tile", "auto", "--fp32-rows", "auto",
                   "--warmup", "0", "--iterations", "1"], 0, "tile=128x64"))
    cases.extend([
        (["--batch", "0"], 2, "Invalid arguments"),
        (["--batch", "1025"], 2, "Invalid arguments"),
        (["--m", "37", "--n", "41", "--k", "53", "--batch", "4",
          "--warmup", "3", "--iterations", "7"], 0, "Batch limit=4"),
        (["--type", "fp32", "--m", "129", "--n", "132", "--k", "20", "--batch", "3",
          "--warmup", "2", "--iterations", "7"], 0, "Batch limit=3"),
        (["--type", "fp32", "--m", "8", "--n", "4", "--k", "4", "--batch", "1024",
          "--warmup", "0", "--iterations", "2"], 0, "correctness PASS"),
        (["--type", "fp32", "--m", "65", "--n", "68", "--k", "12",
          "--fp32-tile", "64", "--fp32-pad", "1", "--fp32-lds-prefetch", "1",
          "--iterations", "2"], 0, "all outputs"),
    ])
    for executable in args.executables:
        for arguments, code, expected in cases:
            command = [str(Path(executable).resolve()), *arguments]
            run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=90)
            if run.returncode != code or expected not in run.stdout:
                raise RuntimeError(f"{subprocess.list2cmdline(command)}\n"
                                   f"Expected exit={code}, text={expected!r}; got exit={run.returncode}\n{run.stdout}")
            if "--m" in arguments and code == 0:
                # Every supported type has a full small reference check plus requested shape.
                count = run.stdout.count("correctness PASS")
                unsupported = run.stdout.count(": UNSUPPORTED (required")
                expected_types = 1 if "--type" in arguments and arguments[arguments.index("--type")+1] != "all" else 6
                if count != 2 * (expected_types - unsupported):
                    raise RuntimeError(f"Missing correctness checks: {run.stdout}")
            print(f"PASS {Path(executable).name} {' '.join(arguments)}")
        print(f"{executable}: {len(cases)} cases PASS")


if __name__ == "__main__":
    main()
