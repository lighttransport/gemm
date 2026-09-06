#!/usr/bin/env python3
"""Count E4M3 byte classes in a staged GLM-5.3F rank manifest/blob."""
import argparse
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("blob", type=Path)
    parser.add_argument("--max-gib", type=float, default=1.0)
    args = parser.parse_args()
    budget = int(args.max_gib * (1 << 30))
    hist = Counter()
    sampled = 0
    with args.manifest.open() as manifest, args.blob.open("rb", buffering=0) as blob:
        for line in manifest:
            fields = line.split()
            if not fields or fields[0].startswith("#") or len(fields) < 4:
                continue
            offset, dtype = int(fields[0]), fields[1]
            if dtype != "F8_E4M3":
                continue
            ndims = int(fields[2])
            if len(fields) < 4 + ndims:
                continue
            shape = [int(x) for x in fields[3:3 + ndims]]
            nbytes = 1
            for dim in shape:
                nbytes *= dim
            nbytes = min(nbytes, budget - sampled)
            if nbytes <= 0:
                break
            blob.seek(offset)
            left = nbytes
            while left:
                data = blob.read(min(left, 8 << 20))
                if not data:
                    raise EOFError("short staged blob read")
                hist.update(data)
                left -= len(data)
            sampled += nbytes
    subnormal = sum(n for q, n in hist.items() if (q & 0x78) == 0 and (q & 7))
    zero = hist[0] + hist[0x80]
    nan = hist[0x7f] + hist[0xff]
    print("GLM53F_FP8_HIST bytes=%d zero=%d subnormal=%d nan=%d "
          "zero_pct=%.6f subnormal_pct=%.6f" %
          (sampled, zero, subnormal, nan, 100.0 * zero / sampled,
           100.0 * subnormal / sampled))


if __name__ == "__main__":
    main()
