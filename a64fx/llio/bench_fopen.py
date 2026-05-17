#!/usr/bin/env python3
"""bench_fopen.py -- file open + read throughput benchmark.

Usage:
    bench_fopen.py <file> [chunk_bytes=1048576] [passes=2] [api=os|stdio]

api=os    : os.open + os.read (unbuffered, syscall-per-chunk)
api=stdio : open(..., 'rb', buffering=0) + .read(chunk)
"""
import os
import sys
import time


def reduce_xor(buf: memoryview, acc: int) -> int:
    """8-byte XOR reduction over `buf`. Returns updated acc."""
    n = len(buf)
    n8 = n // 8
    if n8 > 0:
        # int.from_bytes per 8B is faster than iterating bytes.
        words = memoryview(buf)[: n8 * 8].cast("Q")
        for w in words:
            acc ^= w
    tail = buf[n8 * 8 :]
    for i, b in enumerate(tail):
        acc ^= b << ((i & 7) * 8)
    return acc & ((1 << 64) - 1)


def bench_os(path: str, chunk: int) -> tuple:
    fd = os.open(path, os.O_RDONLY)
    try:
        total = 0
        xsum = 0
        t0 = time.monotonic()
        while True:
            data = os.read(fd, chunk)
            if not data:
                break
            xsum = reduce_xor(memoryview(data), xsum)
            total += len(data)
        t1 = time.monotonic()
    finally:
        os.close(fd)
    return total, t1 - t0, xsum


def bench_stdio(path: str, chunk: int) -> tuple:
    total = 0
    xsum = 0
    with open(path, "rb", buffering=0) as f:
        t0 = time.monotonic()
        while True:
            data = f.read(chunk)
            if not data:
                break
            xsum = reduce_xor(memoryview(data), xsum)
            total += len(data)
        t1 = time.monotonic()
    return total, t1 - t0, xsum


def main():
    if len(sys.argv) < 2:
        print(
            "usage: bench_fopen.py <file> [chunk_bytes=1048576] "
            "[passes=2] [api=os|stdio]",
            file=sys.stderr,
        )
        return 2
    path = sys.argv[1]
    chunk = int(sys.argv[2]) if len(sys.argv) >= 3 else (1 << 20)
    passes = int(sys.argv[3]) if len(sys.argv) >= 4 else 2
    api = sys.argv[4] if len(sys.argv) >= 5 else "os"

    fsize = os.path.getsize(path)
    print(f"file        : {path}")
    print(f"size        : {fsize} bytes ({fsize / (1 << 30):.3f} GiB)")
    print(f"chunk       : {chunk} bytes")
    print(f"passes      : {passes}")
    print(f"api         : {api}\n")

    fn = bench_os if api == "os" else bench_stdio
    for p in range(passes):
        total, dt, xsum = fn(path, chunk)
        gib = total / (1 << 30)
        gb = total / 1e9
        tag = "cold" if p == 0 else "warm"
        print(
            f"pass {p}  {tag}  read={gib:.3f} GiB  time={dt:.3f} s  "
            f"bw={gb / dt:.3f} GB/s ({gib / dt:.3f} GiB/s)  "
            f"xor={xsum:016x}"
        )
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
