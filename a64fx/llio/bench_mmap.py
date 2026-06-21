#!/usr/bin/env python3
"""bench_mmap.py -- mmap-and-touch throughput benchmark.

Usage:
    bench_mmap.py <file> [mode=seq8|seq|page] [passes=2] [populate=0|1]

modes (matching bench_mmap.c):
    seq8 : contiguous 8-byte XOR scan
    seq  : 1 byte per 64B cache line
    page : 1 byte per 4 KiB page
"""
import mmap
import os
import sys
import time


def scan_seq8(view: memoryview) -> int:
    acc = 0
    words = view.cast("Q")
    for w in words:
        acc ^= w
    return acc & ((1 << 64) - 1)


def scan_seq(view: memoryview) -> int:
    acc = 0
    n = len(view)
    for i in range(0, n, 64):
        acc ^= view[i]
    return acc


def scan_page(view: memoryview) -> int:
    acc = 0
    n = len(view)
    for i in range(0, n, 4096):
        acc ^= view[i]
    return acc


SCANS = {"seq8": scan_seq8, "seq": scan_seq, "page": scan_page}


def main():
    if len(sys.argv) < 2:
        print(
            "usage: bench_mmap.py <file> [mode=seq8|seq|page] "
            "[passes=2] [populate=0|1]",
            file=sys.stderr,
        )
        return 2
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) >= 3 else "seq8"
    passes = int(sys.argv[3]) if len(sys.argv) >= 4 else 2
    populate = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
    scan = SCANS.get(mode)
    if scan is None:
        print(f"unknown mode: {mode}", file=sys.stderr)
        return 2

    fsize = os.path.getsize(path)
    print(f"file        : {path}")
    print(f"size        : {fsize} bytes ({fsize / (1 << 30):.3f} GiB)")
    print(f"mode        : {mode}")
    print(f"passes      : {passes}")
    print(f"populate    : {populate}\n")

    fd = os.open(path, os.O_RDONLY)
    try:
        flags = mmap.MAP_PRIVATE
        if populate and hasattr(mmap, "MAP_POPULATE"):
            flags |= mmap.MAP_POPULATE

        tm0 = time.monotonic()
        mm = mmap.mmap(fd, fsize, flags=flags, prot=mmap.PROT_READ)
        tm1 = time.monotonic()
        print(f"mmap        : {(tm1 - tm0) * 1e3:.3f} ms"
              + (" (MAP_POPULATE)" if populate else ""))
        try:
            mm.madvise(mmap.MADV_SEQUENTIAL)
        except (AttributeError, OSError):
            pass

        view = memoryview(mm)
        for p in range(passes):
            t0 = time.monotonic()
            acc = scan(view)
            t1 = time.monotonic()
            dt = t1 - t0
            gib = fsize / (1 << 30)
            gb = fsize / 1e9
            tag = "cold" if p == 0 else "warm"
            print(
                f"pass {p}  {tag}  scan={gib:.3f} GiB  time={dt:.3f} s  "
                f"bw={gb / dt:.3f} GB/s ({gib / dt:.3f} GiB/s)  "
                f"xor={acc:016x}"
            )
            sys.stdout.flush()
        view.release()
        mm.close()
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
