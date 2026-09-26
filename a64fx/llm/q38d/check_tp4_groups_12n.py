#!/usr/bin/env python3
"""Check rank agreement and distinct output streams in a 12-node TP4 run."""
import pathlib
import re
import sys


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_tp4_groups_12n.py RUN_DIR GEN")
    run = pathlib.Path(sys.argv[1])
    expected = int(sys.argv[2])
    groups = []
    for group in range(3):
        hashes = []
        for rank in range(group * 4, (group + 1) * 4):
            path = run / f"rank{rank}.log"
            log = path.read_text()
            if "Q38_GROUP_RESOURCE wall_s=" not in log:
                raise SystemExit(f"incomplete rank {rank}: missing resource footer in {path}")
            match = re.search(r"q38d: TP4 group=(\d+) rank=(\d+) tokens=(\d+) stream_hash=([0-9a-f]+)", log)
            if not match or (int(match[1]), int(match[2]), int(match[3])) != (group, rank % 4, expected):
                raise SystemExit(f"incorrect TP group or stream report for rank {rank}")
            hashes.append(match[4])
            if rank != group * 4:
                continue
            tokens = [int(x) for x in re.findall(r"q38d: token n=\d+ pos=\d+ id=(\d+)", log)]
            if len(tokens) != expected or "RESULT fmt=fp4 act=a16" not in log:
                raise SystemExit(f"incomplete rank {rank}: {len(tokens)}/{expected} tokens in {path}")
            groups.append(tokens)
        if len(set(hashes)) != 1:
            raise SystemExit(f"TP4 group {group} ranks disagree")
    if len({tuple(stream) for stream in groups}) != 3:
        raise SystemExit("TP4 groups did not produce three distinct streams")
    print(f"Q38_TP4_GROUPS_PASS groups=3 ranks=12 tokens_per_context={expected}")


if __name__ == "__main__":
    main()
