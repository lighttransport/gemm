#!/usr/bin/env python3
"""Check independent context hashes on every rank of three TP4 groups."""
import pathlib
import re
import sys


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: check_q38d_batch.py RUN_DIR CONTEXTS GEN")
    run = pathlib.Path(sys.argv[1])
    counts = [int(x) for x in sys.argv[2].split(",")]
    if len(counts) == 1:
        counts *= 3
    if len(counts) != 3 or any(n < 1 for n in counts):
        raise SystemExit("CONTEXTS must be N or N,N,N with positive counts")
    gen = int(sys.argv[3])
    groups = []
    for group in range(3):
        contexts = counts[group]
        rank_hashes = []
        for rank in range(group * 4, (group + 1) * 4):
            log = (run / f"rank{rank}.log").read_text()
            if "Q38_GROUP_RESOURCE wall_s=" not in log:
                raise SystemExit(f"rank {rank} did not exit cleanly")
            if not re.search(rf"q38d_batch: TP4 group={group} contexts={contexts} decode_tokens={contexts * gen} ", log):
                raise SystemExit(f"rank {rank} missing batch timing")
            rows = re.findall(r"q38d_batch: group=(\d+) context=(\d+) output_hash=([0-9a-f]+)", log)
            if len(rows) != contexts or any((int(g), int(c)) != (group, i) for i, (g, c, _) in enumerate(rows)):
                raise SystemExit(f"rank {rank} has incomplete context outputs")
            rank_hashes.append(tuple(h for _, _, h in rows))
        if len(set(rank_hashes)) != 1:
            raise SystemExit(f"group {group} ranks disagree")
        groups.extend(rank_hashes[0])
    print(f"Q38_BATCH_PASS groups=3 contexts_by_group={counts} total_contexts={sum(counts)} gen={gen} distinct_streams={len(set(groups))}")


if __name__ == "__main__":
    main()
