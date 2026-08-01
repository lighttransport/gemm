#!/usr/bin/env python3
"""Build an exact-length packed-u32 prompt with an explicit text tail."""

import argparse
import struct
from pathlib import Path

from glm5_tokenizer import TOKJSON, Tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-bin", required=True)
    ap.add_argument("--tail-text", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    raw = Path(args.base_bin).read_bytes()
    if not raw or len(raw) % 4:
        raise SystemExit("base prompt must be non-empty packed uint32 data")
    base = struct.unpack("<%dI" % (len(raw) // 4), raw)
    tail_text = Path(args.tail_text).read_text()
    tail = Tok(TOKJSON).encode(tail_text)
    if not tail:
        raise SystemExit("tail text tokenized to an empty sequence")
    if len(tail) > args.tokens:
        raise SystemExit("tail has %d tokens, target is only %d" % (len(tail), args.tokens))

    fill_n = args.tokens - len(tail)
    ids = [base[i % len(base)] for i in range(fill_n)]
    ids.extend(tail)
    with open(args.output, "wb") as f:
        f.write(struct.pack("<%dI" % len(ids), *ids))
    print("long prompt: base=%d fill=%d tail=%d total=%d -> %s" %
          (len(base), fill_n, len(tail), len(ids), args.output))


if __name__ == "__main__":
    main()
