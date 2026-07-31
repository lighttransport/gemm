#!/usr/bin/env python3
"""Decode generated_ids from a K3FULLV1 output file using tiktoken."""
from __future__ import print_function

import argparse
import sys

from make_k3_prompt_ids import PAT_STR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--text-output", required=True)
    args = ap.parse_args()
    try:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
    except ImportError as exc:
        raise SystemExit("tiktoken is required: %s" % exc)
    mergeable = load_tiktoken_bpe(args.vocab)
    base = len(mergeable)
    specials = {"<|reserved_token_%d|>" % i: i
                for i in range(base, base + 256)}
    enc = tiktoken.Encoding(name="k3", pat_str=PAT_STR,
                            mergeable_ranks=mergeable,
                            special_tokens=specials)
    line = None
    with open(args.output) as f:
        for candidate in f:
            if candidate.startswith("generated_ids:"):
                line = candidate
                break
    if line is None:
        raise SystemExit("generated_ids line is missing")
    ids = [int(x) for x in line.split(":", 1)[1].split()]
    text = enc.decode(ids)
    with open(args.text_output, "w", encoding="utf-8") as f:
        f.write(text)
    print("K3_DECODED_OUTPUT tokens=%d text=%s" % (len(ids), args.text_output))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print("decode_k3_output: %s" % exc, file=sys.stderr)
        sys.exit(2)
