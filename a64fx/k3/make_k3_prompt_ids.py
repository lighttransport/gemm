#!/usr/bin/env python3
"""Encode a plain-text K3 prompt to the integer-ID format consumed by C11."""
from __future__ import print_function

import argparse
import sys


PAT_STR = "|".join([
    r"[\p{Han}]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
])


def encode(text, vocab):
    try:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
    except ImportError as exc:
        raise RuntimeError("tiktoken is required on the compute node: %s" % exc)
    mergeable = load_tiktoken_bpe(str(vocab))
    base = len(mergeable)
    specials = {"<|reserved_token_%d|>" % i: i
                for i in range(base, base + 256)}
    enc = tiktoken.Encoding(name="k3", pat_str=PAT_STR,
                            mergeable_ranks=mergeable,
                            special_tokens=specials)
    return enc.encode(text, disallowed_special=())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--bos", action="store_true")
    ap.add_argument("--repeat-to", action="store_true",
                    help="repeat the source text until --tokens is reached")
    args = ap.parse_args()
    text = open(args.text, encoding="utf-8").read()
    if not text:
        raise SystemExit("prompt text is empty")
    ids = encode(text, args.vocab)
    if args.repeat_to:
        seed = ids[:]
        while len(ids) < args.tokens:
            ids.extend(seed)
    if args.bos:
        ids = [163584] + ids
    if len(ids) < args.tokens:
        raise SystemExit("prompt encoded to %d tokens, need %d" %
                         (len(ids), args.tokens))
    ids = ids[:args.tokens]
    with open(args.output, "w") as f:
        for i in range(0, len(ids), 32):
            f.write(" ".join(str(x) for x in ids[i:i + 32]) + "\n")
    print("K3_PROMPT_IDS tokens=%d output=%s" % (len(ids), args.output))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        print("make_k3_prompt_ids: %s" % exc, file=sys.stderr)
        sys.exit(2)
