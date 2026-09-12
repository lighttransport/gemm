#!/usr/bin/env python3
"""Emit the V4.1 Engram tokenizer/hash metadata in a compact binary form."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from tokenizers import Regex, Tokenizer, normalizers

MAGIC = b"DS41FENG1"
VOCAB = 129280
LAYERS = (1, 14)
MAX_N = 4
HEADS = 8


def token_map(tokenizer):
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
        normalizers.Lowercase(), normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), "\ue000"), normalizers.Strip(),
        normalizers.Replace("\ue000", " "),
    ])
    backend = tokenizer
    lookup, keys = [], {}
    for token_id in range(len(tokenizer.get_vocab())):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = backend.id_to_token(token_id) if "\ufffd" in text else normalizer.normalize_str(text) or text
        lookup.append(keys.setdefault(key, len(keys)))
    if len(lookup) != VOCAB or len(keys) != 99092:
        raise RuntimeError(f"unexpected tokenizer map: raw={len(lookup)} compressed={len(keys)}")
    return lookup


def primes():
    out = np.zeros((2, 3, HEADS), dtype=np.uint64)
    seen = set()
    candidate = 16000000
    for li in range(2):
        for ni in range(3):
            for hi in range(HEADS):
                while True:
                    candidate += 1
                    if candidate not in seen and all(candidate % d for d in range(2, int(candidate ** 0.5) + 1)):
                        break
                seen.add(candidate)
                out[li, ni, hi] = candidate
    return out


def multipliers():
    out = np.zeros((2, 4), dtype=np.uint64)
    bound = (np.iinfo(np.int64).max // 99092) // 2
    for i, layer in enumerate(LAYERS):
        rng = np.random.default_rng(10007 * layer)
        out[i] = rng.integers(0, bound, size=4, dtype=np.int64).astype(np.uint64) * 2 + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    mapping = np.asarray(token_map(tok), dtype=np.uint32)
    ps, ms = primes(), multipliers()
    with open(args.output, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", VOCAB, 99092, HEADS))
        f.write(mapping.tobytes())
        f.write(ms.tobytes())
        f.write(ps.tobytes())


if __name__ == "__main__":
    main()
