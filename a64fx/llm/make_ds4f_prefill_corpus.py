#!/usr/bin/env python3
"""Create the fixed 4096-token DS4F prefill benchmark corpus."""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
from ds4f_tokenizer import DS4FTokenizer  # noqa: E402


SOURCE_SHA256 = "875817612dda0e7b4865059a0265bceba6a7fc0e91ce050d71d2876508b38c06"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tokens", type=int, default=4096)
    args = ap.parse_args()

    raw = args.source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit("benchmark source changed: %s != %s" % (digest, SOURCE_SHA256))
    text = raw.decode("utf-8")
    ids = DS4FTokenizer(str(args.tokenizer)).encode(text, add_bos=False)
    if len(ids) < args.tokens:
        raise SystemExit("source produced only %d tokens" % len(ids))
    ids = ids[:args.tokens]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(" ".join(str(v) for v in ids) + "\n")
    print("DS4F_PREFILL_CORPUS tokens=%d source_sha256=%s out=%s" %
          (len(ids), digest, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
