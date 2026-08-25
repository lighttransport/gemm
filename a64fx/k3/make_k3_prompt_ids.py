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

K3_BOS = 163584
K3_END_OF_MSG = 163586
K3_OPEN = 163587
K3_CLOSE = 163588
K3_SEP = 163589

K3_THINKING_EFFORT_MAX = (
    "`thinking_effort` guides on how much to think in your thinking channel "
    "(not including the response channel), supported values include `low`, "
    "`medium`, `high`, and `max`.\n"
    "Now the system is invoked with `thinking_effort=max`."
)


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


def _open_tag(tag, attrs, encode_text):
    ids = [K3_OPEN] + encode_text(tag)
    for key, value in attrs:
        # Match encoding_k3._open_tag segment boundaries exactly. Tiktoken is
        # not permitted to merge across these independently encoded pieces.
        ids += encode_text(" " + key)
        ids += encode_text('="')
        ids += encode_text(value)
        ids += encode_text('"')
    return ids + [K3_SEP]


def _close_tag(tag, encode_text):
    return [K3_CLOSE] + encode_text(tag) + [K3_SEP]


def build_chat_ids(content, encode_text, tokens, bos=False, repeat_to=False,
                   thinking=False, natural_length=False):
    """Render one official XTML user turn and generation prompt."""
    prefix = []
    if thinking:
        prefix += _open_tag("message", [("role", "system"),
                                         ("type", "thinking-effort")],
                            encode_text)
        prefix += encode_text(K3_THINKING_EFFORT_MAX)
        prefix += _close_tag("message", encode_text) + [K3_END_OF_MSG]
    prefix += _open_tag("message", [("role", "user")], encode_text)
    suffix = (_close_tag("message", encode_text) + [K3_END_OF_MSG] +
              _open_tag("message", [("role", "assistant")], encode_text) +
              _open_tag("think" if thinking else "response", [], encode_text))
    available = (len(content) if natural_length else
                 tokens - int(bos) - len(prefix) - len(suffix))
    if available < 0:
        raise ValueError("--tokens is too small for the K3 chat envelope")
    if int(bos) + len(prefix) + available + len(suffix) > tokens:
        raise ValueError("natural K3 chat prompt exceeds --tokens")
    body = list(content)
    if repeat_to and body:
        seed = body[:]
        while len(body) < available:
            body.extend(seed)
    if len(body) < available:
        raise ValueError("chat body encoded to %d tokens, need %d" %
                         (len(body), available))
    ids = prefix + body[:available] + suffix
    if bos:
        ids.insert(0, K3_BOS)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--bos", action="store_true")
    ap.add_argument("--chat", action="store_true",
                    help="wrap text as an XTML user turn and open assistant response")
    ap.add_argument("--thinking", action="store_true",
                    help="use K3's default thinking-effort=max chat template")
    ap.add_argument("--repeat-to", action="store_true",
                    help="repeat the source text until --tokens is reached")
    ap.add_argument("--natural-length", action="store_true",
                    help="treat --tokens as a maximum and retain the natural prompt length")
    args = ap.parse_args()
    if args.thinking and not args.chat:
        raise SystemExit("--thinking requires --chat")
    if args.natural_length and args.repeat_to:
        raise SystemExit("--natural-length and --repeat-to are exclusive")
    text = open(args.text, encoding="utf-8").read()
    if not text:
        raise SystemExit("prompt text is empty")
    ids = encode(text, args.vocab)
    if args.chat:
        ids = build_chat_ids(ids, lambda value: encode(value, args.vocab),
                             args.tokens, args.bos, args.repeat_to,
                             args.thinking, args.natural_length)
    else:
        if args.repeat_to:
            seed = ids[:]
            while len(ids) < args.tokens:
                ids.extend(seed)
        if args.bos:
            ids = [K3_BOS] + ids
        if args.natural_length:
            if len(ids) > args.tokens:
                raise SystemExit("prompt encoded to %d tokens, exceeds %d" %
                                 (len(ids), args.tokens))
        elif len(ids) < args.tokens:
            raise SystemExit("prompt encoded to %d tokens, need %d" %
                             (len(ids), args.tokens))
        if not args.natural_length:
            ids = ids[:args.tokens]
    with open(args.output, "w") as f:
        for i in range(0, len(ids), 32):
            f.write(" ".join(str(x) for x in ids[i:i + 32]) + "\n")
    print("K3_PROMPT_IDS tokens=%d max_tokens=%d chat=%d thinking=%d natural=%d output=%s" %
          (len(ids), args.tokens, int(args.chat), int(args.thinking),
           int(args.natural_length), args.output))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        print("make_k3_prompt_ids: %s" % exc, file=sys.stderr)
        sys.exit(2)
