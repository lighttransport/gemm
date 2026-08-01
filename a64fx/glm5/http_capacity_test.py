#!/usr/bin/env python3
"""Send an exact-token-count multi-context request to the GLM-5.2 HTTP worker."""
import argparse
import json
import sys
import time
from urllib.request import Request, urlopen

from glm5_tokenizer import Tok, TOKJSON


def exact_prompt(tok, target, label):
    suffix = "\nReply exactly: capacity %s" % label
    overhead = len(tok.chat(suffix))
    fill = target - overhead
    if fill < 0:
        raise ValueError("target too short")
    text = " x" * fill + suffix
    actual = len(tok.chat(text))
    if actual != target:
        raise RuntimeError("fixture token count %d != %d" % (actual, target))
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:18080/v1/batch/completions")
    p.add_argument("--contexts", type=int, default=2)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--max-new", type=int, default=8)
    p.add_argument("--timeout", type=int, default=21600)
    p.add_argument("--tokenizer", default=TOKJSON)
    args = p.parse_args()
    tok = Tok(args.tokenizer)
    prompts = [exact_prompt(tok, args.tokens, str(i)) for i in range(args.contexts)]
    body = json.dumps({"prompts": prompts, "max_tokens": args.max_new}).encode()
    started = time.time()
    req = Request(args.url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=args.timeout) as response:
        result = json.loads(response.read().decode())
    result["client_elapsed_seconds"] = round(time.time() - started, 3)
    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
