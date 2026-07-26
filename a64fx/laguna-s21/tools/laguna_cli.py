#!/usr/bin/env python3
"""Client for the laguna runner's HTTP interface.

The runner has no tokenizer, so its API is ids-in/ids-out; this does the chat
templating and tokenisation on the client side and prints text.

  laguna_cli.py --host 10.0.0.1 --port 8080 chat "What is the capital of France?"
  laguna_cli.py --port 8080 chat-file q.txt --system "..." --no-think --sample
  laguna_cli.py --port 8080 complete "The A64FX processor"      # raw continuation
  laguna_cli.py --port 8080 health
  laguna_cli.py --port 8080 shutdown

Env: LAGUNA_TOKENIZER (tokenizer.json), as for laguna_tok.py.
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laguna_tok as T  # noqa: E402


def post(host, port, path, payload=None, timeout=3600):
    url = "http://%s:%d%s" % (host, port, path)
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if payload is not None or path != "/health" else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["chat", "chat-file", "complete", "health", "shutdown"])
    ap.add_argument("text", nargs="?", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--system")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--temp", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--min-p", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--raw", action="store_true", help="show special tokens in the reply")
    ap.add_argument("--json", action="store_true", help="print the raw server response")
    a = ap.parse_args()

    if a.command == "health":
        print(json.dumps(post(a.host, a.port, "/health", timeout=10), indent=2)); return 0
    if a.command == "shutdown":
        print(json.dumps(post(a.host, a.port, "/shutdown", payload={}, timeout=30))); return 0
    if a.text is None:
        ap.error("%s needs a prompt (or a file for chat-file)" % a.command)

    tok = T.Tok(T.TOKJSON)
    if a.command == "complete":
        ids = tok.encode(a.text, add_bos=True)
    else:
        user = open(a.text).read() if a.command == "chat-file" else a.text
        msgs = ([{"role": "system", "content": a.system}] if a.system else []) + \
               [{"role": "user", "content": user}]
        prompt = T.render_chat(msgs, add_generation_prompt=True,
                               enable_thinking=not a.no_think)
        if a.show_prompt:
            sys.stderr.write(prompt + "\n")
        ids = tok.encode(prompt)          # the template emits BOS itself

    req = {"ids": ids, "max_new": a.max_new}
    if a.sample:
        req["sample"] = True
    for k, v in (("temp", a.temp), ("top_k", a.top_k), ("top_p", a.top_p),
                 ("min_p", a.min_p), ("seed", a.seed)):
        if v is not None:
            req[k] = v

    r = post(a.host, a.port, "/generate", req)
    if a.json:
        print(json.dumps(r, indent=2)); return 0
    if "error" in r:
        sys.exit("server error: %s" % r["error"])
    print(tok.decode(r["ids"], raw=a.raw))
    sys.stderr.write("[%d tok, stop=%s, prefill %.1f tok/s, decode %.1f tok/s%s]\n" % (
        r["n"], r["stop"], r.get("prefill_tok_s", 0), r.get("decode_tok_s", 0),
        ", nan=%d" % r["nan"] if r.get("nan") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
