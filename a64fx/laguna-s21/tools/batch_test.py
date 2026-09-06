#!/usr/bin/env python3
"""Correctness and throughput test for the batched serve path.

Correctness: a prompt must produce the *same* tokens whether it is served alone or
concurrently with others. Batching changes which sequences share a forward pass,
and nothing else — if it changes the output, a slot is reading another sequence's
KV or the activation packing is wrong.

Throughput: decode is weight-bandwidth-bound, so K concurrent sequences should
take much less than K times as long as one.

Usage: batch_test.py --port 8500 [--k 4] [--max-new 48]
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

sys.path.insert(0, __file__.rsplit('/', 1)[0])
import laguna_tok as T  # noqa: E402


def generate(host, port, ids, max_new, out, idx):
    body = json.dumps({"ids": ids, "max_new": max_new}).encode()
    req = urllib.request.Request("http://%s:%d/generate" % (host, port), data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out[idx] = json.loads(r.read().decode())
    out[idx]["wall"] = time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=48)
    a = ap.parse_args()

    tok = T.Tok(T.TOKJSON)
    questions = ["What is the capital of France?",
                 "Name one Japanese supercomputer.",
                 "What is 17 minus 8?",
                 "Which company makes the A64FX?",
                 "What is the capital of Japan?",
                 "Name a programming language.",
                 "What colour is the sky?",
                 "How many days are in a week?"][:a.k]
    prompts = [tok.encode(T.render_chat([{"role": "user", "content": q}],
                                        add_generation_prompt=True, enable_thinking=False))
               for q in questions]

    print("=== sequential (one at a time) ===")
    seq_out = [None]*len(prompts)
    t0 = time.time()
    for i, p in enumerate(prompts):
        generate(a.host, a.port, p, a.max_new, seq_out, i)
        print("  q%d: %s" % (i, tok.decode(seq_out[i]["ids"]).strip()[:70]))
    t_seq = time.time() - t0

    print("=== concurrent (batched) ===")
    bat_out = [None]*len(prompts)
    threads = [threading.Thread(target=generate,
                                args=(a.host, a.port, p, a.max_new, bat_out, i))
               for i, p in enumerate(prompts)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_bat = time.time() - t0
    for i in range(len(prompts)):
        print("  q%d: %s" % (i, tok.decode(bat_out[i]["ids"]).strip()[:70]))

    print("=== correctness ===")
    bad = 0
    for i in range(len(prompts)):
        same = seq_out[i]["ids"] == bat_out[i]["ids"]
        if not same:
            bad += 1
            n = min(len(seq_out[i]["ids"]), len(bat_out[i]["ids"]))
            d = next((j for j in range(n) if seq_out[i]["ids"][j] != bat_out[i]["ids"][j]), n)
            print("  q%d ** DIFFERS ** at token %d of %d" % (i, d, n))
        else:
            print("  q%d identical (%d tokens)" % (i, len(seq_out[i]["ids"])))
    tot_lockstep = sum(o.get("lockstep_disagree", 0) for o in bat_out)
    print("  lockstep disagreements across the batch: %d" % tot_lockstep)

    print("=== throughput ===")
    gen_tok = sum(o["n"] for o in bat_out)
    print("  sequential: %.1f s for %d requests" % (t_seq, len(prompts)))
    print("  batched   : %.1f s for %d requests   speedup %.2fx" % (t_bat, len(prompts), t_seq/t_bat))
    print("  aggregate decode: %.1f tok/s batched vs %.1f sequential"
          % (gen_tok/t_bat, sum(o["n"] for o in seq_out)/t_seq))
    print("FAIL" if (bad or tot_lockstep) else "batched serving: PASS")
    return 1 if (bad or tot_lockstep) else 0


if __name__ == "__main__":
    sys.exit(main())
