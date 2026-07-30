#!/usr/bin/env python3
"""Create deterministic 16K/32K Laguna retrieval-plus-C++ test prompts."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import laguna_tok


SYSTEM = (
    "You are a meticulous senior software engineer. Output the final answer only. "
    "Use all binding requirements from the supplied context, ignore non-binding "
    "records, and audit headers, constants, assertions, and exact output."
)


def user_text(nrecords):
    needles = {
        nrecords // 8: "BINDING REQUIREMENT ALPHA: the integer constant alpha is 314159.",
        nrecords // 2: "BINDING REQUIREMENT BETA: the string constant beta is cobalt-orchid.",
        (7 * nrecords) // 8: "BINDING REQUIREMENT GAMMA: the integer constant gamma is 271828.",
    }
    rows = [
        "The following audit records contain exactly three binding requirements. "
        "All ordinary records are non-binding filler.\n"
    ]
    for i in range(nrecords):
        rows.append(needles.get(
            i, "Record %05d: module sigma reports stable status; this ordinary line is non-binding."
            % i))
    rows.append(
        "\nTASK: Recover all three binding requirements. Return one complete fenced C++20 "
        "program and no other fenced blocks. Define alpha, beta, and gamma with the "
        "recovered values. In main, assert alpha + gamma == 585987, assert beta == "
        "\"cobalt-orchid\", and print exactly: 314159 cobalt-orchid 271828 followed "
        "by a newline. Include every directly required standard header."
    )
    return "\n".join(rows)


def render_ids(tok, nrecords):
    prompt = laguna_tok.render_chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user_text(nrecords)}],
        add_generation_prompt=True, enable_thinking=False)
    return tok.encode(prompt)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--metadata")
    p.add_argument("--tokenizer", default=laguna_tok.TOKJSON)
    args = p.parse_args(argv)
    tok = laguna_tok.Tok(args.tokenizer)

    lo, hi = 1, max(2, args.target // 4)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(render_ids(tok, mid)) <= args.target:
            lo = mid
        else:
            hi = mid - 1
    ids = render_ids(tok, lo)
    with open(args.out, "w") as f:
        f.write(" ".join(str(x) for x in ids) + "\n")
    meta = {"target_tokens": args.target, "actual_tokens": len(ids),
            "records": lo, "needle_records": [lo // 8, lo // 2, (7 * lo) // 8],
            "alpha": 314159, "beta": "cobalt-orchid", "gamma": 271828}
    if args.metadata:
        with open(args.metadata, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
            f.write("\n")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
