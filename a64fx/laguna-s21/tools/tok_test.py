#!/usr/bin/env python3
"""Tests for laguna_tok: added-token handling, decode round-trip, chat rendering.

Run: LAGUNA_TOKENIZER=~/models/laguna-s21-fp8/tokenizer.json python3 tok_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laguna_tok as L  # noqa: E402

FAILS = []


def check(ok, what, detail=""):
    print("  %-56s %s%s" % (what, "OK" if ok else "** FAIL **",
                            ("  " + detail) if detail else ""))
    if not ok:
        FAILS.append(what)


def main():
    t = L.Tok(L.TOKJSON)

    print("added tokens map to ids, not word pieces:")
    for s, tid in [("〈|EOS|〉", 2), ("<think>", 18), ("</think>", 19),
                   ("<assistant>", 23), ("</assistant>", 24)]:
        ids = t.encode(s)
        check(ids == [tid], "%-14r -> [%d]" % (s, tid), str(ids))

    print("longest-first matching (〈|EOS|〉 must beat its substrings 〈| and |〉):")
    check(t.encode("〈|EOS|〉") == [2], "〈|EOS|〉 is one token")

    print("pre-tokenizer Unicode and newline boundaries:")
    check(L._split_isolated("café 日本語 42") ==
          ["café", " 日本語", " ", "4", "2"],
          "Unicode letters and one-codepoint numeric alternatives")
    check(L._split_isolated("a \nb") == ["a", " ", "\n", "b"],
          "MergedWithNext newline split precedes main regex")

    print("round-trip:")
    cases = ["plain ascii text",
             "unicode: café naïve 日本語 — em dash",
             "〈|EOS|〉<system>S</system>\n<user>U</user>\n<assistant><think>",
             "mixed <think>reasoning</think>answer</assistant>",
             ""]
    for s in cases:
        got = t.decode(t.encode(s), raw=True)
        check(got == s, "exact round-trip %r" % (s[:34],),
              "" if got == s else "got %r" % (got[:40],))

    print("decode visibility:")
    ids = t.encode("〈|EOS|〉<assistant><think>hi</think>bye</assistant>")
    vis, raw = t.decode(ids), t.decode(ids, raw=True)
    check("〈|EOS|〉" not in vis, "special=True tokens hidden by default", repr(vis[:40]))
    check("<think>" in vis and "</assistant>" in vis,
          "special=False markup stays visible")
    check("〈|EOS|〉" in raw, "--raw shows everything")

    print("chat template:")
    try:
        txt = L.render_chat([{"role": "user", "content": "Q?"}],
                            add_generation_prompt=True, enable_thinking=True)
        check(txt.startswith("〈|EOS|〉"), "emits BOS itself (so no --bos)")
        check("<user>Q?</user>" in txt, "user turn wrapped")
        check(txt.endswith("<assistant><think>"), "generation prompt opens thinking")
        ids = t.encode(txt)
        check(ids[0] == 2, "first id is BOS", str(ids[:3]))
        check(ids[-2:] == [23, 18], "ends <assistant><think>", str(ids[-2:]))

        nt = L.render_chat([{"role": "user", "content": "Q?"}],
                           add_generation_prompt=True, enable_thinking=False)
        check(nt.endswith("<assistant></think>"), "--no-think closes thinking at once")
        check(t.encode(nt)[-2:] == [23, 19], "ends <assistant></think>")

        sysd = L.render_chat([{"role": "system", "content": "Custom sys."},
                              {"role": "user", "content": "Q?"}],
                             add_generation_prompt=True)
        check("<system>Custom sys.</system>" in sysd, "custom system message honoured")
        check("Poolside" not in sysd, "custom system replaces the default")

        multi = L.render_chat([{"role": "user", "content": "A"},
                               {"role": "assistant", "content": "B"},
                               {"role": "user", "content": "C"}],
                              add_generation_prompt=True)
        check(multi.count("<user>") == 2 and "<assistant>" in multi,
              "multi-turn history renders")
    except SystemExit as e:
        check(False, "chat template renders", str(e))

    print("FAIL" if FAILS else "tokenizer: PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
