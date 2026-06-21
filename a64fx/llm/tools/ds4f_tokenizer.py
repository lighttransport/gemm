#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure-Python byte-level BPE tokenizer for DeepSeek-V4-Flash (DS4F).

Stdlib only (Python 3.6+). Reimplements the exact tokenizer.json pipeline:
  normalizer (none) -> pre_tokenizer (Split x3 + ByteLevel) -> BPE(merges) .

The Fugaku A64FX nodes have no `tokenizers`/`transformers`/`regex` modules, and
Python's stdlib `re` cannot do Unicode property classes (\\p{L}, \\p{N}, ...),
so the pre_tokenizer regex is hand-scanned with `unicodedata.category`.

This is specific to $HOME/models/ds4f/tokenizer.json (model.type == "BPE",
ByteLevel, byte_fallback == False). It asserts that shape on load.

CLI (file handoff for the C EP runner):
  encode --prompt-file P [--out prompt_ids.txt] [--no-bos]
  decode --ids-file gen_ids.txt   (or --ids "12 34 56")
  selftest                        (round-trip + invariants)
"""
import sys, os, json, argparse, unicodedata

DEFAULT_TOK = os.path.join(os.environ.get("DS4F_MODEL_DIR",
                           os.path.join(os.environ["HOME"], "models", "ds4f")),
                           "tokenizer.json")
BOS_ID = 0
EOS_ID = 1

# ---- pre_tokenizer constants (hardcoded from tokenizer.json; asserted on load) ----
# split3 alt-a leading punctuation set: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
_ASCII_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_ASCII_LETTER = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

def _is_cjk(ch):
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FA5 or   # CJK unified ideographs 一-龥
            0x3040 <= o <= 0x309F or   # Hiragana ぀-ゟ
            0x30A0 <= o <= 0x30FF)     # Katakana ゠-ヿ

def _cat(ch):
    return unicodedata.category(ch)[0]   # L, N, P, S, M, Z, C, ...

def _is_N(ch): return _cat(ch) == 'N'
def _is_L(ch): return _cat(ch) == 'L'
def _is_M(ch): return _cat(ch) == 'M'
def _is_P(ch): return _cat(ch) == 'P'
def _is_S(ch): return _cat(ch) == 'S'
def _is_space(ch): return ch.isspace()
def _is_nl(ch): return ch in '\r\n'


# ---------------------------------------------------------------------------
# Split #1 :  \p{N}{1,3}   (Isolated)   -> isolate runs of 1..3 digits
# ---------------------------------------------------------------------------
def _m_digits(s, i):
    j = i
    while j < len(s) and j - i < 3 and _is_N(s[j]):
        j += 1
    return j - i

# ---------------------------------------------------------------------------
# Split #2 :  [一-龥぀-ゟ゠-ヿ]+   (Isolated)  -> isolate CJK runs
# ---------------------------------------------------------------------------
def _m_cjk(s, i):
    j = i
    while j < len(s) and _is_cjk(s[j]):
        j += 1
    return j - i

# ---------------------------------------------------------------------------
# Split #3 : the GPT-2-style alternation (Isolated). Ordered alternatives:
#   a) [PUNCT][A-Za-z]+
#   b) [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
#   c)  ?[\p{P}\p{S}]+[\r\n]*
#   d) \s*[\r\n]+
#   e) \s+(?!\S)
#   f) \s+
# Returns length of the first alternative that matches at i, else 0.
# ---------------------------------------------------------------------------
def _m_word(s, i):
    n = len(s)
    c = s[i]
    # a) ascii punct followed by >=1 ascii letter
    if c in _ASCII_PUNCT and i + 1 < n and s[i + 1] in _ASCII_LETTER:
        j = i + 1
        while j < n and s[j] in _ASCII_LETTER:
            j += 1
        return j - i
    # b) optional non-(\r\n L P S) prefix, then >=1 (L|M)
    j = i
    if not (_is_nl(c) or _is_L(c) or _is_P(c) or _is_S(c)):
        # prefix consumes exactly one char, but only if a letter/mark follows
        if i + 1 < n and (_is_L(s[i + 1]) or _is_M(s[i + 1])):
            j = i + 1
            while j < n and (_is_L(s[j]) or _is_M(s[j])):
                j += 1
            return j - i
        # else fall through (prefix optional -> try without prefix below)
    if _is_L(c) or _is_M(c):
        j = i
        while j < n and (_is_L(s[j]) or _is_M(s[j])):
            j += 1
        return j - i
    # c)  ?[\p{P}\p{S}]+[\r\n]*
    j = i
    if c == ' ' and i + 1 < n and (_is_P(s[i + 1]) or _is_S(s[i + 1])):
        j = i + 1
    if j < n and (_is_P(s[j]) or _is_S(s[j])):
        k = j
        while k < n and (_is_P(s[k]) or _is_S(s[k])):
            k += 1
        while k < n and _is_nl(s[k]):
            k += 1
        return k - i
    # d) \s*[\r\n]+  : maximal whitespace run ending at a newline
    if _is_space(c):
        k = i
        while k < n and _is_space(s[k]):
            k += 1
        last_nl = -1
        for t in range(i, k):
            if _is_nl(s[t]):
                last_nl = t
        if last_nl >= 0:
            return last_nl + 1 - i
        # e) \s+(?!\S): whitespace run, drop last char unless run hits EOF
        if k == n:
            return k - i
        if k - i >= 2:
            return k - i - 1
        # f) \s+ : single whitespace char
        return 1
    return 0


def _split_isolated(pieces, matchfn):
    out = []
    for s in pieces:
        i = 0
        n = len(s)
        gap = 0
        while i < n:
            L = matchfn(s, i)
            if L > 0:
                if i > gap:
                    out.append(s[gap:i])
                out.append(s[i:i + L])
                i += L
                gap = i
            else:
                i += 1
        if gap < n:
            out.append(s[gap:])
    return out


def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class DS4FTokenizer(object):
    def __init__(self, path=DEFAULT_TOK):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        m = d["model"]
        assert m["type"] == "BPE", "expected BPE model"
        assert not m.get("byte_fallback", False), "byte_fallback not supported"
        self.vocab = m["vocab"]                       # str -> id
        self.id_to_tok = {i: t for t, i in self.vocab.items()}
        # added_tokens (specials) — id -> content and content -> id
        self.added = {}
        for t in d.get("added_tokens", []):
            self.added[t["content"]] = t["id"]
            self.id_to_tok.setdefault(t["id"], t["content"])
        # BPE ranks
        self.ranks = {}
        for r, pair in enumerate(m["merges"]):
            a, b = pair.split(" ") if isinstance(pair, str) else pair
            self.ranks[(a, b)] = r
        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self._cache = {}
        # sanity: every single byte-unicode char must be a known vocab symbol
        miss = [u for u in self.b2u.values() if u not in self.vocab]
        assert not miss, "byte-unicode chars missing from vocab: %r" % miss[:8]

    # ---- pretokenize: text -> list of byte-level-unicode word strings ----
    def _pretokenize(self, text):
        pieces = _split_isolated([text], _m_digits)
        pieces = _split_isolated(pieces, _m_cjk)
        pieces = _split_isolated(pieces, _m_word)
        out = []
        for p in pieces:
            if not p:
                continue
            out.append("".join(self.b2u[b] for b in p.encode("utf-8")))
        return out

    def _bpe(self, token):
        if token in self._cache:
            return self._cache[token]
        word = list(token)
        if len(word) < 2:
            self._cache[token] = word
            return word
        while True:
            best = None
            best_rank = None
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best = (word[i], word[i + 1])
            if best is None:
                break
            a, b = best
            new = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = new
            if len(word) == 1:
                break
        self._cache[token] = word
        return word

    def encode(self, text, add_bos=True):
        ids = [BOS_ID] if add_bos else []
        for tok in self._pretokenize(text):
            for sym in self._bpe(tok):
                ids.append(self.vocab[sym])
        return ids

    def decode(self, ids, skip_special=True, stop_at_eos=True):
        buf = []
        for i in ids:
            if i == EOS_ID and stop_at_eos:
                break
            tok = self.id_to_tok.get(i)
            if tok is None:
                continue
            if i in (BOS_ID, EOS_ID) or i >= 128000:   # specials / placeholders
                if skip_special:
                    continue
            buf.append(tok)
        s = "".join(buf)
        # reverse ByteLevel: each char -> original byte
        try:
            data = bytes(self.u2b[ch] for ch in s if ch in self.u2b)
        except Exception:
            data = b""
        return data.decode("utf-8", errors="replace")


def _selftest(tok):
    samples = [
        "def add(a, b):\n    return a + b\n",
        "Write a Python function to compute the n-th Fibonacci number.",
        "x = [1, 2, 3]; print(x[0] + 42)  # comment",
        "  leading   spaces and\ttabs\n\n",
        "CJK: 日本語 と 漢字 の test 123456",
        "for i in range(10):\n\tprint(i**2)\n",
    ]
    ok = True
    for s in samples:
        ids = tok.encode(s, add_bos=False)
        back = tok.decode(ids, skip_special=True, stop_at_eos=False)
        match = (back == s)
        ok = ok and match
        print("[%s] %d toks  round-trip=%s" % ("OK" if match else "XX", len(ids), match))
        if not match:
            print("   in : %r" % s)
            print("   out: %r" % back)
    # special-token invariants
    assert tok.id_to_tok[0] == "<｜begin▁of▁sentence｜>"
    assert tok.id_to_tok[1] == "<｜end▁of▁sentence｜>"
    print("specials: bos=0 eos=1 OK")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "decode", "selftest"])
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--prompt-file")
    ap.add_argument("--prompt")
    ap.add_argument("--ids-file")
    ap.add_argument("--ids")
    ap.add_argument("--out")
    ap.add_argument("--no-bos", action="store_true")
    a = ap.parse_args()
    tok = DS4FTokenizer(a.tokenizer)

    if a.cmd == "selftest":
        return _selftest(tok)

    if a.cmd == "encode":
        if a.prompt_file:
            with open(a.prompt_file, "r", encoding="utf-8") as f:
                text = f.read()
        elif a.prompt is not None:
            text = a.prompt
        else:
            text = sys.stdin.read()
        ids = tok.encode(text, add_bos=not a.no_bos)
        line = " ".join(str(i) for i in ids)
        if a.out:
            with open(a.out, "w") as f:
                f.write(line + "\n")
            sys.stderr.write("wrote %d ids -> %s\n" % (len(ids), a.out))
        else:
            print(line)
        return 0

    if a.cmd == "decode":
        if a.ids_file:
            with open(a.ids_file) as f:
                raw = f.read()
        elif a.ids is not None:
            raw = a.ids
        else:
            raw = sys.stdin.read()
        ids = [int(x) for x in raw.replace(",", " ").split()]
        sys.stdout.write(tok.decode(ids))
        return 0


if __name__ == "__main__":
    sys.exit(main())
