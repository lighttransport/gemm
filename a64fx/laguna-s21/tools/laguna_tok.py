#!/usr/bin/env python3
"""Exact byte-level BPE for Laguna S-2.1 (reads tokenizer.json; no deps, py3.6 ok).

Implements the checkpoint's Split + ByteLevel + BPE pipeline.  Python's stdlib
``re`` lacks \\p{L}/\\p{N}, so Unicode properties are scanned with
``unicodedata.category`` instead of approximating them as ASCII.

Usage:
  python3 laguna_tok.py encode "The capital of France is" [--bos] > prompt.ids
  python3 laguna_tok.py encode-file prompt.txt > prompt.ids
  python3 laguna_tok.py decode "12 34 56"
  python3 laguna_tok.py decode-file gen.ids
Env: LAGUNA_TOKENIZER (default ~/models/laguna-s21-int4/tokenizer.json)
"""
import sys, os, json, re, unicodedata

TOKJSON = os.environ.get("LAGUNA_TOKENIZER",
                         os.path.expanduser("~/models/laguna-s21-int4/tokenizer.json"))
CHAT_TEMPLATE = os.environ.get("LAGUNA_CHAT_TEMPLATE",
                               os.path.join(os.path.dirname(TOKJSON), "chat_template.jinja"))
EOS_ID = 2  # 〈|EOS|〉, also bos per config.json


def render_chat(messages, add_generation_prompt=True, enable_thinking=True, tools=None):
    """Render the checkpoint's own chat_template.jinja.

    Uses the shipped template rather than reimplementing it, so it cannot drift.
    {% generation %}/{% endgeneration %} are a HuggingFace-only extension used to
    mark assistant spans for loss masking; they emit no text, so stripping them
    leaves the rendered output identical while letting plain jinja2 parse it.
    """
    try:
        import jinja2
    except ImportError:
        sys.exit("chat mode needs jinja2 (pip install --user jinja2), or pass "
                 "pre-rendered text to `encode`")
    src = open(CHAT_TEMPLATE).read()
    src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", src)
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    return env.from_string(src).render(messages=messages, tools=tools,
                                       add_generation_prompt=add_generation_prompt,
                                       enable_thinking=enable_thinking)

def bytes_to_unicode():
    bs = list(range(ord("!"),ord("~")+1))+list(range(ord("\xa1"),ord("\xac")+1))+list(range(ord("\xae"),ord("\xff")+1))
    cs = bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return {b:chr(c) for b,c in zip(bs,cs)}

B2U = bytes_to_unicode()
U2B = {v:k for k,v in B2U.items()}

def _cat(ch): return unicodedata.category(ch)[0]
def _is_l(ch): return _cat(ch) == "L"
def _is_n(ch): return _cat(ch) == "N"
def _is_nl(ch): return ch in "\r\n"

def _main_match(s, i):
    """Match tokenizer.json's main Split-regex alternative at ``i``."""
    n, c = len(s), s[i]
    # (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if c == "'":
        tail = s[i:].lower()
        for contraction in ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d"):
            if tail.startswith(contraction): return len(contraction)
    # [^\r\n\p{L}\p{N}]?\p{L}+
    j = i
    if not (_is_nl(c) or _is_l(c) or _is_n(c)):
        j = i + 1 if i + 1 < n and _is_l(s[i + 1]) else i
    if j < n and _is_l(s[j]):
        k = j + 1
        while k < n and _is_l(s[k]): k += 1
        return k - i
    # \p{N}: one numeric code point, deliberately not a run.
    if _is_n(c): return 1
    #  ?[^\s\p{L}\p{N}]+[\r\n]*
    j = i + 1 if c == " " and i + 1 < n and not (
        s[i + 1].isspace() or _is_l(s[i + 1]) or _is_n(s[i + 1])) else i
    if j < n and not (s[j].isspace() or _is_l(s[j]) or _is_n(s[j])):
        k = j + 1
        while k < n and not (s[k].isspace() or _is_l(s[k]) or _is_n(s[k])): k += 1
        while k < n and _is_nl(s[k]): k += 1
        return k - i
    # \s*[\r\n]+ | \s+(?!\S) | \s+
    if c.isspace():
        k = i
        while k < n and s[k].isspace(): k += 1
        last_nl = -1
        for p in range(i, k):
            if _is_nl(s[p]): last_nl = p
        if last_nl >= 0: return last_nl + 1 - i
        if k == n: return k - i
        return k - i - 1 if k - i >= 2 else 1
    return 0

def _split_main(text):
    pieces=[]; i=gap=0
    while i < len(text):
        size = _main_match(text, i)
        if size:
            if i > gap: pieces.append(text[gap:i])
            pieces.append(text[i:i+size]); i += size; gap = i
        else: i += 1
    if gap < len(text): pieces.append(text[gap:])
    return pieces

_NEWLINE_RUN = re.compile(r"(?:\r?\n)+(?!\r?\n)")

def _split_isolated(text):
    """Apply the checkpoint's MergedWithNext Split, then its Isolated Split."""
    first=[]; end=0; pending=""
    for match in _NEWLINE_RUN.finditer(text):
        before=text[end:match.start()]
        if before:
            first.append(pending+before); pending=""
        pending += match.group(0)
        end=match.end()
    tail=text[end:]
    if tail or pending: first.append(pending+tail)
    pieces=[]
    for part in first or ([text] if text else []):
        pieces.extend(_split_main(part))
    return pieces

class Tok:
    def __init__(self, path):
        j = json.load(open(path)); m = j["model"]
        if m.get("type") != "BPE" or m.get("byte_fallback", False):
            raise ValueError("Laguna tokenizer must be BPE without byte_fallback")
        pts = j.get("pre_tokenizer", {}).get("pretokenizers", [])
        if len(pts) != 3 or pts[2].get("type") != "ByteLevel" or pts[2].get("add_prefix_space"):
            raise ValueError("unsupported Laguna pre-tokenizer shape")
        self.vocab = m["vocab"]
        self.id2tok = {v:k for k,v in self.vocab.items()}
        mg = [tuple(x.split(" ")) if isinstance(x,str) else tuple(x) for x in m["merges"]]
        self.ranks = {p:i for i,p in enumerate(mg)}
        self._cache = {}
        self.added_ids = set(); self.special_ids = set(); added = []
        for a in j.get("added_tokens",[]):
            self.added_ids.add(a["id"])
            if a.get("special"): self.special_ids.add(a["id"])
            self.vocab.setdefault(a["content"], a["id"]); self.id2tok[a["id"]]=a["content"]
            added.append(a["content"])
        # longest-first so 〈|EOS|〉 wins over its substrings 〈| and |〉
        self.added_re = (re.compile("(" + "|".join(re.escape(c) for c in
                         sorted(added, key=len, reverse=True)) + ")") if added else None)
    def _bpe(self, tokens):
        key = "".join(tokens)
        if key in self._cache: return self._cache[key]
        while len(tokens) > 1:
            best=None; pair=None
            for i in range(len(tokens)-1):
                r=self.ranks.get((tokens[i],tokens[i+1]))
                if r is not None and (best is None or r<best):
                    best=r; pair=(tokens[i],tokens[i+1])
            if pair is None: break
            out=[]; i=0
            while i < len(tokens):
                if i+1 < len(tokens) and (tokens[i],tokens[i+1]) == pair:
                    out.append(tokens[i]+tokens[i+1]); i += 2
                else: out.append(tokens[i]); i += 1
            tokens=out
        self._cache[key]=tokens
        return tokens
    def encode(self, text, add_bos=False):
        ids=[EOS_ID] if add_bos else []
        if self.added_re is None:
            self._encode_text(text, ids); return ids
        for i, part in enumerate(self.added_re.split(text)):
            if not part: continue
            if i % 2:                      # odd chunks are the captured added tokens
                ids.append(self.vocab[part])
            else:
                self._encode_text(part, ids)
        return ids
    def _encode_text(self, text, ids):
        for piece in _split_isolated(text):
            s="".join(B2U[b] for b in piece.encode("utf-8"))
            for t in self._bpe(list(s)):
                if t not in self.vocab:
                    raise ValueError("BPE symbol missing from vocabulary: %r" % t)
                ids.append(self.vocab[t])
    def decode(self, ids, raw=False):
        # BPE pieces are byte-level-encoded (each char stands for a byte) but added
        # tokens are literal text, so they cannot go through the same byte decode --
        # 〈|EOS|〉 would be mangled.  Flush the byte buffer around each added token.
        out=[]; buf=[]
        def flush():
            if buf:
                bs=bytes(U2B.get(ch, ord(ch)&0xff) for ch in "".join(buf))
                out.append(bs.decode("utf-8", errors="replace")); del buf[:]
        for i in ids:
            if not raw and i in self.special_ids: continue
            if i in self.added_ids:
                flush(); out.append(self.id2tok.get(i,"")); continue
            t=self.id2tok.get(i)
            if t is not None: buf.append(t)
        flush()
        return "".join(out)

def _opt(argv, name, default=None):
    return argv[argv.index(name)+1] if name in argv and argv.index(name)+1 < len(argv) else default

def main():
    if len(sys.argv)<2: print(__doc__); sys.exit(1)
    cmd=sys.argv[1]; argv=sys.argv; t=Tok(TOKJSON)
    if cmd=="encode":
        print(" ".join(str(i) for i in t.encode(argv[2], "--bos" in argv)))
    elif cmd=="encode-file":
        print(" ".join(str(i) for i in t.encode(open(argv[2]).read(), "--bos" in argv)))
    elif cmd in ("chat","chat-file"):
        user = open(argv[2]).read() if cmd=="chat-file" else argv[2]
        msgs = []
        sysmsg = _opt(argv,"--system")
        if sysmsg is not None: msgs.append({"role":"system","content":sysmsg})
        msgs.append({"role":"user","content":user})
        text = render_chat(msgs, add_generation_prompt=True,
                           enable_thinking=("--no-think" not in argv))
        if "--show-prompt" in argv: sys.stderr.write(text+"\n")
        # the template emits BOS itself -- do not add another
        print(" ".join(str(i) for i in t.encode(text)))
    elif cmd=="decode":
        print(t.decode([int(x) for x in argv[2].split()], "--raw" in argv))
    elif cmd=="decode-file":
        print(t.decode([int(x) for x in open(argv[2]).read().split()], "--raw" in argv))
    else: print(__doc__); sys.exit(1)

if __name__=="__main__": main()
