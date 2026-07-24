#!/usr/bin/env python3
"""Minimal byte-level BPE for Laguna S-2.1 (reads tokenizer.json; no deps, py3.6 ok).

GPT-4-style byte-level BPE: ByteLevel pre-tokenizer + decoder, BPE merges. The
pretokenize regex uses \\p{L}/\\p{N} which stdlib `re` lacks, so we approximate it
for ASCII/English (good enough for smoke-test prompts). Decode is exact.

Usage:
  python3 laguna_tok.py encode "The capital of France is" [--bos] > prompt.ids
  python3 laguna_tok.py encode-file prompt.txt > prompt.ids
  python3 laguna_tok.py decode "12 34 56"
  python3 laguna_tok.py decode-file gen.ids
Env: LAGUNA_TOKENIZER (default ~/models/laguna-s21-int4/tokenizer.json)
"""
import sys, os, json, re

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

class Tok:
    def __init__(self, path):
        j = json.load(open(path)); m = j["model"]
        self.vocab = m["vocab"]
        self.id2tok = {v:k for k,v in self.vocab.items()}
        mg = [tuple(x.split(" ")) if isinstance(x,str) else tuple(x) for x in m["merges"]]
        self.ranks = {p:i for i,p in enumerate(mg)}
        self.added_ids = set(); self.special_ids = set(); added = []
        for a in j.get("added_tokens",[]):
            self.added_ids.add(a["id"])
            if a.get("special"): self.special_ids.add(a["id"])
            self.vocab.setdefault(a["content"], a["id"]); self.id2tok[a["id"]]=a["content"]
            added.append(a["content"])
        # longest-first so 〈|EOS|〉 wins over its substrings 〈| and |〉
        self.added_re = (re.compile("(" + "|".join(re.escape(c) for c in
                         sorted(added, key=len, reverse=True)) + ")") if added else None)
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")
    def _bpe(self, tokens):
        while len(tokens) > 1:
            best=None; bi=-1
            for i in range(len(tokens)-1):
                r=self.ranks.get((tokens[i],tokens[i+1]))
                if r is not None and (best is None or r<best): best=r; bi=i
            if bi<0: break
            tokens=tokens[:bi]+[tokens[bi]+tokens[bi+1]]+tokens[bi+2:]
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
        for piece in self.pat.findall(text):
            s="".join(B2U[b] for b in piece.encode("utf-8"))
            for t in self._bpe(list(s)):
                if t in self.vocab: ids.append(self.vocab[t])
                else:
                    for ch in t: ids.append(self.vocab.get(ch, 0))
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
