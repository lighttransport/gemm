#!/usr/bin/env python3
"""Minimal byte-level BPE for GLM-5.2 (reads tokenizer.json; no deps, py3.6 ok).

Handles the GPT-4-style byte-level BPE: ByteLevel pre-tokenizer + decoder, BPE merges.
The pretokenize regex uses \\p{L}/\\p{N} which stdlib `re` lacks, so we approximate it
for ASCII/English (good enough for prompts). Decode is exact (id -> token -> bytes).

Usage:
  python3 glm5_tokenizer.py encode "The capital of France is" [--bos]   # -> ids
  python3 glm5_tokenizer.py decode "12 34 56"                            # -> text
  python3 glm5_tokenizer.py decode-file gen_ids.txt
Env: GLM5_TOKENIZER (default ~/models/glm5.2/tokenizer.json)
"""
import sys, os, json, re

TOKJSON = os.environ.get("GLM5_TOKENIZER", os.path.expanduser("~/models/glm5.2/tokenizer.json"))

GLM5_GMASK = 154822
GLM5_SOP = 154824
GLM5_USER = 154827
GLM5_ASSISTANT = 154828
GLM5_OBSERVATION = 154829
GLM5_NO_THINK = 154842

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
        j = json.load(open(path))
        m = j["model"]
        self.vocab = m["vocab"]                      # token str -> id
        self.id2tok = {v:k for k,v in self.vocab.items()}
        mg = m["merges"]
        mg = [tuple(x.split(" ")) if isinstance(x,str) else tuple(x) for x in mg]
        self.ranks = {p:i for i,p in enumerate(mg)}
        self.added = {}                              # id -> content (special)
        self.added_ids = set()
        for a in j.get("added_tokens",[]):
            self.added[a["id"]] = a["content"]; self.added_ids.add(a["id"])
            self.vocab.setdefault(a["content"], a["id"]); self.id2tok[a["id"]]=a["content"]
        # ASCII-approx GPT-4 pretok pattern
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
        ids=[]
        if add_bos:
            b=self.vocab.get("[gMASK]") or self.vocab.get("<|begin_of_sentence|>") or GLM5_SOP
            ids.append(b)
        for piece in self.pat.findall(text):
            s="".join(B2U[b] for b in piece.encode("utf-8"))
            for t in self._bpe(list(s)):
                if t in self.vocab: ids.append(self.vocab[t])
                else:
                    for ch in t: ids.append(self.vocab.get(ch, 0))
        return ids
    def chat(self, user_text, think=False):
        """GLM-5.2 chat prefix: [gMASK, sop, <|user|>, text, <|assistant|>, </think>] by default."""
        ids=[GLM5_GMASK, GLM5_SOP, GLM5_USER]+self.encode(user_text)+[GLM5_ASSISTANT]
        if not think:
            ids.append(GLM5_NO_THINK)
        return ids
    def decode(self, ids):
        out=[]
        for i in ids:
            if i in self.added_ids: continue          # skip specials in rendered text
            t=self.id2tok.get(i)
            if t is None: continue
            out.append(t)
        s="".join(out)
        bs=bytes(U2B.get(ch, ord(ch)&0xff) for ch in s)
        return bs.decode("utf-8", errors="replace")

def main():
    if len(sys.argv)<2: print(__doc__); sys.exit(1)
    cmd=sys.argv[1]; t=Tok(TOKJSON)
    if cmd=="encode":
        text=sys.argv[2]; add_bos="--bos" in sys.argv
        print(" ".join(str(i) for i in t.encode(text, add_bos)))
    elif cmd=="chat":
        text=sys.argv[2]
        print(" ".join(str(i) for i in t.chat(text)))
    elif cmd=="chat-file":
        text=open(sys.argv[2]).read()
        print(" ".join(str(i) for i in t.chat(text)))
    elif cmd=="decode":
        ids=[int(x) for x in sys.argv[2].split()]
        print(t.decode(ids))
    elif cmd=="decode-file":
        ids=[int(x) for x in open(sys.argv[2]).read().split()]
        print(t.decode(ids))
    else: print(__doc__); sys.exit(1)

if __name__=="__main__": main()
