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
EOS_ID = 2  # 〈|EOS|〉, also bos per config.json

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
        self.added_ids = set()
        for a in j.get("added_tokens",[]):
            self.added_ids.add(a["id"])
            self.vocab.setdefault(a["content"], a["id"]); self.id2tok[a["id"]]=a["content"]
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
        for piece in self.pat.findall(text):
            s="".join(B2U[b] for b in piece.encode("utf-8"))
            for t in self._bpe(list(s)):
                if t in self.vocab: ids.append(self.vocab[t])
                else:
                    for ch in t: ids.append(self.vocab.get(ch, 0))
        return ids
    def decode(self, ids):
        out=[]
        for i in ids:
            if i in self.added_ids: continue
            t=self.id2tok.get(i)
            if t is not None: out.append(t)
        bs=bytes(U2B.get(ch, ord(ch)&0xff) for ch in "".join(out))
        return bs.decode("utf-8", errors="replace")

def main():
    if len(sys.argv)<2: print(__doc__); sys.exit(1)
    cmd=sys.argv[1]; t=Tok(TOKJSON)
    if cmd=="encode":
        print(" ".join(str(i) for i in t.encode(sys.argv[2], "--bos" in sys.argv)))
    elif cmd=="encode-file":
        print(" ".join(str(i) for i in t.encode(open(sys.argv[2]).read(), "--bos" in sys.argv)))
    elif cmd=="decode":
        print(t.decode([int(x) for x in sys.argv[2].split()]))
    elif cmd=="decode-file":
        print(t.decode([int(x) for x in open(sys.argv[2]).read().split()]))
    else: print(__doc__); sys.exit(1)

if __name__=="__main__": main()
