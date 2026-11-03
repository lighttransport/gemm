#!/usr/bin/env python3
"""Decode generated IDs as text without executing the result."""
import argparse
from pathlib import Path
from tokenizers import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("tokenizer")
p.add_argument("ids", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
tok = Tokenizer.from_file(a.tokenizer)
text = tok.decode([int(x) for x in a.ids.read_text().split()], skip_special_tokens=True)
if "</think>" in text:
    text = text.split("</think>", 1)[1]
if text.lstrip().startswith("```"):
    text = text.split("\n", 1)[1].rsplit("```", 1)[0]
a.output.write_text(text)
