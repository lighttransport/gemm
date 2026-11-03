#!/usr/bin/env python3
"""Decode and sanitize completed fixtures; never execute generated code."""
import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from evalplus.sanitize import sanitize

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--fixtures", type=Path, required=True)
p.add_argument("--set", choices=["mbpp32", "humaneval164"], required=True)
p.add_argument("--kind", choices=["exact", "candidate"], required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
tasks = json.loads((a.fixtures / (a.set + ".json")).read_text())
tok = Tokenizer.from_file(a.tokenizer)
samples = []
for key, task in tasks.items():
    path = a.fixtures / f"{a.set}-{key.split('/')[-1]}.{a.kind}.out"
    ids = [int(x) for x in path.read_text().split()]
    if not ids:
        raise ValueError(f"Empty output: {path}")
    text = tok.decode(ids, skip_special_tokens=True)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    code = sanitize(text, task["entry_point"])
    samples.append({"task_id": key, "solution": code})
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text("".join(json.dumps(s) + "\n" for s in samples))
(a.out.parent / "problems.jsonl").write_text("".join(json.dumps(task) + "\n" for task in tasks.values()))
print(f"Prepared {len(samples)} samples; generated code was not executed.")
