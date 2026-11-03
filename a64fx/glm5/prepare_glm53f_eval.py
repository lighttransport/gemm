#!/usr/bin/env python3
"""Pin coding fixtures and tokenize with the model's canonical tokenizer.json.

Set XDG_CACHE_HOME and TMPDIR to repository-local directories before running.
This prepares data only; it never executes generated or reference code.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from tokenizers import Tokenizer
from evalplus.data import get_mbpp_plus, get_human_eval_plus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--remote-root", required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(args.tokenizer))
    metadata = {"evalplus": importlib.metadata.version("evalplus"),
                "tokenizers": importlib.metadata.version("tokenizers"),
                "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
                "template": "[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>{prompt}<|assistant|><think></think>",
                "generation": {"greedy": True, "max_tokens": 2048}, "sets": {}}
    for name, data, count in [("mbpp32", get_mbpp_plus(), 32),
                              ("humaneval164", get_human_eval_plus(), 164)]:
        ids = sorted(data, key=lambda key: int(key.split("/")[-1]))[:count]
        selected = {key: data[key] for key in ids}
        raw = json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()
        (args.out / (name + ".json")).write_bytes(raw)
        (args.out / (name + ".jsonl")).write_text("".join(json.dumps(selected[key]) + "\n" for key in ids))
        if name == "mbpp32":
            smoke = args.out / "sandbox-smoke"
            smoke.mkdir(exist_ok=True)
            first = selected[ids[0]]
            (smoke / "problems.jsonl").write_text(json.dumps(first) + "\n")
            (smoke / "samples.jsonl").write_text(json.dumps({"task_id": ids[0], "solution": first["canonical_solution"]}) + "\n")
        metadata["sets"][name] = {"ids": ids, "sha256": hashlib.sha256(raw).hexdigest()}
        manifests = {kind: [] for kind in ("exact", "candidate")}
        for key in ids:
            task = data[key]
            stem = name + "-" + key.split("/")[-1]
            instruction = "Return only a complete Python solution in one Python code block, including the function signature and any imports.\n\n" + task["prompt"]
            text = metadata["template"].replace("{prompt}", instruction)
            tokens = tok.encode(text, add_special_tokens=False).ids
            if tok.decode(tokens, skip_special_tokens=False) != text:
                raise RuntimeError("Tokenizer round-trip mismatch: " + key)
            (args.out / (stem + ".ids")).write_text("\n".join(map(str, tokens)) + "\n")
            for kind in manifests:
                manifests[kind].append(f"{key} {args.remote_root}/{stem}.ids {args.remote_root}/{stem}.{kind}.out")
        for kind, lines in manifests.items():
            (args.out / f"{name}.{kind}.manifest").write_text("\n".join(lines) + "\n")
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    cpp_prompt = Path(__file__).parent / "prompts/glm53f_cpp_codegen_task.md"
    cpp_ids = tok.encode(cpp_prompt.read_text(), add_special_tokens=False).ids
    (args.out / "cpp-task.ids").write_text("\n".join(map(str, cpp_ids)) + "\n")
    for kind in ("exact", "candidate"):
        (args.out / f"cpp.{kind}.manifest").write_text(f"task_queue {args.remote_root}/cpp-task.ids {args.remote_root}/cpp.{kind}.out\n")
        policy = "exact" if kind == "exact" else "all"
        combined = (args.out / f"humaneval164.{kind}.manifest").read_text()
        combined += f"task_queue {args.remote_root}/cpp-task.ids {args.remote_root}/cpp.{kind}.out {policy} 8192\n"
        (args.out / f"humaneval_cpp.{kind}.manifest").write_text(combined)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
