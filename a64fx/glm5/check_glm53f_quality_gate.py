#!/usr/bin/env python3
"""Compare paired EvalPlus outcomes; missing or duplicated tasks are errors."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("baseline", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("--metadata", type=Path, required=True)
p.add_argument("--set", choices=["mbpp32", "humaneval164"], required=True)
a = p.parse_args()
baseline, candidate = (json.loads(path.read_text()) for path in (a.baseline, a.candidate))
expected = set(json.loads(a.metadata.read_text())["sets"][a.set]["ids"])
if baseline["hash"] != candidate["hash"]:
    raise ValueError("Unpaired dataset hashes")
if len(expected) != (32 if a.set == "mbpp32" else 164):
    raise ValueError("Unexpected task count")
passed = []
for result in (baseline, candidate):
    if set(result["eval"]) != expected:
        raise ValueError("Missing or extra tasks")
    if any(len(rows) != 1 for rows in result["eval"].values()):
        raise ValueError("Expected exactly one greedy sample per task")
    passed.append({key for key, rows in result["eval"].items()
                   if rows[0]["base_status"] == "pass" and rows[0]["plus_status"] == "pass"})
loss = len(passed[0]) - len(passed[1])
allowed = len(expected) * 5 // 100
summary = {"set": a.set, "tasks": len(expected), "baseline_pass": len(passed[0]),
           "candidate_pass": len(passed[1]), "net_lost": loss, "max_net_lost": allowed,
           "lost_tasks": sorted(passed[0] - passed[1]), "gained_tasks": sorted(passed[1] - passed[0]),
           "quality_pass": loss <= allowed, "speed_gate": "not evaluated here"}
print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary["quality_pass"] else 1)
