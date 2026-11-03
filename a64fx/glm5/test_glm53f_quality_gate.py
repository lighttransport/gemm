#!/usr/bin/env python3
"""Local unit tests for paired quality-gate accounting; no model code runs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
(root / "tmp").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="glm53f-gate-", dir=root / "tmp") as directory:
    d = Path(directory)
    keys = [f"HumanEval/{i}" for i in range(164)]
    (d / "metadata.json").write_text(json.dumps({"sets": {"humaneval164": {"ids": keys}}}))
    def results(passes):
        return {"hash": "same-dataset", "eval": {
            key: [{"base_status": "pass" if i < passes else "fail", "plus_status": "pass"}]
            for i, key in enumerate(keys)}}
    (d / "baseline.json").write_text(json.dumps(results(140)))
    for name, result, success in [("loss8", results(132), True), ("loss9", results(131), False),
                                  ("gain", results(145), True)]:
        (d / "candidate.json").write_text(json.dumps(result))
        run = subprocess.run([sys.executable, str(Path(__file__).with_name("check_glm53f_quality_gate.py")),
                              str(d / "baseline.json"), str(d / "candidate.json"),
                              "--metadata", str(d / "metadata.json"), "--set", "humaneval164"],
                             capture_output=True, text=True)
        assert (run.returncode == 0) == success, (name, run.stdout, run.stderr)
        print(name, "PASS")
    result = results(140)
    del result["eval"][keys[0]]
    (d / "candidate.json").write_text(json.dumps(result))
    run = subprocess.run([sys.executable, str(Path(__file__).with_name("check_glm53f_quality_gate.py")),
                          str(d / "baseline.json"), str(d / "candidate.json"),
                          "--metadata", str(d / "metadata.json"), "--set", "humaneval164"],
                         capture_output=True, text=True)
    assert run.returncode != 0 and "Missing or extra tasks" in run.stderr
    print("missing-task rejection PASS")
