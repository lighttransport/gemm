#!/usr/bin/env python3
"""Run ONLY inside the network-disabled evaluation sandbox.

Usage: python test_glm53f_task_queue.py /source.cpp /work/report.json
Compilation artifacts stay in the sandbox's /local filesystem.
"""
import collections
import json
from pathlib import Path
import random
import subprocess
import sys

source, report = sys.argv[1:]
try:
    build = subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-pedantic", source, "-o", "/local/task_queue"],
                           capture_output=True, text=True, timeout=60)
except subprocess.TimeoutExpired:
    Path(report).write_text(json.dumps({"pass": False, "error": "compile timeout"}) + "\n")
    raise SystemExit(1)
result = {"compile_returncode": build.returncode, "compiler_stderr": build.stderr,
          "seed": 5303, "commands": 0, "pass": False}
if build.returncode == 0:
    rng = random.Random(5303)
    queue = collections.OrderedDict()
    lines, expected = [], []
    invalid = ["ADD 0 zero", "ADD +1 x", "ADD -1 x", "ADD 4294967296 x",
               "ADD 1.0 x", "ADD 0x1 x", "ADD", "ADD 9", "CANCEL", "CANCEL 0",
               "CANCEL 1 x", "ADD \t ", "CANCEL \t", "ADD 99999999999999999999999999 x",
               "POP x", "PEEK x", "COUNT x", "QUIT x", "pop", "UNKNOWN"]
    for i in range(10000):
        op = rng.randrange(7)
        ident = rng.choice([1, 4294967295, rng.randrange(1, 201)])
        if op < 2:
            payload = rng.choice(["", "word", "two words", "x  y", "tabs\tinside", "tail "])
            text_id = str(ident).zfill(rng.choice([1, 12]))
            lines.append(f"ADD {text_id} {payload}")
            if ident in queue:
                expected.append("ERR duplicate")
            else:
                queue[ident] = payload
                expected.append("OK")
        elif op in (2, 3):
            lines.append("POP" if op == 2 else "PEEK\t ")
            if not queue:
                expected.append("ERR empty")
            else:
                first = next(iter(queue))
                expected.append(f"OK {first} {queue[first]}")
                if op == 2:
                    del queue[first]
        elif op == 4:
            lines.append(f"CANCEL \t{ident}\t ")
            if ident in queue:
                del queue[ident]
                expected.append(f"OK cancelled {ident}")
            else:
                expected.append("ERR missing")
        elif op == 5:
            lines.append("COUNT ")
            expected.append(f"OK {len(queue)}")
        else:
            lines.append(rng.choice(invalid))
            expected.append("ERR invalid")
    lines += ["QUIT", "COUNT", "ADD 1 ignored"]
    expected.append("OK bye")
    try:
        run = subprocess.run(["/local/task_queue"], input="\n\n" + "\n".join(lines) + "\n",
                             capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        result["error"] = "execution timeout"
        Path(report).write_text(json.dumps(result) + "\n")
        raise SystemExit(1)
    actual = run.stdout.splitlines()
    result.update(commands=len(expected), returncode=run.returncode, stderr=run.stderr)
    # ADD success wording is absent from the task specification. Accept either
    # conventional spelling, but enforce exactly one response and all state effects.
    for i, want in enumerate(expected):
        if want == "OK" and i < len(actual) and actual[i].startswith("OK "):
            expected[i] = actual[i]
    result["pass"] = run.returncode == 0 and actual == expected
    for i in range(max(len(actual), len(expected))):
        got = actual[i] if i < len(actual) else None
        want = expected[i] if i < len(expected) else None
        if got != want:
            result["first_mismatch"] = {"index": i, "command": lines[i] if i < len(lines) else None,
                                         "expected": want, "actual": got}
            break
Path(report).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items()
                  if key not in ("compiler_stderr", "stderr")}))
raise SystemExit(0 if result["pass"] else 1)
