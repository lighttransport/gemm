#!/usr/bin/env python3
"""Compile/test a Laguna C++ answer and build one compiler-feedback repair turn.

This intentionally executes generated code only when ``--run`` is present. Use
that option only for trusted prompts in an isolated job environment.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import laguna_tok


FENCE = re.compile(r"```(?:cpp|c\+\+|cc)\s*\n(.*?)```", re.I | re.S)


def read_ids(path):
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def extract_cpp(text):
    match = FENCE.search(text)
    if not match:
        raise ValueError("no fenced C++ code block found")
    return match.group(1)


def clipped(text, limit=12000):
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[diagnostic truncated]"


def check_cpp(code, compiler, standard, run, timeout):
    result = {"compile_ok": False, "run_ok": None, "diagnostic": ""}
    with tempfile.TemporaryDirectory(prefix="laguna-cpp-") as tmp:
        src = os.path.join(tmp, "answer.cpp")
        exe = os.path.join(tmp, "answer")
        with open(src, "w") as f:
            f.write(code)
        cmd = [compiler, "-x", "c++", "-std=" + standard,
               "-Wall", "-Wextra", "-Wpedantic", "-pthread", src, "-o", exe]
        try:
            cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["diagnostic"] = "compiler failed: %s" % exc
            return result
        result["compile_ok"] = cp.returncode == 0
        if not result["compile_ok"]:
            result["diagnostic"] = clipped(
                (cp.stderr or cp.stdout).replace(tmp, "<build>"))
            return result
        if not run:
            result["diagnostic"] = clipped(cp.stderr)
            return result
        try:
            rp = subprocess.run([exe], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=timeout)
            result["run_ok"] = rp.returncode == 0
            if not result["run_ok"]:
                result["diagnostic"] = clipped(
                    "exit status %d\nstdout:\n%s\nstderr:\n%s" %
                    (rp.returncode, rp.stdout, rp.stderr)).replace(tmp, "<build>")
            else:
                result["diagnostic"] = clipped(cp.stderr + rp.stdout + rp.stderr)
        except subprocess.TimeoutExpired as exc:
            result["run_ok"] = False
            result["diagnostic"] = clipped(
                "program timed out after %.1fs\nstdout:\n%s\nstderr:\n%s" %
                (timeout, exc.stdout or "", exc.stderr or ""))
    return result


def make_repair_prompt(prompt_ids, answer_ids, diagnostic, tok):
    request = (
        "</assistant>\n<user>Audit and repair the answer. The extracted C++ "
        "program failed automated validation:\n\n" + diagnostic +
        "\n\nReturn a complete corrected answer, not a patch. Re-check every "
        "direct header, API contract, synchronization predicate and matching "
        "notification, ownership and lifetime rule, edge case, and assertion."
        "</user>\n<assistant></think>"
    )
    return prompt_ids + answer_ids + tok.encode(request)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ids", help="generated token-id file")
    p.add_argument("--prompt-ids", help="original rendered prompt ids")
    p.add_argument("--repair-out", help="write a repair-turn prompt here on failure")
    p.add_argument("--tokenizer", default=laguna_tok.TOKJSON)
    p.add_argument("--compiler", default=os.environ.get("CXX", "g++"))
    p.add_argument("--std", default="c++2a",
                   help="compiler language level (default works on Fugaku's g++)")
    p.add_argument("--run", action="store_true",
                   help="execute the generated binary (trusted prompts only)")
    p.add_argument("--timeout", type=float, default=20.0)
    args = p.parse_args(argv)

    tok = laguna_tok.Tok(args.tokenizer)
    try:
        answer_ids = read_ids(args.ids)
        code = extract_cpp(tok.decode(answer_ids))
        result = check_cpp(code, args.compiler, args.std, args.run, args.timeout)
    except (OSError, ValueError) as exc:
        result = {"compile_ok": False, "run_ok": None,
                  "diagnostic": str(exc)}
        answer_ids = read_ids(args.ids) if os.path.exists(args.ids) else []

    passed = result["compile_ok"] and (not args.run or result["run_ok"])
    result["passed"] = bool(passed)
    print(json.dumps(result, indent=2, sort_keys=True))

    if not passed and args.repair_out:
        if not args.prompt_ids:
            p.error("--repair-out requires --prompt-ids")
        repair = make_repair_prompt(read_ids(args.prompt_ids), answer_ids,
                                    result["diagnostic"], tok)
        with open(args.repair_out, "w") as f:
            f.write(" ".join(str(x) for x in repair) + "\n")
        print("repair prompt: %s (%d tokens)" % (args.repair_out, len(repair)))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
