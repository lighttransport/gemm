#!/usr/bin/env python3
"""Compile and execute a DS4F-generated C++17 completion."""

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


def extract_source(text: str) -> str:
    blocks = re.findall(r"```(?:cpp|c\+\+|c)?\s*\n?(.*?)```", text, flags=re.I | re.S)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    # A causal completion may emit a completed program and then reproduce the
    # prompt's fenced prefix. Keep the generated program before that marker.
    marker = re.search(r"```(?:cpp|c\+\+|c)?\s*", text, flags=re.I)
    if marker and marker.start() > 0:
        text = text[:marker.start()]
    if "#include" not in text:
        return text.strip() + "\n"
    # Be forgiving of a short preamble, but never silently turn arbitrary prose
    # into a source file. The quality gate still requires a main and a compiler.
    starts = [m.start() for m in re.finditer(r"(?m)^(?:\s*#include\b|\s*int\s+main\s*\()", text)]
    if starts:
        return text[min(starts):].strip() + "\n"
    return text.strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("completion", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--prefix-file", type=Path, default=None)
    ap.add_argument("--cxx", default=os.environ.get("CXX", "FCC"))
    args = ap.parse_args()

    if not args.completion.is_file():
        print(f"CPP_QUALITY_FAIL missing completion: {args.completion}", file=sys.stderr)
        return 2
    out = args.out_dir or args.completion.parent
    out.mkdir(parents=True, exist_ok=True)
    source = extract_source(args.completion.read_text(encoding="utf-8", errors="replace"))
    if args.prefix_file and "#include" not in source:
        prefix = args.prefix_file.read_text(encoding="utf-8")
        # The generation prompt intentionally ends with the required include
        # scaffold. A causal completion can begin at using namespace std;,
        # so retain only the source suffix rather than prepending task prose.
        inc = prefix.find("#include")
        if inc >= 0:
            prefix = prefix[inc:]
        source = prefix.rstrip() + "\n" + source
    source_path = out / "generated.cpp"
    binary_path = out / "generated_cpp_test"
    source_path.write_text(source, encoding="utf-8")

    if "main" not in source or "CPP_TEST_PASS" not in source:
        print("CPP_QUALITY_FAIL missing required main/test marker")
        return 3

    if isinstance(args.cxx, str):
        cxx = shlex.split(args.cxx)
    else:
        cxx = list(args.cxx)
    compile_cmd = cxx + ["-Nclang", "-O2", "-std=c++17", "-o", str(binary_path), str(source_path)]
    compile_log = out / "compile.log"
    run_log = out / "run.log"
    cp = subprocess.run(compile_cmd, universal_newlines=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
    compile_log.write_text(cp.stdout, encoding="utf-8")
    if cp.returncode != 0:
        print(f"CPP_QUALITY_FAIL compile_rc={cp.returncode} log={compile_log}")
        return 4

    try:
        rp = subprocess.run([str(binary_path)], universal_newlines=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=30)
    except subprocess.TimeoutExpired as exc:
        run_log.write_text((exc.stdout or "") if isinstance(exc.stdout, str) else "", encoding="utf-8")
        print(f"CPP_QUALITY_FAIL timeout log={run_log}")
        return 5
    run_log.write_text(rp.stdout, encoding="utf-8")
    if rp.returncode != 0 or "CPP_TEST_PASS" not in rp.stdout:
        print(f"CPP_QUALITY_FAIL run_rc={rp.returncode} log={run_log}")
        return 6
    print(f"CPP_QUALITY_PASS compile=ok run=ok source={source_path} log={run_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
