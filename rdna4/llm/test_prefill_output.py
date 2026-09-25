#!/usr/bin/env python3
"""Validate user-visible text and generated C from prefill benchmark logs."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


DRIVER = r'''
#include <limits.h>
#include <stdio.h>
int main(void) {
    const int values[] = {INT_MIN, -100, -1, 0, 1, 100, INT_MAX};
    int count = 0;
    for (int l = 0; l < 7; ++l)
        for (int h = l; h < 7; ++h)
            for (int i = 0; i < 7; ++i) {
                int lo = values[l], hi = values[h], x = values[i];
                int expected = x < lo ? lo : (x > hi ? hi : x);
                if (clamp(x, lo, hi) != expected) return 1;
                ++count;
            }
    printf("PASS: %d cases\n", count);
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--require-same-first-token", action="store_true")
    parser.add_argument(
        "--contains", action="append", default=[], metavar="TEXT",
        help="require TEXT in every generated response (may be repeated)")
    parser.add_argument(
        "--allow-no-c", action="store_true",
        help="allow responses without a fenced C program")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    tmp_root = Path(os.environ.get("TMPDIR", root / "tmp"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    first_tokens = set()
    checked = 0
    generated = 0

    with tempfile.TemporaryDirectory(prefix="prefill-output-", dir=tmp_root) as tmp:
        tmp_path = Path(tmp)
        for log in args.logs:
            text = log.read_text(encoding="utf-8", errors="strict")
            if "Result: PASS" not in text:
                raise RuntimeError(f"{log}: benchmark did not pass")
            matches = re.findall(
                r"=== Generated text ===\n(.*?)\n=== end ===", text, re.S)
            if not matches:
                raise RuntimeError(f"{log}: no generated text")
            tokens = re.findall(r"First decoded token id=(\d+)", text)
            if len(tokens) != len(matches):
                raise RuntimeError(f"{log}: missing first-token result")
            first_tokens.update(map(int, tokens))
            for index, body in enumerate(matches):
                generated += 1
                bad = [marker for marker in (
                    "Ġ", "Ċ", "�", "<|im_end|>", "<|im_start|>",
                    "<|endoftext|>") if marker in body]
                if bad:
                    raise RuntimeError(f"{log}: undecoded markers {bad}")
                for expected in args.contains:
                    if expected not in body:
                        raise RuntimeError(
                            f"{log}: generated response lacks {expected!r}")
                match = re.search(r"```(?:c|cpp)?\n(.*?)```", body, re.S)
                if not match:
                    if args.allow_no_c:
                        continue
                    raise RuntimeError(f"{log}: no complete C fence")
                source = tmp_path / f"case-{checked}.c"
                binary = tmp_path / f"case-{checked}"
                source.write_text(match.group(1) + DRIVER)
                subprocess.run(
                    ["gcc", "-std=c11", "-Wall", "-Wextra", "-pedantic",
                     "-fsanitize=undefined", str(source), "-o", str(binary)],
                    check=True)
                result = subprocess.check_output([str(binary)], text=True).strip()
                if result != "PASS: 196 cases":
                    raise RuntimeError(f"{log}: {result}")
                checked += 1

    if args.require_same_first_token and len(first_tokens) != 1:
        raise RuntimeError(f"first generated token differs: {sorted(first_tokens)}")
    print(f"prefill output: PASS ({len(args.logs)} logs, {generated} responses, "
          f"{checked} C programs, "
          f"first tokens {sorted(first_tokens)})")


if __name__ == "__main__":
    main()
