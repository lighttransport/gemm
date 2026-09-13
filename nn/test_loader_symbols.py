# SPDX-License-Identifier: MIT
"""Host-only regression: loader data must never interpose vendor functions."""
import re
import subprocess
import sys


def check(path):
    result = subprocess.run(
        ["nm", "-D", "--defined-only", path], check=True, capture_output=True, text=True
    )
    symbols = [line.split()[-1].split("@")[0] for line in result.stdout.splitlines() if line.split()]
    if "gn_create" not in symbols:
        raise RuntimeError(f"{path}: missing exported GN API; invalid test artifact")
    forbidden = [s for s in symbols if re.match(r"(?:cu|hip|nvrtc|hiprtc)[A-Z]|(?:cuew|rocew)", s)]
    if forbidden:
        raise RuntimeError(f"{path}: vendor loader symbols are exported: {', '.join(forbidden)}")
    print(f"PASS {path}: GN API exported, vendor loader symbols hidden")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: test_loader_symbols.py ELF [ELF ...]")
    for filename in sys.argv[1:]:
        check(filename)
