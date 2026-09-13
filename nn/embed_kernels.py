# SPDX-License-Identifier: MIT
"""Embed project-owned runtime compiler input in a generated build artifact."""
import json
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
for name in ("gn_sm120.cuh", "gn_rdna4.cuh"):
    source = source.replace(f'#include "{name}"', Path(sys.argv[1]).with_name(name).read_text())
Path(sys.argv[2]).write_text("static const char gn_kernel_source[] =\n" +
                           "\n".join(json.dumps(line + "\n") for line in source.splitlines()) + ";\n")
