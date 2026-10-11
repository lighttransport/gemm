"""Embed shared host/device equations once; do not maintain a second DDA copy."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
source = "\n".join((root / name).read_text() for name in
                   ("common/genia_math.h", "common/genia_gaussian.h", "cuda/genia/genia_kernels.cuh"))
Path(sys.argv[1]).write_text("static const char genia_kernel_source[] =\n" +
                           "\n".join(json.dumps(line + "\n") for line in source.splitlines()) + ";\n")
