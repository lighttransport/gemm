#!/usr/bin/env python3
"""Embed a HIPRTC source and its local quoted includes as one C string."""
import json
from pathlib import Path
import re
import sys


def expand(path):
    path = Path(path)
    return re.sub(r'^#include "([^"]+)"\s*$',
                  lambda match: expand(path.parent / match[1]),
                  path.read_text(), flags=re.M)


if __name__ == "__main__":
    print("static const char " + sys.argv[2] + "[] = " + json.dumps(expand(sys.argv[1])) + ";")
