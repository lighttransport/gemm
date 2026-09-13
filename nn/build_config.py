"""Regenerate a build-mode stamp only when the optional SDK mode changes."""
# SPDX-License-Identifier: MIT
from pathlib import Path
import sys

path = Path(sys.argv[1])
value = "\n".join(sys.argv[2:]) + "\n"
if not path.exists() or path.read_text() != value:
    path.write_text(value)
