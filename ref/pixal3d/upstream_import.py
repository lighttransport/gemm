"""Import pinned upstream mathematical modules without optional UI/render packages."""
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def prepare():
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["SPARSE_CONV_BACKEND"] = "none"
    for name in ("pixal3d", "pixal3d.models", "pixal3d.pipelines"):
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / "upstream" / name.replace(".", "/"))]
        sys.modules.setdefault(name, package)
