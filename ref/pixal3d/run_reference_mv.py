"""Run pinned multiview inference without loading RMBG for RGBA-only inputs.

The upstream entry point checks every view's alpha before calling its matting
model, but constructs RMBG eagerly. This launcher replaces only that eager
constructor. An opaque/RGB view still fails instead of silently changing the
reference preprocessing contract.
"""
from pathlib import Path
import runpy
import sys

UPSTREAM = Path(__file__).with_name("upstream")
REF_ROOT = Path(__file__).parent
sys.path[:0] = [str(UPSTREAM), str(REF_ROOT / "cumesh-upstream"),
                str(REF_ROOT / "flexgemm-upstream")]

from pixal3d.pipelines import rembg


class AlphaOnlyRmbg:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image):
        raise RuntimeError(
            "reference view lacks useful alpha; install the authorized RMBG-2.0 weights")

    def to(self, device):
        return self

    def cpu(self):
        return self


rembg.BiRefNet = AlphaOnlyRmbg
runpy.run_path(str(UPSTREAM / "inference_mv.py"), run_name="__main__")
