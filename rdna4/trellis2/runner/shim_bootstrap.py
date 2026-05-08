"""Single entry point for runner/debug/tools scripts.

`setup_paths()` injects sys.path entries for the trellis2 repo root, the rdna4
package, and the ref/trellis2 helpers. `install_all()` installs the shims
(nvdiffrast/cumesh/flash_attn) and the SparseLinear hipBLASLt workaround.

Both are idempotent — calling twice is safe.
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
TRELLIS2_REPO = os.path.join(_REPO_ROOT, 'cpu', 'trellis2', 'trellis2_repo')
RDNA4_DIR = os.path.join(_REPO_ROOT, 'rdna4', 'trellis2')
REF_DIR = os.path.join(_REPO_ROOT, 'ref', 'trellis2')

_PATHS_DONE = False
_SHIMS_DONE = False


def setup_paths() -> None:
    global _PATHS_DONE
    if _PATHS_DONE:
        return
    for p in (REF_DIR, TRELLIS2_REPO, RDNA4_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    _PATHS_DONE = True


def install_all() -> None:
    """Install shims + SparseLinear chunking. Must be called before importing
    trellis2 pipeline code."""
    global _SHIMS_DONE
    if _SHIMS_DONE:
        return
    setup_paths()

    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('ATTN_BACKEND', 'sdpa')

    from shims import texgen_sw_rast, cumesh_xatlas, flash_attn_sdpa
    # Back-compat: ref/trellis2/gen_stage2_ref.py imports `texgen_sw_rast`
    # at the top level. Alias the package modules so legacy `import X` still
    # resolves after the shims/ rename.
    sys.modules.setdefault('texgen_sw_rast', texgen_sw_rast)
    sys.modules.setdefault('cumesh_xatlas_shim', cumesh_xatlas)
    sys.modules.setdefault('flash_attn_sdpa_shim', flash_attn_sdpa)
    texgen_sw_rast.install_as_nvdiffrast()
    cumesh_xatlas.install_as_cumesh()
    try:
        from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
        PACKAGE_DISTRIBUTION_MAPPING.setdefault('flash_attn', ['flash-attn'])
    except ImportError:
        pass
    flash_attn_sdpa.install_as_flash_attn()

    if os.environ.get('USE_RDNA4_LINEAR_CHUNK', '1') == '1':
        # Without this the trailing SparseLinear(64->6) produces ±1e18 garbage
        # on gfx1201; see rdna4/hipblaslt-issue.md.
        if os.environ.get('USE_RDNA4_LINEAR_KERNEL', '1') == '1':
            from kernels import linear_patch
            linear_patch.install()
        else:
            from kernels import spconv_rdna4_ext
            spconv_rdna4_ext.install_sparse_linear_chunking()

    _SHIMS_DONE = True
