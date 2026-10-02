"""RMBG2 native C inference, preserving the reference alpha-only contract."""
from pathlib import Path
import os
import sys
import numpy as np
from PIL import Image

MODEL = Path(os.environ.get("QIMG21_RMBG_MODEL", "/mnt/nvme01/models/RMBG-2.0"))


class MattingError(RuntimeError):
    pass


def remove_background(rgba: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Return uint8 HxW alpha; no Torch or alternate Python interpreter."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from server.vhuman.native_models import rmbg_alpha
    try:
        pieces = device.split(":", 1)
        backend = pieces[0]
        index = int(pieces[1]) if len(pieces) == 2 else 0
        alpha = rmbg_alpha(Image.fromarray(rgba), MODEL, backend=backend, device=index)
    except (OSError, ValueError, RuntimeError) as exc:
        raise MattingError(str(exc)) from exc
    return np.asarray(alpha).copy()
