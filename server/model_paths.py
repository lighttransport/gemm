"""Default weight storage on the NVMe volume containing this checkout."""
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def default_models_root(repo=ROOT):
    override = os.environ.get("VHUMAN_MODELS_ROOT")
    if override:
        return Path(override).expanduser().absolute()
    for parent in Path(repo).resolve().parents:
        if parent.parent == Path("/mnt") and re.fullmatch(r"nvme[0-9]+", parent.name):
            return parent / "models"
    return Path("/mnt/disk1/models")


MODELS_ROOT = default_models_root()
