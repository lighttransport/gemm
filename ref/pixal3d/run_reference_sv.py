"""Run pinned single-view inference with the Blackwell compatibility layer."""
from pathlib import Path
import runpy

# Importing the multiview wrapper installs the alpha-only RMBG and bounded
# NATTEN compatibility hooks without executing its CLI.
from run_reference_mv import UPSTREAM


def main():
    runpy.run_path(str(Path(UPSTREAM) / "inference.py"), run_name="__main__")


if __name__ == "__main__":
    main()
