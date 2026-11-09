"""Run the complete virtual-human suite: python3 -m server.vhuman.test_all.

Includes browser tests (which report skips when Chrome or its dependencies
are unavailable). Optional triangulation tests need requirements-remesh.txt.
"""
from pathlib import Path
import unittest


def load_tests(loader, tests, pattern):
    """Load sibling test modules without requiring namespace-package discovery."""
    root = Path(__file__).parent
    modules = [
        f"{__package__}.{path.stem}"
        for path in sorted(root.glob("test_*.py"))
        if path.stem != "test_all"
    ] + [
        f"{__package__}.body.{path.stem}"
        for path in sorted((root / "body").glob("test_*.py"))
        # Terra tests need mujoco from the separate Terra interpreter (see dev_setup.py).
        if not path.stem.startswith("test_terra")
    ]
    return loader.loadTestsFromNames(modules)


if __name__ == "__main__":
    unittest.main()
