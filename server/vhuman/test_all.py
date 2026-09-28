"""Run the complete virtual-human suite: python3 -m server.vhuman.test_all.

Includes browser tests (which report skips when Chrome or its dependencies
are unavailable). Optional triangulation tests need requirements-remesh.txt.
"""
from pathlib import Path
import unittest


def load_tests(loader, tests, pattern):
    """Load sibling test modules without requiring namespace-package discovery."""
    modules = [
        f"{__package__}.{path.stem}"
        for path in sorted(Path(__file__).parent.glob("test_*.py"))
        if path.stem != "test_all"
    ]
    return loader.loadTestsFromNames(modules)


if __name__ == "__main__":
    unittest.main()
