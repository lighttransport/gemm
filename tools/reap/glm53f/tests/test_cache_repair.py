from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CacheRepairTests(unittest.TestCase):
    def test_materialize_archive_without_changing_shared_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root/"shared"
            info = shared/"repair_fixture-1.0.dist-info"
            info.mkdir(parents=True)
            (info/"METADATA").write_text("Metadata-Version: 2.1\nName: repair-fixture\nVersion: 1.0\n")
            (info/"WHEEL").write_text("Wheel-Version: 1.0\nTag: py3-none-any\n")
            (shared/"payload.py").write_text("VALUE = 1\n")
            cache = root/"cache"
            archive = cache/"archive-v0"
            archive.mkdir(parents=True)
            link = archive/"fixture-id"
            link.symlink_to(shared, target_is_directory=True)
            torch = root/"torch.metadata"
            torch.write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.11.0\nRequires-Dist: repair-fixture==1.0\n")
            script = Path(__file__).resolve().parents[1]/"scripts/repair-uv-cache.py"
            subprocess.run([sys.executable, str(script), "--cache", str(cache), "--torch-metadata", str(torch)], check=True, capture_output=True)
            self.assertFalse(link.is_symlink())
            self.assertTrue(link.is_dir())
            (link/"payload.py").write_text("VALUE = 2\n")
            self.assertEqual((shared/"payload.py").read_text(), "VALUE = 1\n")
