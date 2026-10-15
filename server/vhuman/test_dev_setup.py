"""Installation safety and hardware build selection; no downloads or GPU use."""
import argparse
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import dev_setup


class DevSetupTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=dev_setup.ROOT / "tmp")
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.environ = patch.dict(os.environ)
        self.environ.start()
        self.addCleanup(self.environ.stop)
        self.root_patch = patch.object(dev_setup, "ROOT", root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        args = argparse.Namespace(ssh_identity=None, source_host=None, source_repo="repo",
                                  jobs=2, gpu_arch="sm_80", cuda_home=Path("/cuda"), data_root=root / "assets")
        self.setup = dev_setup.Setup(args)

    def test_installation_masks_gpu_and_keeps_caches_local(self):
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")
        for name in ("TMPDIR", "TEMP", "TMP", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "EM_CACHE"):
            self.assertTrue(Path(os.environ[name]).resolve().is_relative_to(dev_setup.ROOT.resolve()))

    def test_broken_link_is_preserved_before_repair(self):
        path = dev_setup.ROOT / "old"
        path.symlink_to("missing")
        self.setup.directory(path)
        self.assertTrue(path.is_dir())
        self.assertEqual((self.setup.state / "backups/old").readlink(), Path("missing"))
        self.setup.directory(path)

    def test_existing_work_is_never_replaced_by_alias(self):
        path, target = dev_setup.ROOT / "existing", dev_setup.ROOT / "new"
        path.mkdir()
        (path / "work").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "Refusing to replace"):
            self.setup.link(path, target)
        self.assertEqual((path / "work").read_text(), "preserve")

    def test_resolved_cache_link_can_be_repaired(self):
        path = self.setup.state / "tools/bin/old"
        path.symlink_to("missing")
        target = self.setup.state / "tools/new"
        target.mkdir()
        self.setup.link(path, target)
        self.assertEqual(path.resolve(), target.resolve())
        self.assertEqual(self.setup.backup_path(path).readlink(), Path("missing"))

    def test_alias_is_relative_and_idempotent(self):
        path, target = dev_setup.ROOT / "alias", dev_setup.ROOT / "target"
        target.mkdir()
        self.setup.link(path, target)
        self.setup.link(path, target)
        self.assertFalse(path.readlink().is_absolute())
        self.assertEqual(path.resolve(), target.resolve())

    def test_ampere_excludes_fp8_and_fp4_plugins(self):
        with patch.object(self.setup, "run") as run, patch.object(self.setup, "checkout"):
            self.setup.build()
        arguments = [str(arg) for call in run.call_args_list for arg in call.args]
        self.assertNotIn("libq21_fast_sage.so", arguments)
        self.assertNotIn("libq21_fast_fp4.so", arguments)
        self.assertIn("libq21_fast_int8.so", arguments)
        self.assertIn("GPU_ARCH=sm_80", arguments)

    def test_blackwell_build_enables_plugins(self):
        self.setup.args.gpu_arch = "sm_120"
        with patch.object(self.setup, "run") as run, patch.object(self.setup, "checkout"):
            self.setup.build()
        arguments = [str(arg) for call in run.call_args_list for arg in call.args]
        self.assertIn("libq21_fast_sage.so", arguments)
        self.assertIn("libq21_fast_fp4.so", arguments)
        self.assertIn("GPU_ARCH=sm_120", arguments)


if __name__ == "__main__":
    unittest.main()
