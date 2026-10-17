"""Storage selection and migrations without loading or downloading weights."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server.model_paths import ROOT, default_models_root
from .relocate_models import link_shared_models, relocate_models


class ModelStorageTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.repo, self.models = self.root / "repo", self.root / "models"
        self.source = self.repo / "tmp/vhuman-musclemimic-checkpoint"
        self.source.mkdir(parents=True)
        (self.source / "weights").write_bytes(b"checkpoint")

    def test_root_follows_checkout_volume_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_models_root('/mnt/nvme01/work/gemm/repo'), Path('/mnt/nvme01/models'))
            self.assertEqual(default_models_root('/mnt/nvme02/work/gemm/repo'), Path('/mnt/nvme02/models'))
            self.assertEqual(default_models_root('/srv/repo'), Path('/mnt/disk1/models'))
        with patch.dict(os.environ, VHUMAN_MODELS_ROOT=str(self.models)):
            self.assertEqual(default_models_root(), self.models)

    def test_move_preserves_inode_and_old_path_and_is_idempotent(self):
        inode = (self.source / 'weights').stat().st_ino
        preview = relocate_models(self.repo, self.models)
        self.assertEqual(len(preview), 1)
        self.assertFalse(self.models.exists())
        relocate_models(self.repo, self.models, apply=True)
        target = self.models / 'vhuman/mm-fullbody-base/weights'
        self.assertEqual(target.stat().st_ino, inode)
        self.assertEqual((self.source / 'weights').read_bytes(), b'checkpoint')
        self.assertTrue(self.source.is_symlink())
        self.assertEqual(relocate_models(self.repo, self.models, apply=True), [])

    def test_conflict_preflight_leaves_all_sources_intact(self):
        target = self.models / 'vhuman/mm-fullbody-base'
        target.mkdir(parents=True)
        (target / 'weights').write_bytes(b'different')
        with self.assertRaisesRegex(ValueError, 'Refusing to replace'):
            relocate_models(self.repo, self.models, apply=True)
        self.assertFalse(self.source.is_symlink())
        self.assertEqual((target / 'weights').read_bytes(), b'different')

    def test_external_alias_is_not_copied_or_moved(self):
        external = self.root / 'shared-models'
        external.mkdir()
        old = self.repo / 'tmp/vhuman-rig/models'
        old.parent.mkdir(parents=True)
        old.symlink_to(external)
        relocate_models(self.repo, self.models, apply=True)
        self.assertEqual((self.models / 'vhuman-rig-models').resolve(), external)
        self.assertTrue(old.is_symlink())

    def test_broken_source_alias_is_reported(self):
        old = self.repo / 'tmp/vhuman-rig/models'
        old.parent.mkdir(parents=True)
        old.symlink_to('missing')
        with self.assertRaisesRegex(ValueError, 'Broken source model link'):
            relocate_models(self.repo, self.models, apply=True)
        self.assertFalse(self.source.is_symlink())

    def test_shared_models_are_aliased_without_replacing_local_weights(self):
        shared = self.root / 'shared'
        (shared / 'qimg-21').mkdir(parents=True)
        (shared / 'Pixal3D').mkdir()
        local = self.models / 'Pixal3D'
        local.mkdir(parents=True)
        (local / 'keep').write_text('local')
        preview = link_shared_models(shared, self.models)
        self.assertEqual(len(preview), 1)
        self.assertFalse((self.models / 'qimg-21').exists())
        link_shared_models(shared, self.models, apply=True)
        self.assertEqual((self.models / 'qimg-21').resolve(), shared / 'qimg-21')
        self.assertFalse(local.is_symlink())
        self.assertEqual(link_shared_models(shared, self.models, apply=True), [])


if __name__ == '__main__':
    unittest.main()
