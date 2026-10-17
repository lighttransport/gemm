"""Shared storage resolution without downloads, GPU use or host mutations."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from server import python_env


class PythonEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=python_env.ROOT / 'tmp')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.shared = self.root / 'shared'
        self.shared.mkdir()
        environment = patch.dict(os.environ, GEMM_VENVS_ROOT=str(self.shared), UV_CACHE_DIR=str(self.root / 'cache'))
        environment.start()
        self.addCleanup(environment.stop)

    def test_existing_environment_is_preserved(self):
        legacy = self.root / 'branch/.venv'
        legacy.mkdir(parents=True)
        self.assertEqual(python_env.environment_path(legacy, link=True), legacy)
        self.assertFalse(legacy.is_symlink())

    def test_new_profile_has_stable_compatibility_alias(self):
        legacy = self.root / 'branch/tmp/cpu-venv'
        target = python_env.environment_path(legacy, link=True)
        self.assertTrue(target.is_relative_to(self.shared))
        self.assertEqual(legacy.resolve(), target)
        self.assertEqual(python_env.environment_path(legacy, link=True), target)
        other = python_env.environment_path(self.root / 'other/tmp/cpu-venv')
        self.assertNotEqual(target, other)

    def test_registry_restores_exact_migrated_profile(self):
        legacy = self.root / 'branch/.venv'
        target = self.shared / 'cuda-py310-locked'
        (self.shared / 'registry.json').write_text(json.dumps({str(legacy): {'destination': str(target)}}))
        self.assertEqual(python_env.environment_path(legacy, link=True), target)
        self.assertEqual(legacy.resolve(), target)

    def test_cache_override(self):
        self.assertEqual(python_env.cache_path(), self.root / 'cache')


if __name__ == '__main__':
    unittest.main()
