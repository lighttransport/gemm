import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from glm_reap.bank import Bank, Candidate
from glm_reap.common import execution_options


class MemoryTests(unittest.TestCase):
    def test_half_physical_cores_and_overrides(self):
        with patch.dict(os.environ, {}, clear=True), patch("psutil.cpu_count", return_value=16):
            options = execution_options({"output": "unused"})
            self.assertEqual(options["cpu_threads"], 8)
            with patch.dict(os.environ, {"REAP_CPU_THREADS": "3", "REAP_MEMORY_MODE": "mmap"}):
                options = execution_options({"output": "unused"})
                self.assertEqual(options["cpu_threads"], 3)
                self.assertEqual(options["memory_mode"], "mmap")

    def test_mmap_bank_is_on_disk_and_matches_ram(self):
        with tempfile.TemporaryDirectory() as directory:
            value = np.arange(512, dtype=np.float32).reshape(2, 256)
            bank = Bank("mmap", directory, reserve=0)
            bank.add("weight", [Candidate("F32", value.shape, value.view(np.uint8).reshape(-1))])
            raw = bank.tensors["weight"][0].raw
            self.assertIsInstance(raw, np.memmap)
            np.testing.assert_array_equal(bank.read("weight"), value)
            np.testing.assert_array_equal(np.memmap(raw.filename, dtype=np.uint8, mode="r"), raw)
            self.assertEqual(len(list(Path(directory).glob("*.bin"))), 1)
