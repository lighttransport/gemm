"""Exercise the real compiled ELF and fail-closed paths; no GPU required."""
from pathlib import Path
import unittest
import patch_formats as patch


class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Path("build/e2m1_e2m1.cubin").read_bytes()
        cls.records = patch.instructions(Path("build/e2m1_e2m1.sass").read_text())

    def test_only_format_bits_change(self):
        sections = patch.text_sections(self.data)
        for mode in (1, 2, 3):
            changed = patch.patched_bytes(self.data, self.records, mode)
            expected = {sections[n][0] + pc + 9 for n, pc, *_ in self.records}
            actual = {i for i, (a, b) in enumerate(zip(self.data, changed)) if a != b}
            self.assertEqual(actual, expected)
            for i in actual:
                self.assertEqual(changed[i] ^ self.data[i], mode << 6)

    def test_corrupt_instruction_rejected(self):
        data = bytearray(self.data)
        name, pc, *_ = self.records[0]
        data[patch.text_sections(data)[name][0] + pc] ^= 1
        with self.assertRaises(ValueError):
            patch.patched_bytes(data, self.records, 3)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            patch.patched_bytes(self.data, self.records + self.records[:1], 3)

    def test_already_patched_rejected(self):
        records = list(self.records)
        n, pc, fmt, lo, hi = records[0]
        records[0] = (n, pc, fmt, lo, hi | 1 << 14)
        with self.assertRaises(ValueError):
            patch.patched_bytes(self.data, records, 3)

    def test_unknown_opcode_rejected(self):
        with self.assertRaises(ValueError):
            patch.instructions("Function : gemm_ref\nOMMA.UNKNOWN\n")
        with self.assertRaises(ValueError):
            patch.instructions("nothing here")


if __name__ == "__main__":
    unittest.main()
