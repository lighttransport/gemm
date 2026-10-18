"""Check format-only host changes and reject unsafe fatbinary containers."""
import json
from pathlib import Path
import struct
import unittest

import patch_cutlass as container
import patch_formats as patch


class CutlassPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path("build/libcutlass_e0m3_base.so").read_bytes()
        cls.output = Path("build/libcutlass_e0m3.so").read_bytes()
        cls.manifest = json.loads(Path("build/libcutlass_e0m3.patch.json").read_text())

    def test_only_designated_format_bytes_change(self):
        self.assertEqual(len(self.source), len(self.output))
        expected = set()
        records = patch.instructions(Path("build/libcutlass_e0m3.sass").read_text(), container.PREFIXES)
        for offset, elf in container.cuda_elves(self.source):
            sections = patch.text_sections(elf)
            for name, pc, *_ in records:
                if name in sections:
                    expected.add(offset + sections[name][0] + pc + 9)
        changes = {i for i, (a, b) in enumerate(zip(self.source, self.output)) if a != b}
        self.assertEqual(changes, expected)
        self.assertEqual(len(changes), self.manifest["instruction_count"])
        self.assertTrue(changes)
        for i in changes:
            self.assertEqual(self.source[i] ^ self.output[i], 0xC0)

    def test_ptx_fallback_rejected(self):
        data = bytearray(self.source)
        base, _ = container.fatbin_section(data)
        struct.pack_into("<H", data, base + 16, 1)
        with self.assertRaisesRegex(ValueError, "non-ELF"):
            container.cuda_elves(data)

    def test_compressed_elf_rejected(self):
        data = bytearray(self.source)
        offset, _ = container.cuda_elves(data)[0]
        data[offset:offset + 4] = b"\x28\xb5\x2f\xfd"
        with self.assertRaises(ValueError):
            container.cuda_elves(data)

    def test_wrong_architecture_rejected(self):
        data = bytearray(self.source)
        offset, _ = container.cuda_elves(data)[0]
        data[offset + 49] = 100
        with self.assertRaisesRegex(ValueError, "SM120"):
            container.cuda_elves(data)

    def test_truncated_container_rejected(self):
        base, size = container.fatbin_section(self.source)
        with self.assertRaises(ValueError):
            container.cuda_elves(self.source[:base + size - 1])

    def test_unknown_kernel_prefix_rejected(self):
        sass = Path("build/libcutlass_e0m3.sass").read_text()
        with self.assertRaisesRegex(ValueError, "unexpected OMMA"):
            patch.instructions(sass)


if __name__ == "__main__":
    unittest.main()
