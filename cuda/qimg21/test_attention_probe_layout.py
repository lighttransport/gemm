import unittest
from types import SimpleNamespace

from attention_probe import native_editing_segments


def layout(text):
    return SimpleNamespace(read_text=lambda: text)


class NativeEditingSegmentsTest(unittest.TestCase):
    def test_interleaved_text_and_images(self):
        source = layout("4 3 2 0 1 0 1 2 2 2 2")
        self.assertEqual(native_editing_segments(source, 10),
                         (6, [(0, 1, True), (1, 5, False), (5, 6, True)]))

    def test_rejects_tensor_shape_mismatch(self):
        with self.assertRaises(ValueError):
            native_editing_segments(layout("4 3 2 0 1 0 1 2 2 2 2"), 9)

    def test_rejects_split_image_block(self):
        with self.assertRaises(ValueError):
            native_editing_segments(layout("5 4 2 0 1 0 1 1 2 4 2 2"), 13)


if __name__ == "__main__":
    unittest.main()
