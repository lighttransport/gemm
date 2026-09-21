"""Compare native editing metadata/scatter/RoPE with official Diffusers, CPU."""
from pathlib import Path
import subprocess
import unittest

import numpy as np
import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Rope, QwenImage21Transformer2DModel,
)


class JointLayoutTest(unittest.TestCase):
    def test_layouts(self):
        binary = Path(__file__).with_name("test_joint_layout")
        cases = [
            ([0, 0, 0, 1, 1], 3, [(1, 2, 4)]),
            ([0, 1, 0, 0, 1, 1], 4, [(1, 2, 2), (1, 2, 4)]),
            # Adjacent condition images must not merge into a single block.
            ([0, 1, 1, 0, 1], 4, [(1, 2, 2), (1, 2, 2), (1, 2, 2)]),
            ([0, 1, 1, 1, 0, 1, 1], 5, [(1, 3, 4), (1, 4, 2)]),
        ]
        for mask, text_slots, shapes in cases:
            data = [len(mask), text_slots, len(shapes), *mask]
            data += [v for _, h, w in shapes for v in (h, w)]
            result = subprocess.run([str(binary)], input=" ".join(map(str, data)),
                                    text=True, capture_output=True, check=True)
            lines = result.stdout.splitlines()
            n, prefix, image_count = map(int, lines[0].split())
            rows = np.array([list(map(int, line.split())) for line in lines[1:n+1]])
            attention = np.array([list(map(int, line.split())) for line in lines[n+1:]], dtype=bool)
            slot_mask = torch.tensor(mask, dtype=torch.bool)
            repeats = torch.where(slot_mask, 4, 1)
            image_mask = torch.repeat_interleave(slot_mask, repeats)
            ids, target = QwenImage21Transformer2DModel.build_token_metadata(image_mask, shapes)
            np.testing.assert_array_equal(rows[:, 2], ids.numpy())
            self.assertEqual(prefix, int((~target).sum()))
            self.assertEqual(image_count, int(image_mask.sum()))
            index = torch.arange(n)
            expected_mask = ((index[:, None] >= index[None, :]) |
                             ((ids[:, None] >= 0) & (ids[:, None] == ids[None, :])))
            np.testing.assert_array_equal(attention, expected_mask.numpy())

            rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
            expected_rope = rope(shapes, image_mask, torch.device("cpu"))
            got_rope = torch.cat([rope.freqs[axis][rows[:, 3+axis]] for axis in range(3)], dim=-1)
            torch.testing.assert_close(got_rope, expected_rope, rtol=0, atol=0)

            # Check text/image source indices against actual repeat+scatter.
            text = torch.arange(text_slots, dtype=torch.int64)
            original = torch.cat([text, torch.zeros(len(mask)-text_slots, dtype=torch.int64)])
            original = original.repeat_interleave(repeats)
            original[image_mask] = 1000 + torch.arange(image_count)
            got = np.where(rows[:, 1] >= 0, 1000 + rows[:, 1], rows[:, 0])
            np.testing.assert_array_equal(got, original.numpy())

    def test_invalid_layout(self):
        binary = Path(__file__).with_name("test_joint_layout")
        for data in ("2 1 1 0 0 2 2",  # target slot is not marked
                     "2 1 1 0 1 3 3",  # block does not fit four-token slots
                     "4 3 2 1 0 1 1 2 4 2 2"):  # text splits an image block
            result = subprocess.run([str(binary)], input=data, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
