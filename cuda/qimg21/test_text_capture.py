"""CPU-only checks for native text encoder reference boundaries."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from text_capture import capture_text_encoder


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.model.language_model.norm = torch.nn.RMSNorm(4, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask):
        hidden = input_ids[..., None].expand(-1, -1, 4).to(torch.bfloat16)
        hidden = self.model.language_model.layers[0](hidden)
        return self.model.language_model.norm(hidden)


class TextCaptureTest(unittest.TestCase):
    def test_exact_input_and_prenorm_without_mutation(self):
        root = Path(__file__).resolve().parents[2]
        (root / "tmp").mkdir(exist_ok=True)
        encoder = Encoder()
        pipe = SimpleNamespace(text_encoder=encoder, _drop_idx=2)
        ids = torch.tensor([[0, 151643, 23, 24]], dtype=torch.int64)
        mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.int64)
        expected = encoder(input_ids=ids, attention_mask=mask)
        with tempfile.TemporaryDirectory(dir=root / "tmp", prefix="qimg21-text-") as work:
            folder = Path(work)
            with capture_text_encoder(pipe, folder):
                actual = encoder(input_ids=ids, attention_mask=mask)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            saved_ids = np.load(folder / "input_ids.npy")
            self.assertEqual(saved_ids.dtype, np.int64)
            np.testing.assert_array_equal(saved_ids, ids.numpy())
            np.testing.assert_array_equal(np.load(folder / "hidden_prenorm.npy"),
                                          ids[..., None].expand(-1, -1, 4).bfloat16().float().numpy())
            metadata = json.loads((folder / "capture.json").read_text())
            self.assertEqual(metadata["drop_idx"], 2)
            self.assertEqual(metadata["norm_calls"], 1)
            self.assertFalse(encoder._forward_pre_hooks)
            self.assertFalse(encoder.model.language_model.norm._forward_pre_hooks)
            np.testing.assert_array_equal(np.load(folder / "layer_00.npy"),
                                          np.load(folder / "hidden_prenorm.npy"))
            self.assertFalse(encoder.model.language_model.layers[0]._forward_hooks)

            with self.assertRaisesRegex(RuntimeError, "expected fixture boundaries"):
                with capture_text_encoder(pipe, folder):
                    pass  # Stale files from the preceding run must not satisfy validation.
            self.assertFalse(encoder._forward_pre_hooks)
            with self.assertRaisesRegex(ValueError, "injected"):
                with capture_text_encoder(pipe, folder):
                    raise ValueError("injected")
            self.assertFalse(encoder.model.language_model.norm._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
