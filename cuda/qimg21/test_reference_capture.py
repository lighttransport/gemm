#!/usr/bin/env python3
"""Exercise CFG fixture indexing without loading model weights or using GPU."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import reference


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_in = torch.nn.Identity()
        self.proj_out = torch.nn.Identity()

    def forward(self, hidden_states, encoder_hidden_states, timestep):
        self.img_in(hidden_states)
        return self.proj_out(torch.full_like(hidden_states, encoder_hidden_states[0, 0, 0]))


class Pipeline:
    def __init__(self):
        self.transformer = Transformer()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def enable_sequential_cpu_offload(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        assert kwargs["negative_prompt"] == ""
        assert kwargs["use_kv_cache"] is False
        sample = torch.ones(1, 4, 64, dtype=torch.bfloat16)
        for step in range(2):
            positive = torch.full((1, 3, 4096), 2 + step, dtype=torch.bfloat16)
            negative = torch.full((1, 1, 4096), 1 + step, dtype=torch.bfloat16)
            timestep = torch.tensor([0.8 - step * 0.7], dtype=torch.bfloat16)
            for embeds in (positive, negative):
                self.transformer(hidden_states=sample, encoder_hidden_states=embeds, timestep=timestep)
            kwargs["callback_on_step_end"](
                self, step, timestep, {"latents": sample, "prompt_embeds": positive})
        return SimpleNamespace(images=[Image.new("RGBA", (32, 32))])


class ReferenceCaptureTest(unittest.TestCase):
    def test_cfg_steps_and_empty_negative_prompt(self):
        root = Path(__file__).resolve().parents[2]
        (root / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="qimg21-capture-", dir=root / "tmp") as work:
            folder = Path(work)
            args = ["reference.py", "--model", work, "--negative-prompt", "",
                    "--true-cfg-scale", "3", "--height", "32", "--width", "32",
                    "--steps", "2", "--dump-dir", work, "--dump-pred-dir", work]
            generator = torch.Generator
            with patch.object(sys, "argv", args), \
                 patch("diffusers.QwenImage21Pipeline", Pipeline), \
                 patch("torch.cuda.is_available", return_value=True), \
                 patch("torch.cuda.synchronize"), \
                 patch("torch.Generator", side_effect=lambda **kw: generator(device="cpu")):
                self.assertEqual(reference.main(), 0)
            self.assertEqual(sorted(p.name for p in folder.glob("pred_*.npy")),
                             ["pred_000.npy", "pred_001.npy"])
            for step in range(2):
                np.testing.assert_array_equal(np.load(folder / f"pred_{step:03d}.npy"),
                                              np.full((1, 4, 64), 4 + step, dtype=np.float32))
                self.assertTrue((folder / f"input_{step:03d}.npy").exists())
                self.assertTrue((folder / f"timestep_{step:03d}.npy").exists())
                self.assertTrue((folder / f"positive_{step:03d}.npy").exists())
                self.assertTrue((folder / f"negative_{step:03d}.npy").exists())
            self.assertEqual(np.load(folder / "negative_prompt_embeds.npy").shape, (1, 1, 4096))
            self.assertEqual(np.load(folder / "prompt_embeds.npy").shape, (1, 3, 4096))


if __name__ == "__main__":
    unittest.main()
