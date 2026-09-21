"""Check cache-discard semantics against actual Diffusers modules on CPU."""
import unittest
from types import SimpleNamespace

import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import (
    QwenImage21ResidualBlock, QwenImage21Resample, QwenImage21Decoder3d,
)

from vae_reference import DiscardFirstFrameCache, discard_single_frame_cache


class VaeCacheTest(unittest.TestCase):
    def test_complete_reduced_decoder_bit_exact(self):
        torch.manual_seed(17)
        decoder = QwenImage21Decoder3d(
            dim=8, z_dim=4, dim_mult=[1, 2, 4, 8, 8], num_res_blocks=2,
            temperal_upsample=[True, True, True, False],
            out_channels=4, is_residual=True).eval()
        vae = SimpleNamespace(decoder=decoder)
        source = torch.randn(1, 4, 1, 2, 2)
        with torch.inference_mode():
            expected = decoder(source, feat_cache=[None] * 256, feat_idx=[0], first_chunk=True)
            with discard_single_frame_cache(vae) as calls:
                actual = decoder(source, feat_cache=[None] * 256, feat_idx=[0], first_chunk=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertGreater(len(calls[0].written), 30)
        self.assertFalse(decoder._forward_pre_hooks)
        with self.assertRaisesRegex(ValueError, "one untiled"):
            with discard_single_frame_cache(vae):
                decoder(source, feat_cache=[None] * 256, feat_idx=[0], first_chunk=False)
        self.assertFalse(decoder._forward_pre_hooks)

    def test_official_modules_bit_exact(self):
        torch.manual_seed(42)
        modules = [QwenImage21ResidualBlock(8, 8).eval(),
                   QwenImage21Resample(8, "upsample3d").eval()]
        source = torch.randn(1, 8, 1, 4, 4)
        for module in modules:
            normal = [None] * 8
            discard = DiscardFirstFrameCache(8)
            with torch.inference_mode():
                expected = module(source, feat_cache=normal, feat_idx=[0])
                actual = module(source, feat_cache=discard, feat_idx=[0])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(discard.written)
            self.assertEqual(len(discard.written), sum(v is not None for v in normal))

    def test_reuse_fails(self):
        cache = DiscardFirstFrameCache(1)
        self.assertIsNone(cache[0])
        cache[0] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "reread"):
            _ = cache[0]
        with self.assertRaisesRegex(RuntimeError, "rewrote"):
            cache[0] = None
        with self.assertRaises(IndexError):
            _ = cache[1]


if __name__ == "__main__":
    unittest.main()
