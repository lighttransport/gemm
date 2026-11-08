"""Pore spectrum and staged-asset cache regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .reconstruction import skin_templates as skin


class SkinPoreTests(unittest.TestCase):
    def test_isotropic_grain_has_no_dominant_grid_frequency(self):
        peaks=[]
        for mode in ('legacy','isotropic'):
            image=skin.micro_normal(grain=mode)
            normal=image.astype(float)/127.5-1
            self.assertLess(np.abs(np.linalg.norm(normal,axis=-1)-1).max(),.008)
            power=np.abs(np.fft.fft2(normal[...,:2],axes=(0,1)))**2
            power[0,0]=0
            peaks.append(power.sum(-1).max()/power.sum())
        self.assertLess(peaks[1],peaks[0]/10)

    def test_isotropic_cache_keeps_legacy_asset_and_tracks_seed(self):
        scratch=Path(__file__).resolve().parents[2]/'tmp'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            out=Path(directory);(out/'skin_micro_normal.png').write_bytes(b'legacy fixture')
            source=out/'skin_templates.json';source.write_text(json.dumps(dict(seed=7)))
            with patch.object(skin,'micro_normal',return_value=np.full((4,4,3),128,np.uint8)) as generate:
                target=skin.ensure_isotropic_pores(out)
                self.assertTrue(target.is_file())
                skin.ensure_isotropic_pores(out)
                self.assertEqual(generate.call_count,1)
                source.write_text(json.dumps(dict(seed=8)))
                skin.ensure_isotropic_pores(out)
                self.assertEqual(generate.call_count,2)
            self.assertEqual((out/'skin_micro_normal.png').read_bytes(),b'legacy fixture')
