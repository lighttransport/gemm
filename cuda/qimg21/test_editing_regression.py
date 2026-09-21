import json
from pathlib import Path
import tempfile
import unittest

from editing_regression import guidance_scale


class EditingRegressionTest(unittest.TestCase):
    def test_cfg_metadata(self):
        root = Path(__file__).resolve().parents[2] / "tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            ref = Path(folder)
            manifest = ref / "run.json"
            manifest.write_text(json.dumps({"use_true_cfg": False, "true_cfg_scale": 1}))
            self.assertEqual(guidance_scale(ref), 1)
            manifest.write_text(json.dumps({"use_true_cfg": True, "true_cfg_scale": 4}))
            with self.assertRaises(ValueError):
                guidance_scale(ref)
            (ref / "negative_prompt_embeds.npy").touch()
            self.assertEqual(guidance_scale(ref), 4)
            for scale in (float("nan"), float("inf"), 1, "4"):
                manifest.write_text(json.dumps({"use_true_cfg": True, "true_cfg_scale": scale}))
                with self.assertRaises(ValueError):
                    guidance_scale(ref)


if __name__ == "__main__":
    unittest.main()
