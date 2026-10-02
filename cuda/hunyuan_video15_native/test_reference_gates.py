"""Acceptance gates must reject incomplete runs and stale reference tensors."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ref.hunyuan_video15_native import verify

SCRATCH = ROOT / "tmp/hv15-native/tests"
SCRATCH.mkdir(parents=True, exist_ok=True)


class ReferenceGates(unittest.TestCase):
    def recipe(self):
        return dict(backend="hv15n_cuda_experimental", task="i2v", preset="fast12",
                    frames=81, fps=24, width=480, height=848, steps=12, cfg=1,
                    flow_shift=7, upstream_reference_revision=verify.UPSTREAM,
                    metrics={"memory_fit": "pass"})

    def test_complete_recipe_and_every_step_required(self):
        recipe = self.recipe()
        names = verify.pipeline_names(recipe)
        self.assertTrue(all(f"latent_step_{i}" in names for i in range(12)))
        self.assertIn("vae_encoded", names)
        self.assertIn("siglip_hidden", names)
        for key, value in (("frames", 80), ("cfg", 6), ("steps", 50), ("flow_shift", 5)):
            with self.assertRaisesRegex(ValueError, "recipe"):
                verify.pipeline_names(dict(recipe, **{key: value}))
        with self.assertRaisesRegex(ValueError, "memory-budget"):
            verify.pipeline_names(dict(recipe, metrics={"memory_fit": "unverified"}))
        recipe.update(task="t2v", preset="quality", steps=50, cfg=6, flow_shift=5)
        names = verify.pipeline_names(recipe)
        self.assertEqual(sum(n.startswith("latent_step_") for n in names), 50)
        self.assertIn("qwen_negative_hidden", names)
        self.assertNotIn("vae_encoded", names)

    def test_stale_reference_provenance_and_tensor_rejected(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            out = Path(folder)
            np.save(out / "qwen_hidden.npy", np.arange(6, dtype=np.float32))
            provenance = {"generation_manifest_sha256": "run-A", "weights": {"qwen": "weight-A"}}
            verify.record_references(out, ["qwen_hidden"], provenance)
            verify.validate_references(out, ["qwen_hidden"], provenance)
            with self.assertRaisesRegex(ValueError, "stale"):
                verify.validate_references(out, ["qwen_hidden"], dict(provenance, generation_manifest_sha256="run-B"))
            np.save(out / "qwen_hidden.npy", np.ones(6, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "stale"):
                verify.validate_references(out, ["qwen_hidden"], provenance)

    def test_partial_and_nonfinite_native_capture_rejected(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            out = Path(folder)
            (out / "x.json").write_text(json.dumps({"shape": [1, 2], "dtype": "float32"}))
            np.array([1], dtype="<f4").tofile(out / "x.f32")
            with self.assertRaisesRegex(ValueError, "byte count"):
                verify.convert(out)
            np.array([1, np.nan], dtype="<f4").tofile(out / "x.f32")
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                verify.convert(out)
            np.save(out / "vae_decoded.npy", np.ones((1, 3, 81, 16, 16), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "bounded/invalid shape"):
                verify.validate_pipeline_shapes(out, ["vae_decoded"])


if __name__ == "__main__":
    unittest.main()
