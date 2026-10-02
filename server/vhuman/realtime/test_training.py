import json
from pathlib import Path
import unittest
import numpy as np
from .src.avatar.provenance import sha256

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "tmp/vhuman-realtime/training-tests"


def receipt(path, role):
    return dict(path=path.name, source="original synthetic unit-test fixture", revision="v1", license="CC0-1.0",
                sha256=sha256(path), roles=[role])


class MotionTrainingTests(unittest.TestCase):
    def test_train_held_out_take_and_load_streaming(self):
        import torch
        from .src.animation.train import train
        from .src.animation.causal import MotionAdapter
        from .src.pipeline.protocol import TTSFeatureFrame
        WORK.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(9)
        takes, receipts = [], []
        for split in ("train", "validation"):
            path = WORK / (split + ".npz")
            hidden = rng.normal(size=(4, 12)).astype(np.float32)
            codes = rng.integers(2048, size=(4, 16), dtype=np.int32)
            controls = np.repeat((1/(1 + np.exp(-hidden[:, :1])))[:, None], 8, axis=1)
            np.savez(path, hidden=hidden, codes=codes, controls=controls)
            takes.append(dict(path=path.name, split=split)); receipts.append(receipt(path, "motion-training"))
        manifest = WORK / "motion.json"
        manifest.write_text(json.dumps(dict(format="vhuman.motion_corpus.v1", purpose="diagnostic", takes=takes,
            provenance=receipts, names=["jawOpen"], ranges=[[0, 1]], tts_revision="synthetic-test")))
        checkpoint = WORK / "motion.pt"
        result = train(manifest, checkpoint, epochs=2)
        with np.load(WORK / "train.npz") as data:
            expected_mean = data["hidden"].mean(0)
        state = torch.load(checkpoint, weights_only=True)["state_dict"]
        np.testing.assert_allclose(state["hidden_mean"].numpy(), expected_mean)
        self.assertTrue(np.isfinite(result["validation"]))
        with self.assertRaises(ValueError): MotionAdapter(checkpoint, "synthetic-test")
        model = MotionAdapter(checkpoint, "synthetic-test", allow_diagnostic=True)
        values = model.push(TTSFeatureFrame(0, 0, codes[0], hidden[0], "synthetic-test"))
        self.assertEqual([v.sample_position for v in values], list(range(0, 1920, 240)))
        with self.assertRaises(ValueError): model.push(TTSFeatureFrame(0, 3840, codes[0], hidden[0], "synthetic-test"))
        model.reset(1)
        self.assertEqual(model.push(TTSFeatureFrame(0, 1920, codes[0], hidden[0], "synthetic-test")), [])
        from .src.benchmark.motion import evaluate
        report = evaluate(manifest, checkpoint, WORK / "motion-evaluation.json", reference_parity=True)
        self.assertEqual(len(report["takes"]), 1)
        self.assertLess(report["takes"][0]["step_full_max_error"], 1e-5)


class AppearanceTrainingTests(unittest.TestCase):
    def test_cuda_appearance_fit_changes_radiance(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest("CUDA required")
        from .src.avatar.train import fit
        from .src.avatar.bundle import GaussianAvatar
        WORK.mkdir(parents=True, exist_ok=True)
        path = WORK / "appearance.npz"
        vertices = np.array([[[-.1, -.1, 1], [.1, -.1, 1], [0, .1, 1]]], np.float32)
        np.savez(path, vertices=vertices, triangles=np.array([[0, 1, 2]], np.int32),
                 images=np.zeros((1, 64, 64, 3), np.float32), controls=np.zeros((1, 1), np.float32),
                 view=np.eye(4, dtype=np.float32)[None],
                 intrinsics=np.array([[[200, 0, 32], [0, 200, 32], [0, 0, 1]]], np.float32))
        manifest = WORK / "appearance.json"
        manifest.write_text(json.dumps(dict(format="vhuman.appearance_corpus.v1", purpose="diagnostic", data=path.name,
            provenance=[receipt(path, "appearance-training")], control_names=["jawOpen"])))
        result = fit(manifest, WORK / "fitted.npz", count=128, steps=2)
        avatar = GaussianAvatar.load(WORK / "fitted.npz")
        self.assertTrue(np.isfinite(result["loss_l1"]))
        self.assertTrue(avatar.metadata["trained"])
        self.assertLess(float(avatar.arrays["rgb"].mean()), .5)
        self.assertEqual(avatar.metadata["purpose"], "diagnostic")
        self.assertEqual(avatar.metadata["covariance_policy"], "trace-v1")
        from .src.benchmark.appearance import evaluate
        report = evaluate(manifest, WORK / "fitted.npz", WORK / "evaluation")
        self.assertEqual(report["evaluation"], "training-reconstruction")
        self.assertTrue(np.isfinite(report["frames"][0]["linear_l1"]))


if __name__ == "__main__": unittest.main()
