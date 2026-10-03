import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from glm_reap.pipeline import calibrate
from glm_reap.reap import Saliency


class CalibrationTests(unittest.TestCase):
    def test_resume_does_not_double_count_finished_windows(self):
        class Fixture:
            def __init__(self, *args):
                pass
            def identity(self):
                return {"source": "fixture"}
        class Runtime:
            fail = True
            def __init__(self, checkpoint, config, selected=None, device=None):
                self.saliency = None
            def begin_saliency(self):
                self.saliency = {0: Saliency(2)}
            def hidden(self, ids, vision=None):
                if self.saliency is not None:
                    score = self.saliency[0]
                    if Runtime.fail and score.count[0] == 1:
                        raise InterruptedError("fixture interruption")
                    score.count += 1
                    score.weighted_norm += np.array([2., 1.])
                return torch.ones(2, 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.jsonl").write_text("{}\n")
            config = {"source": "fixture", "output": directory, "gpu_limit_gib": 10, "activation_budget_gib": 1, "reserve_disk_gib": 0, "sequence_length": 16, "calibration_windows": 2, "quant_windows": 1, "keep_experts": 1, "activation_reservoir": 4, "seed": 42}
            def batches(*args, limit=None, **kwargs):
                for _ in range(limit):
                    yield ([1, 2], [False, True])
            with patch("glm_reap.pipeline.Checkpoint", Fixture), patch("glm_reap.runtime.Runtime", Runtime), patch("glm_reap.pipeline.windows", batches):
                with self.assertRaises(InterruptedError):
                    calibrate(config, root, "cpu")
                state = json.loads((root / "calibration-progress.json").read_text())
                self.assertEqual(state["completed_windows"], 1)
                Runtime.fail = False
                result = calibrate(config, root, "cpu")
                self.assertEqual(result["scores"]["0"]["count"], [2, 2])
                self.assertEqual(result["scores"]["0"]["weighted_norm_sum"], [4., 2.])
                self.assertEqual(result["selected"]["0"], [0])
                self.assertTrue((root / "activation-state.npz").exists())
