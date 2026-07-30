#!/usr/bin/env python3
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import models


class K3AdapterTest(unittest.TestCase):
    def setUp(self):
        self.a = models.get("k3")

    def test_contract_is_partial_oneshot(self):
        d = models.describe()["k3"]
        self.assertFalse(d["supports_serve"])
        self.assertIn("build", d["modes"])
        self.assertIn("stage", d["modes"])
        self.assertIn("generate", d["modes"])

    def test_stage_is_stage_only_and_arg_driven(self):
        argv, env, cwd = self.a.stage({
            "np": 96, "layer": 2, "experts": "0-7",
            "model_dir": "/weights/k3", "stage_dir": "/local/k3-test",
            "result_dir": "/shared/k3-stage-result",
        })
        self.assertIn("--stage-only", argv)
        self.assertEqual(argv[argv.index("--nodes") + 1], "96")
        self.assertEqual(argv[argv.index("--experts") + 1], "0-7")
        self.assertEqual(env, {})
        self.assertEqual(cwd, models.K3_DIR)

    def test_generate_reuses_stage_by_default(self):
        argv, _env, _cwd = self.a.generate({
            "np": 12, "layer": 1, "tokens": 32,
            "stage_dir": "/local/k3-test", "result_dir": "/shared/k3-run",
        })
        self.assertIn("--reuse-stage", argv)
        self.assertEqual(argv[argv.index("--tokens") + 1], "32")

    def test_real_layer_range_and_np_are_checked(self):
        with self.assertRaises(models.ConfigError):
            self.a.generate({"np": 97})
        with self.assertRaises(models.ConfigError):
            self.a.generate({"np": 12, "layer": 0})


if __name__ == "__main__":
    unittest.main()
