#!/usr/bin/env python3
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import models


class LagunaAdapterTest(unittest.TestCase):
    def setUp(self):
        self.a = models.get("laguna")

    def test_fp16_kv_selects_runner_and_launcher_flag(self):
        cfg = {"variant": "fp8", "kv_fp16": True, "port": 8080}
        argv, _env, _cwd = self.a.serve(cfg)
        self.assertIn("--kv-fp16", argv)
        self.assertTrue(self.a.runner_bin(cfg).endswith(
            "laguna_s21_fp8_kvfp16_ep_runner"))

    def test_fp16_kv_builds_dedicated_target(self):
        argv, _env, _cwd = self.a.build({
            "variant": "fp8", "kv_fp16": True,
        })
        self.assertIn("fp8-kvfp16", argv)

    def test_fp16_kv_rejects_non_fp8_weights(self):
        with self.assertRaises(models.ConfigError):
            self.a.generate({"variant": "bf16", "kv_fp16": True})

    def test_cpp_quality_is_explicit_one_shot_flag(self):
        argv, _env, _cwd = self.a.generate({
            "variant": "fp8", "chat": "write C++", "quality_cpp": True,
        })
        self.assertIn("--quality-cpp", argv)
        self.assertEqual(argv[argv.index("--max-new") + 1], "4096")
        serve, _env, _cwd = self.a.serve({
            "variant": "fp8", "port": 8080, "quality_cpp": True,
        })
        self.assertNotIn("--quality-cpp", serve)
        with self.assertRaises(models.ConfigError):
            self.a.generate({"variant": "fp8", "quality_cpp": True})


if __name__ == "__main__":
    unittest.main()
