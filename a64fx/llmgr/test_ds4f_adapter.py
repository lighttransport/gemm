#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import models


class Ds4fAdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = models.get("ds4f")

    def test_is_wire_proxy_and_serving_capable(self):
        self.assertTrue(self.adapter.supports_serve)
        self.assertTrue(self.adapter.proxy_protocol)
        self.assertEqual(models.get_by_openai_model("ds4f"), self.adapter)
        self.assertEqual(models.get_by_openai_model("deepseek-v4-flash"),
                         self.adapter)

    def test_serve_configuration_is_environment_driven(self):
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.dict(os.environ, {"DS4F_NP": "2"}, clear=False):
            cfg = {"port": 8088, "work_dir": root,
                   "model_dir": "/weights/full", "stage_dir": "/local/full",
                   "tokenizer": "/models/tokenizer.json", "np": 2,
                   "ctx": 32768, "q8_dense": 0}
            argv, env, cwd = self.adapter.serve(cfg)
            self.assertEqual(argv[-1], self.adapter.SINGLE_LAUNCHER)
            self.assertEqual(cwd, root)
            self.assertEqual(env["PORT"], "8088")
            self.assertEqual(env["DS4F_REAL"], "1")
            self.assertEqual(env["NP"], "2")
            self.assertEqual(env["DS4F_STAGE_DIR"], "/local/full")
            self.assertEqual(env["DS4F_MODEL_DIR"], "/weights/full")
            self.assertEqual(env["TOK"], "/models/tokenizer.json")
            self.assertEqual(env["CTX"], "32768")
            self.assertEqual(env["DS4F_Q8_DENSE"], "0")

    def test_readiness_counts_only_matching_rank_files(self):
        with tempfile.TemporaryDirectory() as root:
            for rank in range(2):
                with open(os.path.join(root, "ds4f_ep_rank%02d.txt" % rank), "w") as f:
                    f.write("SERVE ready\n")
            self.assertEqual(self.adapter.readiness({"work_dir": root, "np": 2,
                                                     "deployment": "ep"}, 0)[0],
                             True)


if __name__ == "__main__":
    unittest.main()
