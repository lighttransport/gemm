#!/usr/bin/env python3
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import models


class Qwen36AdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = models.get("qwen36")

    def test_is_wire_proxy_and_configurable(self):
        self.assertTrue(self.adapter.supports_serve)
        self.assertTrue(self.adapter.proxy_protocol)
        self.assertEqual(models.get_by_openai_model("qwen3.6"), self.adapter)
        cfg = {
            "server": "/opt/llama/bin/llama-server",
            "model": "/weights/qwen.gguf",
            "port": 8090,
            "devices": "Vulkan1,Vulkan0",
            "tensor_split": "1.4,1",
            "ctx": 524288,
        }
        argv, env, cwd = self.adapter.serve(cfg)
        self.assertEqual(argv[0], "/opt/llama/bin/llama-server")
        self.assertIn("Vulkan1,Vulkan0", argv)
        self.assertIn("524288", argv)
        self.assertIn("--fit", argv)
        self.assertIn("--fit-target", argv)
        self.assertEqual(env, {})
        self.assertEqual(cwd, "/weights")

    def test_model_and_server_can_come_from_environment(self):
        with mock.patch.dict(os.environ, {
                "QWEN36_MODEL": "/models/qwen.gguf",
                "LLMGR_LLAMA_SERVER": "/bin/llama-server",
        }, clear=False):
            argv, _env, cwd = self.adapter.serve({"port": 8091})
        self.assertEqual(argv[0], "/bin/llama-server")
        self.assertIn("/models/qwen.gguf", argv)
        self.assertEqual(cwd, "/models")

    def test_missing_model_is_a_config_error(self):
        with mock.patch.dict(os.environ, {"QWEN36_MODEL": ""}, clear=False):
            with self.assertRaises(models.ConfigError):
                self.adapter.serve({"port": 8092, "server": "/bin/llama-server"})

    def test_invalid_path_types_are_config_errors(self):
        with self.assertRaises(models.ConfigError):
            self.adapter.serve({"port": 8093, "server": ["llama-server"],
                                "model": "/models/qwen.gguf"})


if __name__ == "__main__":
    unittest.main()
