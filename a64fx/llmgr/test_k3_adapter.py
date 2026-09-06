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
        self.assertTrue(d["supports_cache"])

    def test_semantic_api_is_not_available_for_partial_runner(self):
        with self.assertRaises(models.ConfigError):
            self.a.responses_request({"input": "hello"})

    def test_stage_is_stage_only_and_arg_driven(self):
        argv, env, cwd = self.a.stage({
            "np": 96, "layer": 2, "experts": "0-7",
            "model_dir": "/weights/k3", "stage_dir": "/local/k3-test",
            "result_dir": "/shared/k3-stage-result",
        })
        self.assertIn("--stage-only", argv)
        self.assertEqual(argv[argv.index("--nodes") + 1], "96")
        self.assertEqual(argv[argv.index("--tp-nodes") + 1], "96")
        self.assertEqual(argv[argv.index("--experts") + 1], "0-7")
        self.assertEqual(env, {})
        self.assertEqual(cwd, models.K3_DIR)

    def test_stage_and_profile_use_same_context_parallel_partition(self):
        stage, _env, _cwd = self.a.stage({
            "np": 24, "tp_np": 3, "layer": 2,
            "stage_dir": "/local/k3-test",
            "model_dir": "/weights/k3",
        })
        profile = self.a.profile_argv({
            "np": 24, "tp_np": 3, "layer": 2, "tokens": 4,
            "stage_dir": "/local/k3-test",
        })
        self.assertEqual(stage[stage.index("--tp-nodes") + 1], "3")
        self.assertEqual(profile[profile.index("--tp-nodes") + 1], "3")

    def test_generate_reuses_stage_by_default(self):
        argv, _env, _cwd = self.a.generate({
            "np": 12, "layer": 1, "tokens": 32,
            "stage_dir": "/local/k3-test", "result_dir": "/shared/k3-run",
        })
        self.assertIn("--reuse-stage", argv)
        self.assertEqual(argv[argv.index("--tokens") + 1], "32")

    def test_generate_adds_cache_load_save(self):
        argv, _env, _cwd = self.a.generate({
            "np": 12, "layer": 1, "tokens": 16,
            "cache_load": "/tmp/k3-cache-load.bin",
            "cache_save": "/tmp/k3-cache-save.bin",
            "result_dir": "/shared/k3-run",
        })
        self.assertIn("--cache-load", argv)
        self.assertEqual(argv[argv.index("--cache-load") + 1],
                         "/tmp/k3-cache-load.bin")
        self.assertIn("--cache-save", argv)
        self.assertEqual(argv[argv.index("--cache-save") + 1],
                         "/tmp/k3-cache-save.bin")

    def test_cache_paths_must_be_strings(self):
        with self.assertRaises(models.ConfigError):
            self.a.generate({
                "np": 12, "layer": 1,
                "cache_load": 123,
                "result_dir": "/shared/k3-run",
            })
        with self.assertRaises(models.ConfigError):
            self.a.generate({
                "np": 12, "layer": 1,
                "cache_save": 456,
                "result_dir": "/shared/k3-run",
            })
        with self.assertRaises(models.ConfigError):
            self.a.generate({
                "np": 12, "layer": 1,
                "cache_load": "/shared/bad\x00cache",
                "result_dir": "/shared/k3-run",
            })

    def test_192_nodes_split_into_two_tp96_contexts(self):
        argv, _env, _cwd = self.a.generate({
            "np": 192, "layer": 1, "tokens": 32, "dummy": True,
            "result_dir": "/shared/k3-run",
        })
        self.assertEqual(argv[argv.index("--nodes") + 1], "192")
        self.assertEqual(argv[argv.index("--tp-nodes") + 1], "96")

    def test_context_parallel_flags_for_common_node_counts(self):
        for np_, tp in ((16, 1), (24, 3), (32, 4), (48, 4)):
            argv, _env, _cwd = self.a.generate({
                "np": np_, "tp_np": tp, "layer": 1, "tokens": 8,
                "result_dir": "/shared/k3-run",
            })
            self.assertEqual(argv[argv.index("--nodes") + 1], str(np_))
            self.assertEqual(argv[argv.index("--tp-nodes") + 1], str(tp))

    def test_context_parallel_flags_for_scale_node_counts(self):
        for np_, tp in ((72, 12), (96, 24)):
            argv, _env, _cwd = self.a.generate({
                "np": np_, "tp_np": tp, "layer": 1, "tokens": 8,
                "result_dir": "/shared/k3-run",
            })
            self.assertEqual(argv[argv.index("--nodes") + 1], str(np_))
            self.assertEqual(argv[argv.index("--tp-nodes") + 1], str(tp))

    def test_context_parallel_for_small_node_counts(self):
        for np_, tp in ((64, 16), (128, 32), (160, 20)):
            argv, _env, _cwd = self.a.generate({
                "np": np_, "tp_np": tp, "layer": 1, "tokens": 8,
                "result_dir": "/shared/k3-run",
            })
            self.assertEqual(argv[argv.index("--nodes") + 1], str(np_))
            self.assertEqual(argv[argv.index("--tp-nodes") + 1], str(tp))

    def test_context_parallel_rejects_incompatible_tp_nodes(self):
        with self.assertRaises(models.ConfigError):
            self.a.generate({"np": 16, "tp_np": 5, "layer": 1, "tokens": 8,
                             "result_dir": "/shared/k3-run"})
        with self.assertRaises(models.ConfigError):
            self.a.generate({"np": 7, "tp_np": 2, "layer": 1, "tokens": 8,
                             "result_dir": "/shared/k3-run"})
        with self.assertRaises(models.ConfigError):
            self.a.generate({"np": 12, "tp_np": 97, "layer": 1, "tokens": 8,
                             "result_dir": "/shared/k3-run"})

    def test_cache_load_save_absent_or_none_are_omitted(self):
        argv, _env, _cwd = self.a.generate({
            "np": 12, "layer": 1, "tokens": 8,
            "result_dir": "/shared/k3-run",
            "cache_load": None,
            "cache_save": None,
        })
        self.assertNotIn("--cache-load", argv)
        self.assertNotIn("--cache-save", argv)

    def test_cache_flags_can_be_combined_with_context_parallel(self):
        argv, _env, _cwd = self.a.generate({
            "np": 12, "tp_np": 3, "layer": 1, "tokens": 8,
            "cache_load": "/shared/prefix.kv",
            "cache_save": "/shared/continuation.kv",
            "result_dir": "/shared/k3-run",
        })
        self.assertEqual(argv[argv.index("--tp-nodes") + 1], "3")
        self.assertEqual(argv[argv.index("--cache-load") + 1],
                         "/shared/prefix.kv")
        self.assertEqual(argv[argv.index("--cache-save") + 1],
                         "/shared/continuation.kv")

    def test_openai_aliases_resolve_to_laguna_adapter(self):
        self.assertEqual(models.get_by_openai_model("laguna-s21").name, "laguna")
        self.assertEqual(models.get_by_openai_model("laguna").name, "laguna")
        with self.assertRaises(models.ConfigError):
            models.get_by_openai_model("k3")

    def test_real_layer_range_and_np_are_checked(self):
        with self.assertRaises(models.ConfigError):
            self.a.generate({"np": 513})
        with self.assertRaises(models.ConfigError):
            self.a.generate({"np": 192, "tp_np": 72})
        with self.assertRaises(models.ConfigError):
            self.a.generate({"np": 12, "layer": 0})


if __name__ == "__main__":
    unittest.main()
