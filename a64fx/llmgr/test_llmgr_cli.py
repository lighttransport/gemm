#!/usr/bin/env python3
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import llmgr_cli as cli


class LlmgCliTest(unittest.TestCase):
    def test_chat_forwards_cache_flags(self):
        calls = []

        def fake_call(args, method, path, body=None, stream=False, timeout=None):
            calls.append((method, path, body, stream, timeout))
            return {"id": "job-1"}

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main([
                "chat", "Hello",
                "--model", "laguna-s21",
                "--max-new", "3",
                "--cache-load", "/tmp/laguna-load.bin",
                "--cache-save", "/tmp/laguna-save.bin",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][1], "/v1/chat/completions")
        body = calls[0][2]
        self.assertEqual(body["cache_load"], "/tmp/laguna-load.bin")
        self.assertEqual(body["cache_save"], "/tmp/laguna-save.bin")

    def test_chat_forwards_sampling_aliases(self):
        calls = []

        def fake_call(args, method, path, body=None, stream=False, timeout=None):
            calls.append((method, path, body, stream, timeout))
            return {"id": "job-1"}

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main([
                "chat", "Hello",
                "--model", "laguna-s21",
                "--top-k", "5",
                "--top-p", "0.77",
                "--min-p", "0.1",
                "--seed", "42",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][1], "/v1/chat/completions")
        body = calls[0][2]
        self.assertEqual(body["top_k"], 5)
        self.assertEqual(body["top_p"], 0.77)
        self.assertEqual(body["min_p"], 0.1)
        self.assertEqual(body["seed"], 42)

    def test_start_forwards_cache_flags_to_runner(self):
        calls = []

        def fake_call(args, method, path, body=None, stream=False, timeout=None):
            calls.append((method, path, body))
            return {"id": "run-1"}

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main([
                "start", "--model", "k3",
                "--cache-load", "/tmp/k3-load.bin",
                "--cache-save", "/tmp/k3-save.bin",
                "--np", "12",
                "--layer", "1",
                "--tokens", "8",
                "--result-dir", "/shared/k3-run",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][1], "/runner/start")
        body = calls[0][2]
        self.assertEqual(body["cache_load"], "/tmp/k3-load.bin")
        self.assertEqual(body["cache_save"], "/tmp/k3-save.bin")
        self.assertEqual(body["model"], "k3")
        self.assertNotIn("tp_np", body)

    def test_start_forwards_context_and_system_cache_contract(self):
        calls = []

        def fake_call(args, method, path, body=None, stream=False, timeout=None):
            calls.append((method, path, body))
            return {"id": "run-1"}

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main([
                "start", "--model", "k3", "--context-id", "agent-1",
                "--system-prompt", "You are a coding agent.",
                "--system-prompt-cache-key", "repo-main",
                "--cache-scope", "context", "--np", "12", "--layer", "1",
                "--tokens", "8", "--result-dir", "/shared/k3-run",
            ])
        self.assertEqual(rc, 0)
        body = calls[0][2]
        self.assertEqual(body["context_id"], "agent-1")
        self.assertEqual(body["system_prompt"], "You are a coding agent.")
        self.assertEqual(body["system_prompt_cache_key"], "repo-main")
        self.assertEqual(body["cache_scope"], "context")

    def test_start_forwards_context_parallel_hint_to_runner(self):
        calls = []

        def fake_call(args, method, path, body=None, stream=False, timeout=None):
            calls.append((method, path, body))
            return {"id": "run-1"}

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main([
                "start", "--model", "k3",
                "--np", "72",
                "--tp-np", "12",
                "--cache-load", "/tmp/k3-load.bin",
                "--cache-save", "/tmp/k3-save.bin",
                "--layer", "1",
                "--tokens", "8",
                "--result-dir", "/shared/k3-run",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][1], "/runner/start")
        body = calls[0][2]
        self.assertEqual(body["tp_np"], 12)
        self.assertEqual(body["cache_load"], "/tmp/k3-load.bin")
        self.assertEqual(body["cache_save"], "/tmp/k3-save.bin")

    def test_start_forwards_context_parallel_for_common_counts(self):
        calls = []

        def fake_call(args, method, path, body=None, stream=False, timeout=None):
            calls.append((method, path, body))
            return {"id": "run-1"}

        for np_, tp in ((16, 1), (24, 3), (32, 4), (48, 4), (72, 12), (96, 24)):
            calls.clear()
            with mock.patch.object(cli, "call", side_effect=fake_call):
                rc = cli.main([
                    "start", "--model", "k3",
                    "--np", str(np_),
                    "--tp-np", str(tp),
                    "--layer", "1",
                    "--tokens", "8",
                    "--result-dir", "/shared/k3-run",
                ])
            self.assertEqual(rc, 0)
            self.assertEqual(calls[0][0], "POST")
            self.assertEqual(calls[0][1], "/runner/start")
            body = calls[0][2]
            self.assertEqual(body["np"], np_)
            self.assertEqual(body["tp_np"], tp)


    def test_kv_save_load_passes_through_action_and_path(self):
        calls = []

        def fake_call(args, method, path, body=None, stream=False, timeout=None):
            calls.append((method, path, body, stream, timeout))
            return {"ok": True}

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main([
                "kv", "load", "--model", "k3", "--path", "/shared/k3-load.bin",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][1], "/kv")
        body = calls[0][2]
        self.assertEqual(body["action"], "load")
        self.assertEqual(body["model"], "k3")
        self.assertEqual(body["path"], "/shared/k3-load.bin")

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main([
                "kv", "save", "--model", "k3", "--path", "/shared/k3-save.bin",
            ])
        self.assertEqual(rc, 0)
        body = calls[1][2]
        self.assertEqual(body["action"], "save")
        self.assertEqual(body["path"], "/shared/k3-save.bin")

    def test_kv_clear_action_requires_no_path(self):
        calls = []

        def fake_call(args, method, path, body=None, stream=False, timeout=None):
            calls.append(body)
            return {"ok": True}

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main(["kv", "clear", "--model", "k3"])
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0]["action"], "clear")
        self.assertNotIn("path", calls[0])

    def test_kv_stats_forwards_path_and_expected_np(self):
        calls = []

        def fake_call(args, method, path, body=None, stream=False, timeout=None):
            calls.append((method, path, body))
            return {"ok": True}

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main([
                "kv", "stats", "--model", "k3",
                "--path", "/shared/k3-cache", "--np", "12",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][0:2], ("POST", "/kv"))
        self.assertEqual(calls[0][2]["action"], "stats")
        self.assertEqual(calls[0][2]["path"], "/shared/k3-cache")
        self.assertEqual(calls[0][2]["np"], 12)

    def test_kv_clear_strips_path_if_provided(self):
        calls = []

        def fake_call(args, method, path, body=None, stream=False, timeout=None):
            calls.append(body)
            return {"ok": True}

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main([
                "kv", "clear", "--model", "k3", "--path", "/shared/ignore.bin",
            ])
        self.assertEqual(rc, 0)
        self.assertNotIn("path", calls[0])

    def test_chat_stream_forwards_stream_true(self):
        calls = []

        def fake_call(args, method, path, body=None, stream=False, timeout=None):
            calls.append((method, path, body, stream))
            if stream:
                return None
            return {"id": "job-1"}

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main([
                "chat", "Hello stream", "--model", "laguna-s21", "--stream",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][1], "/v1/chat/completions")
        self.assertTrue(calls[0][3])


    def test_kv_load_without_path_is_client_error(self):
        calls = []

        def fake_call(*_args, **_kwargs):
            calls.append("called")
            return {"ok": True}

        with mock.patch.object(cli, "call", side_effect=fake_call):
            rc = cli.main(["kv", "load", "--model", "k3"])
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
