#!/usr/bin/env python3
import os
import sys
import io
import json
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
import models
import llmgr_server as server


class ServerModelResolutionTest(unittest.TestCase):
    def test_openai_aliases_resolve_in_http_server(self):
        self.assertEqual(server._resolve_adapter("laguna").name, "laguna")
        self.assertEqual(server._resolve_adapter("laguna-s21").name, "laguna")

    def test_internal_model_names_still_resolve(self):
        self.assertEqual(server._resolve_adapter("k3").name, "k3")

    def test_default_model_is_laguna(self):
        self.assertEqual(server._resolve_adapter().name, "laguna")

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(models.ConfigError):
            server._resolve_adapter("bogus-model")


class ServerKVRouteTest(unittest.TestCase):
    def _fake_handler(self):
        h = object.__new__(server.Handler)
        out = {}

        def send_json(obj, status=200):
            out["status"] = status
            out["body"] = obj
            return None

        def err(msg, status=400):
            out["status"] = status
            out["body"] = {"error": msg}
            return None

        h._send_json = send_json
        h._err = err
        return h, out

    def test_ds4f_wire_proxy_forwards_responses_request(self):
        h, out = self._fake_handler()
        response = SimpleNamespace(
            status=200,
            headers={"Content-Type": "application/json"},
            read=lambda: b'{"id":"resp-1","status":"completed"}',
            close=lambda: None,
        )
        runner = SimpleNamespace(port=8088)
        with mock.patch.object(server, "_ready_serve", return_value=runner), \
                mock.patch.object(server.urllib.request, "urlopen",
                                  return_value=response) as urlopen:
            h._proxy_protocol(models.get("ds4f"), "/v1/responses", {
                "model": "ds4f", "input": "hello"})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8088/v1/responses")
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["id"], "resp-1")

    def test_k3_kv_load_returns_cache_flags(self):
        h, out = self._fake_handler()
        with mock.patch.object(
                server, "_resolve_adapter", return_value=models.get("k3")):
            h._post_kv({"action": "load", "model": "k3", "path": "/tmp/k3-load.bin"})
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["action"], "load")
        self.assertEqual(out["body"]["restart_flags"],
                         ["--cache-load", "/tmp/k3-load.bin"])

    def test_kv_actions_recorded_on_child_when_child_present(self):
        h, out = self._fake_handler()
        child = SimpleNamespace(
            meta={},
            id="run-1",
            kind="oneshot",
            state="running",
        )
        with mock.patch.object(server, "_resolve_adapter", return_value=models.get("k3")), \
             mock.patch.object(server, "_children", {"run-1": child}):
            h._post_kv({"action": "load", "model": "k3", "id": "run-1",
                        "path": "/tmp/k3-load.bin"})
            self.assertEqual(out["status"], 200)
            self.assertEqual(out["body"]["action"], "load")
            h._post_kv({"action": "save", "model": "k3", "id": "run-1",
                        "path": "/tmp/k3-save.bin"})
            self.assertEqual(out["status"], 200)
            self.assertEqual(out["body"]["action"], "save")
        self.assertEqual(child.meta["kv"][0]["action"], "load")
        self.assertEqual(child.meta["kv"][0]["path"], "/tmp/k3-load.bin")
        self.assertEqual(child.meta["kv"][1]["action"], "save")
        self.assertEqual(child.meta["kv"][1]["path"], "/tmp/k3-save.bin")

    def test_k3_kv_save_requires_no_restart_path(self):
        h, out = self._fake_handler()
        with mock.patch.object(
                server, "_resolve_adapter", return_value=models.get("k3")):
            h._post_kv({"action": "save", "model": "k3", "path": "/tmp/k3-save.bin"})
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["action"], "save")
        self.assertEqual(out["body"]["restart_flags"],
                         ["--cache-save", "/tmp/k3-save.bin"])

    def test_kv_missing_path_is_rejected(self):
        h, out = self._fake_handler()
        with mock.patch.object(
                server, "_resolve_adapter", return_value=models.get("k3")):
            h._post_kv({"action": "load", "model": "k3"})
        self.assertEqual(out["status"], 400)
        self.assertIn("needs 'path'", out["body"]["error"])

    def test_kv_rejects_non_string_and_nul_paths(self):
        h, out = self._fake_handler()
        with mock.patch.object(server, "_resolve_adapter",
                              return_value=models.get("k3")):
            h._post_kv({"action": "load", "model": "k3", "path": 7})
            self.assertEqual(out["status"], 400)
            self.assertIn("must be a string", out["body"]["error"])
            h._post_kv({"action": "save", "model": "k3",
                        "path": "/shared/bad\x00cache"})
        self.assertEqual(out["status"], 400)
        self.assertIn("must not contain NUL", out["body"]["error"])

    def test_kv_clear_reports_unavailable_command(self):
        h, out = self._fake_handler()
        with mock.patch.object(
                server, "_resolve_adapter", return_value=models.get("k3")):
            h._post_kv({"action": "clear", "model": "k3"})
        self.assertEqual(out["status"], 200)
        self.assertFalse(out["body"]["applied"])
        self.assertIn("clear", out["body"]["note"])

    def test_chat_alias_for_openai_route(self):
        h, out = self._fake_handler()
        with mock.patch.object(server, "_ready_serve", return_value=None), \
             mock.patch.object(server.models, "get_by_openai_model") as resolve, \
             mock.patch.object(server, "laguna_openai"):
            resolve.return_value = models.get("k3")
            out["status"] = 200
            h._post_openai({"model": "k3"}, chat=True)
        self.assertEqual(out["status"], 400)
        self.assertIn("does not support OpenAI-style", out["body"]["error"])

    def test_kv_laguna_adapter_reports_no_restart_flags(self):
        h, out = self._fake_handler()
        with mock.patch.object(
                server, "_resolve_adapter", return_value=models.get("laguna")):
            h._post_kv({"action": "save", "model": "laguna", "path": "/tmp/laguna.bin"})
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["restart_flags"], [])
        self.assertIn("does not support cache", out["body"]["note"])

    def _fake_http_handler(self):
        h = self._fake_handler()[0]
        h._authorized = lambda: True
        h._read_json = lambda: {}
        h._split = lambda: (h.path, {})
        return h

    def test_get_models_exposes_openai_aliases_and_cache_capability(self):
        h, out = self._fake_handler()
        h._authorized = lambda: True
        h.path = "/models"
        h._split = lambda: ("/models", {})
        h.do_GET()
        self.assertEqual(out["status"], 200)
        body = out["body"]
        self.assertIn("k3", body)
        self.assertEqual(body["k3"]["supports_cache"], True)
        self.assertIn("laguna", body["laguna"]["openai_models"])
        self.assertIn("laguna-s21", body["laguna"]["openai_models"])

    def test_get_v1_models_returns_openai_aliases(self):
        h, out = self._fake_handler()
        h._authorized = lambda: True
        h.path = "/v1/models"
        h._split = lambda: ("/v1/models", {})
        h.do_GET()
        self.assertEqual(out["status"], 200)
        ids = {m["id"] for m in out["body"]["data"]}
        self.assertIn("laguna", ids)
        self.assertIn("laguna-s21", ids)
        self.assertNotIn("k3", ids)

    def test_get_runner_lists_child_info(self):
        h, out = self._fake_handler()
        h._authorized = lambda: True
        h.path = "/runner"
        h._split = lambda: ("/runner", {})
        child = mock.Mock()
        child.info.return_value = {"id": "run-1", "state": "running", "kind": "oneshot"}
        with mock.patch.object(server, "_children", {"run-1": child}):
            h.do_GET()
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["count"], 1)
        self.assertEqual(out["body"]["children"][0]["id"], "run-1")
        self.assertEqual(out["body"]["children"][0]["kind"], "oneshot")

    def test_get_inference_queue_exposes_info(self):
        h, out = self._fake_handler()
        h._authorized = lambda: True
        h.path = "/inference/queue"
        h._split = lambda: ("/inference/queue", {})
        with mock.patch.object(server._inference, "info", return_value={"queued": 0}):
            h.do_GET()
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"], {"queued": 0})

    def test_get_nodes_without_fanout_uses_local_commands_only(self):
        h, out = self._fake_handler()
        h._authorized = lambda: True
        h.path = "/nodes"
        h._split = lambda: ("/nodes", {"fanout": ["0"]})
        uname = mock.Mock()
        uname.nodename = "headnode"
        with mock.patch.object(server.os, "uname", return_value=uname), \
             mock.patch.object(server, "_job_info", return_value={"PJM_TEST": "1"}), \
             mock.patch.object(server, "_fanout") as fanout, \
             mock.patch.object(server, "_run_sync", side_effect=[
                 (0, "22:11 up"),
                 (0, "MemTotal:  128000 kB\nMemAvailable: 64000 kB"),
                 (0, "/local 100G 100G 0% /local"),
             ]):
            h.do_GET()
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["head"], "headnode")
        self.assertEqual(out["body"]["job"], {"PJM_TEST": "1"})
        self.assertNotIn("fanout_rc", out["body"])
        self.assertEqual(out["body"]["local"]["uptime"], "22:11 up")
        self.assertEqual(out["body"]["local"]["meminfo"], "MemTotal:  128000 kB\nMemAvailable: 64000 kB")
        fanout.assert_not_called()

    def test_get_nodes_fanout_invokes_fanout_and_reports_payload(self):
        h, out = self._fake_handler()
        h._authorized = lambda: True
        h.path = "/nodes"
        h._split = lambda: ("/nodes", {})
        uname = mock.Mock()
        uname.nodename = "headnode"
        with mock.patch.object(server.os, "uname", return_value=uname), \
             mock.patch.object(server, "_job_info", return_value={"PJM_TEST": "2"}), \
             mock.patch.object(server, "_fanout", return_value=(0, "rank0 host0\nrank1 host1")), \
             mock.patch.object(server, "_run_sync", side_effect=[
                 (0, "23:11 up"),
                 (0, "MemTotal:  128000 kB\nMemAvailable: 64000 kB"),
                 (0, "/local 100G 100G 0% /local"),
             ]):
            h.do_GET()
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["head"], "headnode")
        self.assertEqual(out["body"]["job"], {"PJM_TEST": "2"})
        self.assertEqual(out["body"]["fanout_rc"], 0)
        self.assertEqual(out["body"]["fanout"], ["rank0 host0", "rank1 host1"])

    def test_get_nodes_with_invalid_timeout_fails_with_error(self):
        h, out = self._fake_handler()
        h._authorized = lambda: True
        h.path = "/nodes"
        h._split = lambda: ("/nodes", {"fanout": ["1"], "timeout": ["bad"]})
        uname = mock.Mock()
        uname.nodename = "headnode"
        with mock.patch.object(server.os, "uname", return_value=uname), \
             mock.patch.object(server, "_job_info", return_value={}), \
             mock.patch.object(server, "_run_sync", side_effect=[
                 (0, "23:11 up"),
                 (0, "MemTotal:  128000 kB\nMemAvailable: 64000 kB"),
                 (0, "/local 100G 100G 0% /local"),
             ]):
            h.do_GET()
        self.assertEqual(out["status"], 500)
        self.assertIn("invalid literal for int()", out["body"]["error"])

    def test_get_nodes_timeout_is_passed_to_fanout(self):
        h, out = self._fake_handler()
        h._authorized = lambda: True
        h.path = "/nodes"
        h._split = lambda: ("/nodes", {"timeout": ["42"]})
        uname = mock.Mock()
        uname.nodename = "headnode"
        fanout = mock.Mock(return_value=(7, "rank0 ok"))
        with mock.patch.object(server.os, "uname", return_value=uname), \
             mock.patch.object(server, "_job_info", return_value={}), \
             mock.patch.object(server, "_fanout", fanout), \
             mock.patch.object(server, "_run_sync", side_effect=[
                 (0, "23:11 up"),
                 (0, "MemTotal:  128000 kB\nMemAvailable: 64000 kB"),
                 (0, "/local 100G 100G 0% /local"),
             ]):
            h.do_GET()
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["fanout_rc"], 7)
        self.assertEqual(out["body"]["fanout"], ["rank0 ok"])
        fanout.assert_called_once()
        self.assertEqual(fanout.call_args[1]["timeout"], 42)

    def test_health_and_root_share_same_payload_shape(self):
        h, out = self._fake_handler()
        h._authorized = lambda: True
        with mock.patch.object(server.time, "monotonic", return_value=1000.0), \
             mock.patch.object(server, "_START_TIME", 10.0), \
             mock.patch.object(server, "VERSION", "test"), \
             mock.patch.object(server.os, "uname", return_value=mock.Mock(nodename="headnode")):
            h.path = "/health"
            h._split = lambda: ("/health", {})
            with mock.patch.object(server, "REPO", "/tmp/repo"), \
                 mock.patch.object(server.bhs, "_sessions", []):
                server._children = {}
                h.do_GET()
                health = out["body"]
                health_status = dict(out["body"])
            # verify / maps via rstrip logic by resetting and rerunning
            h.path = "/"
            h._split = lambda: ("/", {})
            out["status"] = None
            out["body"] = None
            h.do_GET()
            root = out["body"]
        self.assertEqual(health_status["ok"], True)
        self.assertEqual(health_status["service"], "llmgr")
        self.assertEqual(root["service"], "llmgr")

    def test_unknown_get_path_returns_404(self):
        h, out = self._fake_handler()
        h._authorized = lambda: True
        h.path = "/definitely-not-found"
        h._split = lambda: ("/definitely-not-found", {})
        h.do_GET()
        self.assertEqual(out["status"], 404)
        self.assertIn("not found", out["body"]["error"])

    def test_get_without_authorization_is_401(self):
        h, out = self._fake_handler()
        h._authorized = lambda: False
        h.path = "/health"
        h._split = lambda: ("/health", {})
        h.do_GET()
        self.assertEqual(out["status"], 401)
        self.assertEqual(out["body"], {"error": "unauthorized"})

    def test_completion_alias_calls_openai_chat_handler(self):
        h = self._fake_http_handler()
        h.path = "/chat/completions"
        calls = {}

        def post_openai(body, chat):
            calls["chat"] = chat
            return h._send_json({"ok": True})

        h._post_openai = post_openai
        with mock.patch.object(server, "laguna_openai"):
            h.do_POST()
        self.assertEqual(calls.get("chat"), True)
        self.assertEqual(calls, {"chat": True})

    def test_completion_alias_calls_openai_text_handler(self):
        h = self._fake_http_handler()
        h.path = "/completion"
        calls = {}

        def post_openai(body, chat):
            calls["chat"] = chat
            return h._send_json({"ok": True})

        h._post_openai = post_openai
        with mock.patch.object(server, "laguna_openai"):
            h.do_POST()
        self.assertEqual(calls.get("chat"), False)
        self.assertEqual(calls, {"chat": False})

    def test_v1_responses_route_dispatches_to_responses_handler(self):
        h = self._fake_http_handler()
        h.path = "/v1/responses"
        calls = {}

        def post_responses(body):
            calls["body"] = body
            return h._send_json({"ok": True})

        h._post_openai_responses = post_responses
        with mock.patch.object(server, "laguna_openai"):
            h.do_POST()
        self.assertEqual(calls["body"], {})

    def test_unknown_openai_model_is_rejected(self):
        h, out = self._fake_handler()
        h._post_openai({"model": "k3", "messages": [{"role": "user", "content": "hi"}]},
                       chat=True)
        self.assertEqual(out["status"], 400)
        self.assertIn("unknown OpenAI model", out["body"]["error"])

    def test_openai_cache_fields_forward_to_native_request(self):
        h, out = self._fake_handler()
        native = {"ids": [1, 2, 3], "max_new": 32, "stream": True}
        called = {}

        def fake_native_request(body, chat=False):
            called["body"] = body
            return native, object()

        job = SimpleNamespace(
            id="job-1",
            done=mock.Mock(wait=lambda *args, **kwargs: True),
            error=None,
            result={"ok": True},
            events=None,
        )

        with mock.patch.object(server, "_ready_serve", return_value=SimpleNamespace(port=8080)), \
             mock.patch.object(server.models, "get_by_openai_model") as resolve, \
             mock.patch.object(server.laguna_openai, "native_request", side_effect=fake_native_request), \
             mock.patch.object(server._inference, "submit", return_value=job):
            resolve.return_value = models.get("laguna")
            h._post_openai({
                "model": "laguna-s21",
                "messages": [{"role": "user", "content": "ping"}],
                "cache_load": "/tmp/serve-cache-load.bin",
                "cache_save": "/tmp/serve-cache-save.bin",
            }, chat=True)
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"], {"ok": True})
        self.assertEqual(called["body"]["cache_load"], "/tmp/serve-cache-load.bin")
        self.assertEqual(called["body"]["cache_save"], "/tmp/serve-cache-save.bin")

    def test_prompt_cache_key_maps_to_stable_private_k3_cache(self):
        adapter = models.get("k3")
        with tempfile.TemporaryDirectory() as state, \
                mock.patch.object(server, "STATE_DIR", state):
            body = {"model": "k3", "prompt_cache_key": "codex-prefix-v1",
                    "np": 12, "tp_np": 12, "layer": 1, "layers": 3}
            first = server._apply_prompt_cache(body, adapter)
            self.assertNotIn("cache_load", first)
            self.assertTrue(first["cache_save"].startswith(state))
            self.assertNotIn("codex-prefix-v1", first["cache_save"])
            os.makedirs(first["cache_save"])
            for index in range(12):
                with open(os.path.join(first["cache_save"],
                                       "k3_ep_cache_l001_003_n012_t048_g000_r%03d.bin" % index), "wb") as f:
                    f.write(b"x")
            second = server._apply_prompt_cache(body, adapter)
            self.assertEqual(second["cache_load"], first["cache_save"])
            self.assertEqual(second["cache_save"], first["cache_save"])

    def test_prompt_cache_key_does_not_override_explicit_paths(self):
        adapter = models.get("k3")
        body = {"model": "k3", "prompt_cache_key": "key",
                "cache_load": "/shared/load", "cache_save": "/shared/save"}
        self.assertIs(server._apply_prompt_cache(body, adapter), body)
        laguna_body = {"model": "laguna", "prompt_cache_key": "key"}
        self.assertIs(server._apply_prompt_cache(laguna_body,
                                                 models.get("laguna")),
                      laguna_body)

    def test_prompt_cache_key_isolates_layout_and_model(self):
        adapter = models.get("k3")
        with tempfile.TemporaryDirectory() as state, \
                mock.patch.object(server, "STATE_DIR", state):
            base = {"model": "k3", "prompt_cache_key": "same-prefix",
                    "np": 12, "tp_np": 12, "layer": 1, "layers": 3}
            paths = {
                server._apply_prompt_cache(dict(base), adapter)["cache_save"],
                server._apply_prompt_cache(dict(base, model="k3-alt"),
                                            adapter)["cache_save"],
                server._apply_prompt_cache(dict(base, layers=4),
                                            adapter)["cache_save"],
                server._apply_prompt_cache(dict(base, mla_cache="fp32"),
                                            adapter)["cache_save"],
                server._apply_prompt_cache(dict(base, np=6),
                                            adapter)["cache_save"],
                server._apply_prompt_cache(dict(base, tp_np=6),
                                            adapter)["cache_save"],
            }
            self.assertEqual(len(paths), 6)

    def test_prompt_cache_key_does_not_load_incomplete_shard_set(self):
        adapter = models.get("k3")
        with tempfile.TemporaryDirectory() as state, \
                mock.patch.object(server, "STATE_DIR", state):
            body = {"model": "k3", "prompt_cache_key": "partial",
                    "np": 12, "tp_np": 12}
            first = server._apply_prompt_cache(body, adapter)
            os.makedirs(first["cache_save"])
            for index in range(11):
                with open(os.path.join(first["cache_save"],
                                       "k3_ep_cache_l001_001_n012_t048_g000_r%03d.bin" % index), "wb") as f:
                    f.write(b"x")
            second = server._apply_prompt_cache(body, adapter)
            self.assertNotIn("cache_load", second)

    def test_prompt_cache_key_ignores_stale_layout_shards(self):
        adapter = models.get("k3")
        with tempfile.TemporaryDirectory() as state, \
                mock.patch.object(server, "STATE_DIR", state):
            body = {"model": "k3", "prompt_cache_key": "stale-layout",
                    "np": 12, "layer": 1, "layers": 3, "threads": 48}
            first = server._apply_prompt_cache(body, adapter)
            os.makedirs(first["cache_save"])
            for index in range(12):
                with open(os.path.join(first["cache_save"],
                                       "k3_ep_cache_l001_001_n012_t048_g000_r%03d.bin" % index), "wb") as f:
                    f.write(b"x")
            second = server._apply_prompt_cache(body, adapter)
            self.assertNotIn("cache_load", second)

    def test_prompt_cache_key_validation(self):
        adapter = models.get("k3")
        with self.assertRaises(ValueError):
            server._apply_prompt_cache({"prompt_cache_key": "\0bad"}, adapter)
        with self.assertRaises(ValueError):
            server._apply_prompt_cache({"prompt_cache_key": ""}, adapter)

    def test_openai_endpoint_applies_prompt_cache_key_before_submit(self):
        h, out = self._fake_handler()
        native = {"ids": [1], "max_new": 8, "stream": True}
        called = {}
        adapter = models.get("k3")
        job = SimpleNamespace(
            id="job-cache-key", done=mock.Mock(wait=lambda *args, **kwargs: True),
            error=None, result={"ok": True}, events=None)

        def fake_native_request(body, chat=False):
            called["body"] = body
            return native, object()

        with tempfile.TemporaryDirectory() as state, \
                mock.patch.object(server, "STATE_DIR", state), \
                mock.patch.object(server, "_ready_serve",
                                  return_value=SimpleNamespace(port=8080)), \
                mock.patch.object(server.models, "get_by_openai_model") as resolve, \
                mock.patch.object(server.laguna_openai, "native_request",
                                  side_effect=fake_native_request), \
                mock.patch.object(adapter, "responses_request",
                                  side_effect=server.laguna_openai.responses_request), \
                mock.patch.object(adapter, "responses_response",
                                  side_effect=server.laguna_openai.responses_response), \
                mock.patch.object(adapter, "native_request",
                                  side_effect=fake_native_request), \
                mock.patch.object(server._inference, "submit", return_value=job), \
                mock.patch.object(adapter, "supports_serve", True):
            resolve.return_value = adapter
            request = {"model": "k3", "prompt_cache_key": "codex-key",
                       "messages": [{"role": "user", "content": "ping"}]}
            h._post_openai(request, chat=True)
        self.assertEqual(out["status"], 200)
        self.assertIn("cache_save", called["body"])
        self.assertTrue(called["body"]["cache_save"].startswith(state))
        self.assertNotIn("cache_load", called["body"])
        self.assertNotIn("cache_save", request)

    def test_responses_route_translates_and_returns_response_object(self):
        h, out = self._fake_handler()
        native = {"ids": [1], "max_new": 8, "stream": True}
        called = {}

        def fake_native_request(body, chat=False):
            called["body"] = body
            called["chat"] = chat
            return native, object()

        job = SimpleNamespace(
            id="resp-job", done=mock.Mock(wait=lambda *args, **kwargs: True),
            error=None, events=None,
            result={"choices": [{"message": {"content": "ok"}}],
                    "usage": {"total_tokens": 1}},
        )
        with mock.patch.object(server, "_ready_serve",
                              return_value=SimpleNamespace(port=8080)), \
             mock.patch.object(server.models, "get_by_openai_model") as resolve, \
             mock.patch.object(server.laguna_openai, "native_request",
                               side_effect=fake_native_request), \
             mock.patch.object(server._inference, "submit", return_value=job):
            resolve.return_value = models.get("laguna")
            h._post_openai_responses({
                "model": "laguna-s21", "instructions": "short",
                "input": "hello", "cache_load": "/tmp/prefix.kv",
            })
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["object"], "response")
        self.assertEqual(out["body"]["id"], "resp-job")
        self.assertTrue(called["chat"])
        self.assertEqual(called["body"]["messages"][0]["role"], "system")
        self.assertEqual(called["body"]["cache_load"], "/tmp/prefix.kv")

    def test_responses_route_applies_k3_prompt_cache_key(self):
        h, out = self._fake_handler()
        native = {"ids": [1], "max_new": 8, "stream": True}
        called = {}
        adapter = models.get("k3")
        job = SimpleNamespace(
            id="resp-cache-job", done=mock.Mock(wait=lambda *args, **kwargs: True),
            error=None, events=None,
            result={"choices": [{"message": {"content": "ok"}}]},
        )

        def fake_native_request(body, chat=False):
            called["body"] = body
            called["chat"] = chat
            return native, object()

        with tempfile.TemporaryDirectory() as state, \
                mock.patch.object(server, "STATE_DIR", state), \
                mock.patch.object(server, "_ready_serve",
                                  return_value=SimpleNamespace(port=8080)), \
                mock.patch.object(server.models, "get_by_openai_model") as resolve, \
                mock.patch.object(server.laguna_openai, "native_request",
                                  side_effect=fake_native_request), \
                mock.patch.object(adapter, "responses_request",
                                  side_effect=server.laguna_openai.responses_request), \
                mock.patch.object(adapter, "responses_response",
                                  side_effect=server.laguna_openai.responses_response), \
                mock.patch.object(adapter, "native_request",
                                  side_effect=fake_native_request), \
                mock.patch.object(server._inference, "submit", return_value=job), \
                mock.patch.object(adapter, "supports_serve", True):
            resolve.return_value = adapter
            h._post_openai_responses({
                "model": "k3", "prompt_cache_key": "responses-prefix",
                "input": "hello",
            })
        self.assertEqual(out["status"], 200)
        self.assertTrue(called["chat"])
        self.assertIn("cache_save", called["body"])
        self.assertTrue(called["body"]["cache_save"].startswith(state))

    def test_responses_route_reuses_complete_k3_prompt_cache(self):
        h, out = self._fake_handler()
        native = {"ids": [1], "max_new": 8, "stream": True}
        called = {}
        adapter = models.get("k3")
        job = SimpleNamespace(
            id="resp-cache-hit", done=mock.Mock(wait=lambda *args, **kwargs: True),
            error=None, events=None,
            result={"choices": [{"message": {"content": "ok"}}]},
        )

        def fake_native_request(body, chat=False):
            called["body"] = body
            return native, object()

        with tempfile.TemporaryDirectory() as state, \
                mock.patch.object(server, "STATE_DIR", state), \
                mock.patch.object(server, "_ready_serve",
                                  return_value=SimpleNamespace(port=8080)), \
                mock.patch.object(server.models, "get_by_openai_model") as resolve, \
                mock.patch.object(server.laguna_openai, "native_request",
                                  side_effect=fake_native_request), \
                mock.patch.object(adapter, "responses_request",
                                  side_effect=server.laguna_openai.responses_request), \
                mock.patch.object(adapter, "responses_response",
                                  side_effect=server.laguna_openai.responses_response), \
                mock.patch.object(adapter, "native_request",
                                  side_effect=fake_native_request), \
                mock.patch.object(server._inference, "submit", return_value=job), \
                mock.patch.object(adapter, "supports_serve", True):
            resolve.return_value = adapter
            request = {"model": "k3", "prompt_cache_key": "responses-hit",
                       "np": 12, "input": "ping"}
            seed = server._apply_prompt_cache(dict(request), adapter)
            os.makedirs(seed["cache_save"])
            for index in range(12):
                with open(os.path.join(seed["cache_save"],
                                       "k3_ep_cache_l001_001_n012_t048_g000_r%03d.bin" % index), "wb") as f:
                    f.write(b"x")
            h._post_openai_responses(request)
        self.assertEqual(out["status"], 200)
        self.assertEqual(called["body"]["cache_load"],
                         called["body"]["cache_save"])

    def test_responses_route_skips_incomplete_k3_prompt_cache(self):
        h, out = self._fake_handler()
        native = {"ids": [1], "max_new": 8, "stream": True}
        called = {}
        adapter = models.get("k3")
        job = SimpleNamespace(
            id="resp-cache-partial", done=mock.Mock(wait=lambda *args, **kwargs: True),
            error=None, events=None,
            result={"choices": [{"message": {"content": "ok"}}]},
        )

        def fake_native_request(body, chat=False):
            called["body"] = body
            return native, object()

        with tempfile.TemporaryDirectory() as state, \
                mock.patch.object(server, "STATE_DIR", state), \
                mock.patch.object(server, "_ready_serve",
                                  return_value=SimpleNamespace(port=8080)), \
                mock.patch.object(server.models, "get_by_openai_model") as resolve, \
                mock.patch.object(server.laguna_openai, "native_request",
                                  side_effect=fake_native_request), \
                mock.patch.object(adapter, "responses_request",
                                  side_effect=server.laguna_openai.responses_request), \
                mock.patch.object(adapter, "responses_response",
                                  side_effect=server.laguna_openai.responses_response), \
                mock.patch.object(adapter, "native_request",
                                  side_effect=fake_native_request), \
                mock.patch.object(server._inference, "submit", return_value=job), \
                mock.patch.object(adapter, "supports_serve", True):
            resolve.return_value = adapter
            request = {"model": "k3", "prompt_cache_key": "responses-partial",
                       "np": 12, "input": "ping"}
            seed = server._apply_prompt_cache(dict(request), adapter)
            os.makedirs(seed["cache_save"])
            for index in range(11):
                with open(os.path.join(seed["cache_save"],
                                       "k3_ep_cache_l001_001_n012_t048_g000_r%03d.bin" % index), "wb") as f:
                    f.write(b"x")
            h._post_openai_responses(request)
        self.assertEqual(out["status"], 200)
        self.assertIn("cache_save", called["body"])
        self.assertNotIn("cache_load", called["body"])

    def test_responses_stream_emits_deltas_and_completed_response(self):
        h = object.__new__(server.Handler)
        h.send_response = mock.Mock()
        h.send_header = mock.Mock()
        h.end_headers = mock.Mock()
        h.wfile = io.BytesIO()
        h._sse = server.Handler._sse.__get__(h, server.Handler)
        events = mock.Mock()
        events.get.side_effect = [
            {"event": "token", "text": "<think>plan"},
            {"event": "token", "text": "</think>answer"},
            None,
        ]
        job = SimpleNamespace(
            id="resp-stream", events=events, error=None,
            result={"choices": [{"message": {"content": "answer"}}],
                    "usage": {"total_tokens": 2}},
        )
        h._stream_responses(job, {"model": "laguna-s21"})
        raw = h.wfile.getvalue().decode("utf-8")
        frames = [json.loads(line[6:]) for line in raw.splitlines()
                  if line.startswith("data: ") and line[6:] != "[DONE]"]
        self.assertEqual(frames[0]["type"], "response.created")
        self.assertEqual(frames[1]["type"],
                         "response.reasoning_summary_text.delta")
        self.assertEqual(frames[2]["type"], "response.output_text.delta")
        self.assertEqual(frames[-1]["type"], "response.completed")
        self.assertEqual(frames[-1]["response"]["id"], "resp-stream")
        self.assertIn("data: [DONE]", raw)

    def test_openai_completion_route_uses_text_mode_native_conversion(self):
        h, out = self._fake_handler()
        native = {"ids": [3, 2, 1], "max_new": 12, "stream": False}
        called = {}

        def fake_native_request(body, chat=False):
            called["chat"] = chat
            called["body"] = body
            return native, object()

        job = SimpleNamespace(
            id="job-1",
            done=mock.Mock(wait=lambda *args, **kwargs: True),
            error=None,
            result={"ok": True},
            events=None,
        )

        with mock.patch.object(server, "_ready_serve", return_value=SimpleNamespace(port=8080)), \
             mock.patch.object(server.models, "get_by_openai_model") as resolve, \
             mock.patch.object(server.laguna_openai, "native_request", side_effect=fake_native_request), \
             mock.patch.object(server._inference, "submit", return_value=job):
            resolve.return_value = models.get("laguna")
            h._post_openai({
                "model": "laguna",
                "prompt": "hello",
                "max_completion_tokens": 12,
            }, chat=False)
        self.assertEqual(out["status"], 200)
        self.assertEqual(called["chat"], False)
        self.assertEqual(out["body"], {"ok": True})
        self.assertEqual(called["body"]["prompt"], "hello")

    def test_openai_chat_stream_frames_reasoning_and_tool_calls(self):
        h = object.__new__(server.Handler)
        h.send_response = mock.Mock()
        h.send_header = mock.Mock()
        h.end_headers = mock.Mock()
        h.wfile = io.BytesIO()
        h._sse = server.Handler._sse.__get__(h, server.Handler)
        h._stream_openai = server.Handler._stream_openai.__get__(
            h, server.Handler)
        events = mock.Mock()
        events.get.side_effect = [
            {"event": "token", "text": "<think>inspect cache"},
            {"event": "token", "text": "</think>answer"},
            {"event": "token", "text": "<tool_call>lookup"},
            None,
        ]
        tool_call = {"id": "call-1", "type": "function",
                     "function": {"name": "lookup", "arguments": "{}"}}
        job = SimpleNamespace(
            id="job-stream", events=events, error=None,
            result={"choices": [{"message": {"tool_calls": [tool_call]},
                                  "finish_reason": "tool_calls"}]})

        h._stream_openai(job, {"model": "laguna-s21",
                               "enable_thinking": True}, chat=True)
        raw = h.wfile.getvalue().decode("utf-8")
        frames = [line[6:] for line in raw.splitlines()
                  if line.startswith("data: ")]
        self.assertEqual(frames[-1], "[DONE]")
        self.assertEqual(json.loads(frames[0])["choices"][0]["delta"]["role"],
                         "assistant")
        payloads = [json.loads(frame) for frame in frames[:-1]]
        self.assertEqual(payloads[1]["choices"][0]["delta"]["reasoning"],
                         "inspect cache")
        self.assertEqual(payloads[2]["choices"][0]["delta"]["content"],
                         "answer")
        self.assertEqual(payloads[-1]["choices"][0]["delta"]["tool_calls"],
                         [tool_call])
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"],
                         "tool_calls")

    def test_k3_runner_start_context_parallel_across_common_counts(self):
        h, out = self._fake_handler()

        def fake_new_child(kind, label, argv, env, cwd, port=None, meta=None):
            h._new_argv = argv
            return SimpleNamespace(id="run-42", state="running", port=None,
                                   log_path="/tmp/run.log")

        for np_, tp in ((16, 1), (24, 3), (32, 4), (48, 4), (72, 12), (96, 24)):
            with mock.patch.object(server, "_resolve_adapter",
                                  return_value=models.get("k3")), \
                 mock.patch.object(server, "_new_child", side_effect=fake_new_child):
                h._post_runner_start({
                    "model": "k3",
                    "np": np_,
                    "tp_np": tp,
                    "result_dir": "/tmp/k3-start-result",
                })
            self.assertEqual(out["status"], 202)
            self.assertEqual(h._new_argv[h._new_argv.index("--tp-nodes") + 1],
                             str(tp))

    def test_v1_aliases_call_openai_handlers(self):
        h = self._fake_http_handler()
        h.path = "/v1/chat/completions"
        calls = {}

        def post_openai(body, chat):
            calls["chat"] = chat
            return h._send_json({"ok": True})

        h._post_openai = post_openai
        with mock.patch.object(server, "laguna_openai"):
            h.do_POST()
        self.assertEqual(calls.get("chat"), True)

        calls.clear()
        h.path = "/v1/completions"
        with mock.patch.object(server, "laguna_openai"):
            h.do_POST()
        self.assertEqual(calls.get("chat"), False)

    def test_openai_n_greater_than_one_is_rejected(self):
        h, out = self._fake_handler()
        with mock.patch.object(server, "_ready_serve", return_value=SimpleNamespace(port=8080)), \
             mock.patch.object(server.models, "get_by_openai_model") as resolve, \
             mock.patch.object(server.laguna_openai, "native_request") as native_request:
            resolve.return_value = models.get("laguna")
            native_request.side_effect = ValueError(
                "OpenAI wrapper currently supports n=1 only")
            h._post_openai({"model": "laguna", "messages": [{"role": "user", "content": "hi"}], "n": 2}, chat=True)
        self.assertEqual(out["status"], 400)
        self.assertIn("OpenAI wrapper currently supports n=1 only", out["body"]["error"])

    def test_kv_stats_without_child_returns_empty_child_payload(self):
        h, out = self._fake_handler()
        with mock.patch.object(server, "_resolve_adapter", return_value=models.get("k3")):
            h._post_kv({"action": "stats", "model": "k3"})
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["action"], "stats")
        self.assertNotIn("child", out["body"])

    def test_kv_stats_reports_k3_cache_shards(self):
        h, out = self._fake_handler()
        with tempfile.TemporaryDirectory() as cache_dir:
            with open(os.path.join(cache_dir, "k3_ep_cache_a.bin"), "wb") as f:
                f.write(b"abc")
            with open(os.path.join(cache_dir, "k3_ep_cache_b.bin"), "wb") as f:
                f.write(b"12345")
            with open(os.path.join(cache_dir, "unrelated.log"), "wb") as f:
                f.write(b"ignored")
            with mock.patch.object(server, "_resolve_adapter",
                                  return_value=models.get("k3")):
                h._post_kv({"action": "stats", "model": "k3",
                            "path": cache_dir, "np": 2})
        self.assertEqual(out["status"], 200)
        self.assertEqual(out["body"]["cache"]["shards"], 2)
        self.assertEqual(out["body"]["cache"]["bytes"], 8)
        self.assertTrue(out["body"]["cache"]["complete"])

    def test_kv_rejects_unknown_action(self):
        h, out = self._fake_handler()
        with mock.patch.object(server, "_resolve_adapter", return_value=models.get("k3")):
            h._post_kv({"action": "recycle", "model": "k3"})
        self.assertEqual(out["status"], 400)
        self.assertIn("action must be", out["body"]["error"])

    def test_k3_runner_start_rejects_serve_mode(self):
        h, out = self._fake_handler()
        with mock.patch.object(server, "_resolve_adapter",
                              return_value=models.get("k3")):
            h._post_runner_start({"model": "k3", "mode": "serve", "port": 8080})
        self.assertEqual(out["status"], 400)
        self.assertIn("k3 has no serve mode", out["body"]["error"])

    def test_k3_runner_start_includes_cache_flags(self):
        h, out = self._fake_handler()
        captured = {}

        def fake_new_child(kind, label, argv, env, cwd, port=None, meta=None):
            captured["argv"] = argv
            captured["kind"] = kind
            captured["label"] = label
            return SimpleNamespace(id="run-42", state="running", port=None,
                                   log_path="/tmp/run.log")

        with mock.patch.object(server, "_resolve_adapter",
                              return_value=models.get("k3")), \
             mock.patch.object(server, "_new_child", side_effect=fake_new_child):
            h._post_runner_start({
                "model": "k3",
                "np": 12,
                "cache_load": "/tmp/miss.bin",
                "cache_save": "/tmp/hit.bin",
                "result-dir": "/tmp/k3-start-result",
            })
        self.assertEqual(out["status"], 202)
        self.assertEqual(captured["kind"], "oneshot")
        self.assertIn("--cache-load", captured["argv"])
        self.assertIn("/tmp/miss.bin", captured["argv"])
        self.assertIn("--cache-save", captured["argv"])
        self.assertIn("/tmp/hit.bin", captured["argv"])

    def test_k3_runner_start_includes_context_parallel(self):
        h, out = self._fake_handler()
        captured = {}

        def fake_new_child(kind, label, argv, env, cwd, port=None, meta=None):
            captured["argv"] = argv
            captured["kind"] = kind
            captured["label"] = label
            return SimpleNamespace(id="run-42", state="running", port=None,
                                   log_path="/tmp/run.log")

        with mock.patch.object(server, "_resolve_adapter",
                              return_value=models.get("k3")), \
             mock.patch.object(server, "_new_child", side_effect=fake_new_child):
            h._post_runner_start({
                "model": "k3",
                "np": 72,
                "tp_np": 12,
                "cache_load": "/tmp/miss.bin",
                "cache_save": "/tmp/hit.bin",
                "result-dir": "/tmp/k3-start-result",
            })
        self.assertEqual(out["status"], 202)
        self.assertEqual(captured["kind"], "oneshot")
        self.assertIn("--tp-nodes", captured["argv"])
        self.assertEqual(captured["argv"][captured["argv"].index("--tp-nodes") + 1], "12")

if __name__ == "__main__":
    unittest.main()
