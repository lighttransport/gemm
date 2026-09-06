#!/usr/bin/env python3
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import ds4f_serve as serve


class AgentInterfaceTest(unittest.TestCase):
    def test_responses_input_accepts_string_and_tool_continuation(self):
        got = serve.responses_input_to_openai({
            "instructions": "be concise",
            "input": "inspect the repo",
        })
        self.assertEqual(got, [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "inspect the repo"},
        ])
        got = serve.responses_input_to_openai({"input": [
            {"type": "function_call", "call_id": "c1", "name": "bash",
             "arguments": '{"command":"pwd"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ]})
        self.assertEqual(got[0]["tool_calls"][0]["id"], "c1")
        self.assertEqual(got[1], {"role": "tool", "tool_call_id": "c1", "content": "ok"})

    def test_stream_holds_internal_tool_marker(self):
        state = {"raw": "", "emitted": 0}
        self.assertEqual(serve.stream_visible_delta(state, "hello "), "hello ")
        self.assertEqual(serve.stream_visible_delta(state, "<tool_"), "")
        self.assertEqual(serve.stream_visible_delta(state, "call>{\"name\":\"bash\"}</tool_call>"), "")
        self.assertNotIn("<tool_call>", state["raw"][:state["emitted"]])

    def test_stream_holds_dsv4_tool_marker(self):
        state = {"raw": "", "emitted": 0}
        self.assertEqual(serve.stream_visible_delta(state, "answer\n\n<｜DS"), "answer")
        self.assertEqual(serve.stream_visible_delta(
            state, "ML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">"), "")

    def test_cache_is_durable_and_agent_specific(self):
        with tempfile.TemporaryDirectory() as root, tempfile.NamedTemporaryFile() as tok:
            old_root, old_tok = serve.AGENT_CACHE_ROOT, serve.TOK
            serve.AGENT_CACHE_ROOT, serve.TOK = root, tok.name
            try:
                def fake_infer(prompt, max_tokens, samp, **kwargs):
                    with open(kwargs["save_path"], "wb") as f:
                        f.write(b"0123456789abcdef")
                    return [1, 2], [], ""

                with mock.patch.object(serve, "encode", return_value=[10, 20]), \
                     mock.patch.object(serve, "infer", side_effect=fake_infer):
                    first = serve.prepare_agent_cache("codex", [{"role": "system", "content": "x"}], [], [1, 2, 3])
                    # A newly generated snapshot is immediately reusable; the
                    # triggering request must not prefill the prefix twice.
                    self.assertTrue(first[2])
                    self.assertEqual(first[3], 2)
                    second = serve.prepare_agent_cache("codex", [{"role": "system", "content": "x"}], [], [1, 2, 3])
                    self.assertTrue(second[2])
                    other = serve.prepare_agent_cache("claude-code", [{"role": "system", "content": "x"}], [], [1, 2, 3])
                    self.assertNotEqual(first[1], other[1])
                    self.assertTrue(os.path.isfile(first[1]))
                    with open(first[1].replace(".kv", ".json")) as f:
                        self.assertEqual(json.load(f)["agent"], "codex")
            finally:
                serve.AGENT_CACHE_ROOT, serve.TOK = old_root, old_tok

    def test_cache_accepts_current_codex_sized_prefix(self):
        with tempfile.TemporaryDirectory() as root, tempfile.NamedTemporaryFile() as tok:
            old_root, old_tok, old_limit = (serve.AGENT_CACHE_ROOT, serve.TOK,
                                            serve.CACHE_MAX_TOKENS)
            serve.AGENT_CACHE_ROOT, serve.TOK = root, tok.name
            serve.CACHE_MAX_TOKENS = 14336
            prefix = list(range(11000))
            try:
                def fake_infer(prompt, max_tokens, samp, **kwargs):
                    with open(kwargs["save_path"], "wb") as f:
                        f.write(b"0123456789abcdef")
                    return [], [], ""

                with mock.patch.object(serve, "encode", return_value=prefix), \
                     mock.patch.object(serve, "infer", side_effect=fake_infer):
                    got = serve.prepare_agent_cache(
                        "codex", [{"role": "system", "content": "large"}], [], prefix + [1])
                self.assertTrue(got[2])
                self.assertEqual(got[3], 11000)
            finally:
                serve.AGENT_CACHE_ROOT, serve.TOK, serve.CACHE_MAX_TOKENS = \
                    old_root, old_tok, old_limit

    def test_agent_prefix_includes_stable_messages_before_dynamic_user(self):
        messages = [
            {"role": "system", "content": "instructions"},
            {"role": "developer", "content": "stable environment"},
            {"role": "user", "content": "dynamic question"},
        ]
        prompt = serve.build_chat_prompt(messages, [])
        # A character tokenizer makes the required exact-prefix property clear.
        with mock.patch.object(serve, "encode", side_effect=lambda s: list(map(ord, s))):
            text, ids = serve.agent_prefix_text(messages, [], list(map(ord, prompt)))
        self.assertTrue(text.endswith("User: "))
        self.assertIn("stable environment", text)
        self.assertNotIn("dynamic question", text)
        self.assertEqual(ids, list(map(ord, text)))

    def test_agent_prefix_retreats_to_token_boundary(self):
        messages = [
            {"role": "system", "content": "instructions"},
            {"role": "developer", "content": "stable environment"},
            {"role": "user", "content": "dynamic"},
        ]
        prompt = serve.build_chat_prompt(messages, [])
        full_ids = list(map(ord, prompt))

        def boundary_encode(text):
            ids = list(map(ord, text))
            # Simulate a BPE token crossing the candidate boundary while the
            # same text one character shorter is a proper prompt prefix.
            if text.endswith("User: "):
                ids[-1] = 999999
            return ids

        with mock.patch.object(serve, "encode", side_effect=boundary_encode):
            text, ids = serve.agent_prefix_text(messages, [], full_ids)
        self.assertTrue(text.endswith("User:"))
        self.assertEqual(ids, full_ids[:len(ids)])

    def test_long_agent_cache_extends_existing_system_snapshot(self):
        with tempfile.TemporaryDirectory() as root, tempfile.NamedTemporaryFile() as tok:
            old_root, old_tok = serve.AGENT_CACHE_ROOT, serve.TOK
            serve.AGENT_CACHE_ROOT, serve.TOK = root, tok.name
            calls = []
            try:
                def fake_infer(prompt, max_tokens, samp, **kwargs):
                    calls.append(kwargs)
                    with open(kwargs["save_path"], "wb") as f:
                        f.write(b"0123456789abcdef")
                    return [], [], ""

                char_encode = lambda s: list(map(ord, s))
                system = [{"role": "system", "content": "instructions"}]
                richer = system + [
                    {"role": "developer", "content": "stable environment"},
                    {"role": "user", "content": "dynamic"},
                ]
                with mock.patch.object(serve, "encode", side_effect=char_encode), \
                     mock.patch.object(serve, "infer", side_effect=fake_infer):
                    system_prompt = serve.build_chat_prompt(system, [])
                    serve.prepare_agent_cache("codex", system, [], char_encode(system_prompt))
                    rich_prompt = serve.build_chat_prompt(richer, [])
                    got = serve.prepare_agent_cache("codex", richer, [], char_encode(rich_prompt))
                self.assertTrue(got[2])
                self.assertTrue(calls[1]["cache_load"])
                self.assertGreater(calls[1]["cached_tokens"], 0)
            finally:
                serve.AGENT_CACHE_ROOT, serve.TOK = old_root, old_tok

    def test_conversation_cache_paths_do_not_cross_agents(self):
        old_base = serve.BASE
        serve.BASE = "/tmp/ds4f-audit"
        old = dict(serve._conv)
        try:
            serve._conv.update({"agent": None, "path": None, "ids": None})
            with mock.patch.object(serve, "encode", return_value=[1, 2]), \
                    mock.patch.object(serve, "prepare_agent_cache",
                                      return_value=([1, 2], "/tmp/system.kv", False, 0)):
                serve._select_cache("codex", [], [], "prompt")
                _, _, _, codex_path, _ = serve._select_cache(
                    "claude-code", [], [], "prompt")
            self.assertEqual(codex_path, "/tmp/ds4f-audit.conv.claude-code")
        finally:
            serve.BASE = old_base
            serve._conv.update(old)

    def test_completion_parses_tool_calls_without_visible_marker(self):
        content, calls, finish = serve.parse_completion(
            'Need a command\n<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
            False)
        self.assertEqual(content, "Need a command")
        self.assertEqual(calls[0]["function"]["name"], "bash")
        self.assertEqual(finish, "tool_calls")

    def test_responses_previous_id_reconstructs_tool_loop(self):
        serve._response_contexts.clear()
        serve._response_context_order[:] = []
        serve._remember_response("resp-1", [
            {"role": "system", "content": "agent"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "bash", "arguments": "{}"}}]},
        ])
        got = serve._response_context_messages({
            "previous_response_id": "resp-1",
            "input": [{"type": "function_call_output", "call_id": "call-1",
                       "output": "result"}],
        })
        self.assertEqual(got[-1]["role"], "tool")
        self.assertEqual(got[-1]["tool_call_id"], "call-1")

    def test_response_context_survives_frontend_memory_reset(self):
        with tempfile.TemporaryDirectory() as root:
            old_root = serve.RESPONSE_STATE_DIR
            serve.RESPONSE_STATE_DIR = root
            try:
                serve._remember_response("resp-durable", [{"role": "user", "content": "hello"}],
                                         "ctx-durable")
                serve._response_contexts.clear(); serve._response_context_ids.clear()
                got = serve._response_context_messages({
                    "previous_response_id": "resp-durable", "input": "again"})
                self.assertEqual(got[0]["content"], "hello")
                self.assertEqual(serve._response_context_ids["resp-durable"], "ctx-durable")
            finally:
                serve.RESPONSE_STATE_DIR = old_root

    def test_context_id_validation(self):
        self.assertEqual(serve._body_context_id({"context_id": "ctx-1"}), "ctx-1")
        with self.assertRaises(ValueError):
            serve._body_context_id({"context_id": "x" * 257})


if __name__ == "__main__":
    unittest.main()
