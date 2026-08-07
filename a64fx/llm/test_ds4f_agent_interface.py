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
                    self.assertFalse(first[2])
                    second = serve.prepare_agent_cache("codex", [{"role": "system", "content": "x"}], [], [1, 2, 3])
                    self.assertTrue(second[2])
                    other = serve.prepare_agent_cache("claude-code", [{"role": "system", "content": "x"}], [], [1, 2, 3])
                    self.assertNotEqual(first[1], other[1])
                    self.assertTrue(os.path.isfile(first[1]))
                    with open(first[1].replace(".kv", ".json")) as f:
                        self.assertEqual(json.load(f)["agent"], "codex")
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


if __name__ == "__main__":
    unittest.main()
