#!/usr/bin/env python3
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import anthropic_api


class AnthropicApiTest(unittest.TestCase):
    def test_messages_tools_translate_to_openai_chat(self):
        got = anthropic_api.request({
            "model": "claude-sonnet-4",
            "system": "You are an agent",
            "max_tokens": 32,
            "tools": [{"name": "bash", "description": "run shell",
                       "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "inspect repo"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "call-1", "name": "bash",
                     "input": {"command": "pwd"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call-1",
                     "content": "ok"}]},
            ],
        })
        self.assertEqual(got["messages"][0]["role"], "system")
        self.assertEqual(got["messages"][2]["role"], "assistant")
        self.assertEqual(got["messages"][2]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(got["messages"][3]["role"], "tool")
        self.assertEqual(got["tools"][0]["function"]["name"], "bash")

    def test_response_and_stream_events_use_anthropic_shape(self):
        result = anthropic_api.response(
            {"model": "claude-sonnet-4"},
            {"choices": [{"message": {"content": "run tool",
                                        "tool_calls": [{"id": "call-1",
                                            "function": {"name": "bash",
                                                         "arguments": "{\"command\":\"pwd\"}"}}]},
                          "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 4, "completion_tokens": 3}},
            request_id="msg-1")
        self.assertEqual(result["type"], "message")
        self.assertEqual(result["content"][1]["type"], "tool_use")
        self.assertEqual(result["stop_reason"], "tool_use")
        self.assertEqual(anthropic_api.stream_text(0, "hi")["delta"]["type"],
                         "text_delta")
        self.assertEqual(anthropic_api.stream_done({"choices": [{
            "message": {}, "finish_reason": "stop"}]} )["delta"]["stop_reason"],
                         "end_turn")

    def test_metadata_cache_fields_are_forwarded(self):
        got = anthropic_api.request({
            "metadata": {"prompt_cache_key": "claude-prefix",
                          "cache_load": "/shared/cache"},
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4})
        self.assertEqual(got["prompt_cache_key"], "claude-prefix")
        self.assertEqual(got["cache_load"], "/shared/cache")


if __name__ == "__main__":
    unittest.main()
