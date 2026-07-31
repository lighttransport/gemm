import os
import sys
import warnings
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import laguna_openai as api


class LagunaOpenAITest(unittest.TestCase):
    def test_reasoning_and_content(self):
        got = api.parse_assistant("<think>check carefully</think>The answer is 4.")
        self.assertEqual(got["reasoning"], "check carefully")
        self.assertEqual(got["content"], "The answer is 4.")

    def test_tool_call(self):
        got = api.parse_assistant(
            "</think><tool_call>weather<arg_key>city</arg_key>"
            "<arg_value>Tokyo</arg_value><arg_key>days</arg_key>"
            "<arg_value>2</arg_value></tool_call>")
        call = got["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "weather")
        self.assertIn('"days": 2', call["function"]["arguments"])

    def test_completion_response_exposes_tool_calls(self):
        body = {"model": "laguna-s21"}
        text = "</think><tool_call>add<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>"
        native = {"prompt_ids": [2, 3], "n": 2}
        got = api.completion_response(body, text, native, chat=True)
        self.assertEqual(got["object"], "chat.completion")
        message = got["choices"][0]["message"]
        self.assertIn("tool_calls", message)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "add")
        self.assertEqual(message["tool_calls"][0]["index"], 0)

    def test_normalize_openai_content(self):
        got = api.normalize_messages([{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "ignored"}}]}])
        self.assertEqual(got[0]["content"], "hello")

    def test_normalize_tool_arguments_does_not_mutate_openai_body(self):
        message = {"role": "assistant", "content": None,
                   "tool_calls": [{"id": "call-1", "type": "function",
                                   "function": {"name": "echo",
                                                "arguments": '{"x": 1}'}}]}
        got = api.normalize_messages([message])
        self.assertEqual(got[0]["tool_calls"][0]["function"]["arguments"],
                         {"x": 1})
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"],
                         '{"x": 1}')

    def test_malformed_tool_calls_are_client_errors(self):
        with self.assertRaises(ValueError):
            api.normalize_messages([{"role": "assistant", "tool_calls": [3]}])
        with self.assertRaises(ValueError):
            api.normalize_messages([{"role": "assistant",
                                      "tool_calls": [{"function": "echo"}]}])

    def test_cache_fields_are_forwarded(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            req, _tok = api.native_request({
                "messages": [{"role": "user", "content": "ping"}],
                "cache_load": "/tmp/cache-load.bin",
                "cache_save": "/tmp/cache-save.bin",
            })
        self.assertEqual(req["cache_load"], "/tmp/cache-load.bin")
        self.assertEqual(req["cache_save"], "/tmp/cache-save.bin")

    def test_openai_extension_fields_do_not_leak_into_native_request(self):
        req, _tok = api.native_request({
            "messages": [{"role": "user", "content": "ping"}],
            "prompt_cache_key": "agent-prefix-v1",
            "prompt_cache_retention": "24h",
            "parallel_tool_calls": False,
            "response_format": {"type": "text"},
        })
        self.assertNotIn("prompt_cache_key", req)
        self.assertNotIn("prompt_cache_retention", req)
        self.assertNotIn("parallel_tool_calls", req)
        self.assertNotIn("response_format", req)

    def test_responses_input_maps_instructions_and_function_output(self):
        got = api.responses_request({
            "model": "laguna-s21",
            "instructions": "Be concise",
            "input": [{"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}]},
                       {"type": "function_call_output", "call_id": "c1",
                        "output": "done"}],
            "max_output_tokens": 17,
            "cache_load": "/shared/prefix.kv",
        })
        self.assertEqual(got["messages"][0],
                         {"role": "system", "content": "Be concise"})
        self.assertEqual(got["messages"][1]["content"], "hi")
        self.assertEqual(got["messages"][2]["role"], "tool")
        self.assertEqual(got["max_completion_tokens"], 17)
        self.assertEqual(got["cache_load"], "/shared/prefix.kv")

    def test_responses_response_wraps_text_and_function_calls(self):
        got = api.responses_response(
            {"model": "laguna-s21"},
            {"choices": [{"message": {
                "content": "answer",
                "tool_calls": [{"id": "c1", "function": {
                    "name": "lookup", "arguments": "{}"}}]},
                "finish_reason": "tool_calls"}],
             "usage": {"total_tokens": 4}},
            request_id="resp-test")
        self.assertEqual(got["object"], "response")
        self.assertEqual(got["id"], "resp-test")
        self.assertEqual(got["output"][0]["content"][0]["text"], "answer")
        self.assertEqual(got["output"][1]["type"], "function_call")
        self.assertEqual(got["usage"]["total_tokens"], 4)

    def test_responses_rejects_unknown_input_items(self):
        with self.assertRaises(ValueError):
            api.responses_request({"input": [{"type": "computer_use"}]})
        with self.assertRaises(ValueError):
            api.responses_request({"input": ["not an object"]})

    def test_top_k_min_p_seed_are_forwarded(self):
        req, _tok = api.native_request({
            "messages": [{"role": "user", "content": "ping"}],
            "top_k": 3,
            "min_p": 0.2,
            "seed": 123,
        })
        self.assertEqual(req["top_k"], 3)
        self.assertEqual(req["min_p"], 0.2)
        self.assertEqual(req["seed"], 123)

    def test_max_tokens_legacy_alias(self):
        req, _tok = api.native_request({
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 32,
        })
        self.assertEqual(req["max_new"], 32)

    def test_n_must_be_one(self):
        with self.assertRaises(ValueError):
            api.native_request({
                "messages": [{"role": "user", "content": "ping"}],
                "n": 2,
            })


if __name__ == "__main__":
    unittest.main()
