import os
import sys
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

    def test_normalize_openai_content(self):
        got = api.normalize_messages([{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "ignored"}}]}])
        self.assertEqual(got[0]["content"], "hello")


if __name__ == "__main__":
    unittest.main()
