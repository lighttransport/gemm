"""Regression tests for Qwen XML tool-call translation."""
import unittest

from qwen_tools import parse_calls, tool_registry


class QwenToolsTest(unittest.TestCase):
    def setUp(self):
        self.registry = tool_registry([{
            "type": "namespace",
            "name": "shell",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo text",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }],
        }, {
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch",
        }])

    def test_namespaced_function_call(self):
        text, calls = parse_calls(
            "<tool_call>\n<function=shell.echo>\n"
            "<parameter=text>\nhello\n</parameter>\n"
            "</function>\n</tool_call>", self.registry)
        self.assertEqual(text, "")
        self.assertEqual(calls[0]["type"], "function_call")
        self.assertEqual(calls[0]["name"], "echo")
        self.assertEqual(calls[0]["namespace"], "shell")
        self.assertEqual(calls[0]["arguments"], '{"text": "hello"}')

    def test_invalid_or_unknown_call_stays_text(self):
        text = "<tool_call><function=unknown></function></tool_call>"
        parsed, calls = parse_calls(text, self.registry)
        self.assertEqual(parsed, text)
        self.assertEqual(calls, [])

    def test_unique_namespaced_call_accepts_bare_name(self):
        text, calls = parse_calls(
            "<tool_call>\n<function=echo>\n"
            "<parameter=text>\nhello\n</parameter>\n"
            "</function>\n</tool_call>", self.registry)
        self.assertEqual(text, "")
        self.assertEqual(calls[0]["name"], "echo")
        self.assertEqual(calls[0]["namespace"], "shell")


if __name__ == "__main__":
    unittest.main()
