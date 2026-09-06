#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
from qwen36_prompt_cache import (extract_system_prompt, extract_tools,
                                 restore_slot, validate_save)


class Qwen36PromptCacheTest(unittest.TestCase):
    def test_extracts_openai_system_and_developer_messages(self):
        request = {"messages": [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "ignored"},
            {"role": "developer", "content": [{"type": "text", "text": "rules"}]},
        ]}
        self.assertEqual(extract_system_prompt(request), "base\n\nrules")

    def test_extracts_responses_instructions_and_anthropic_system(self):
        self.assertEqual(extract_system_prompt({
            "instructions": "codex rules", "input": [
                {"role": "user", "content": "ignored"}]}), "codex rules")
        self.assertEqual(extract_system_prompt({
            "system": [{"type": "text", "text": "claude rules"}],
            "messages": []}), "claude rules")

    def test_empty_request_has_no_system_prompt(self):
        self.assertEqual(extract_system_prompt({"messages": []}), "")

    def test_extracts_anthropic_and_responses_tools(self):
        tools = extract_tools({
            "tools": [{"name": "read", "description": "read a file",
                       "input_schema": {"type": "object"}}],
            "input": [{"type": "additional_tools", "tools": [{
                "type": "namespace", "name": "functions", "tools": [{
                    "type": "custom", "name": "exec", "parameters": {
                        "type": "object"}}]}]}]})
        self.assertEqual([x["function"]["name"] for x in tools],
                         ["read", "functions_exec"])

    def test_save_requires_nonempty_slot_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "slot.bin")
            with self.assertRaises(RuntimeError):
                validate_save({}, path)
            with open(path, "wb") as stream:
                stream.write(b"slot")
            self.assertEqual(validate_save({"filename": "slot.bin"}, path), 4)

    def test_save_rejects_server_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "slot save failed"):
                validate_save({"error": "no slot"}, os.path.join(directory, "x"))

    def test_restore_posts_to_selected_slot(self):
        with mock.patch("qwen36_prompt_cache.post", return_value={"ok": True}) as post:
            self.assertEqual(restore_slot("http://server", 3, "slot.bin"),
                             {"ok": True})
        post.assert_called_once_with("http://server", "/slots/3?action=restore",
                                     {"filename": "slot.bin"})


if __name__ == "__main__":
    unittest.main()
