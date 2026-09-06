"""Non-thinking Qwen framing and byte-exact continuation-prefix regressions."""
import unittest

from codex_server import chat_prefix, chat_prompt


class ChatTemplateTest(unittest.TestCase):
    def test_system_developer_merge(self):
        messages = [{"role": "system", "content": " System rule. "},
                    {"role": "developer", "content": "Developer rule."},
                    {"role": "user", "content": "hello"}]
        self.assertEqual(chat_prefix(messages),
                         "<|im_start|>system\nSystem rule.\nDeveloper rule.<|im_end|>\n")
        self.assertTrue(chat_prompt(messages).startswith(chat_prefix(messages)))
        self.assertNotIn("<|im_start|>developer", chat_prompt(messages))

    def test_assistant_history_preserves_generation_prefix(self):
        messages = [{"role": "system", "content": "Answer directly."},
                    {"role": "user", "content": "Write C code."}]
        initial = chat_prompt(messages)
        answer = "```c\nint main(void) { return 0; }\n```"
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": "Add a print statement."}]
        self.assertTrue(chat_prompt(messages).startswith(initial + answer + "<|im_end|>\n"))
        self.assertEqual(chat_prompt(messages).count("<think>\n\n</think>\n\n"), 2)

    def test_empty_system_and_content_parts(self):
        messages = [{"role": "system", "content": " "},
                    {"role": "user", "content": [{"type": "input_text", "text": " hello "}]}]
        self.assertEqual(chat_prefix(messages), "")
        self.assertEqual(chat_prompt(messages),
                         "<|im_start|>user\nhello<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n")


if __name__ == "__main__":
    unittest.main()
