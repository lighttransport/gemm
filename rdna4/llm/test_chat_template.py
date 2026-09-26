"""Qwen framing and byte-exact continuation-prefix regressions."""
import json
from pathlib import Path
import unittest

from codex_server import chat_prefix, chat_prompt, fit_context, responses_output
from qwen_chat import (decode_raw, encode_raw, responses_input_messages,
                       split_generation)
from qwen_tools import tool_registry

try:
    import jinja2
except ImportError:          # pragma: no cover - optional reference renderer
    jinja2 = None

TEMPLATE = Path(__file__).with_name("qwen38_chat_template.jinja")


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

    def test_context_trimming_preserves_equal_message_order(self):
        old = {"role": "user", "content": "x" * 1000}
        first = {"role": "user", "content": "same"}
        second = {"role": "user", "content": "same"}
        kept = fit_context([old, first, second], 128, 0)
        self.assertEqual(len(kept), 2)
        self.assertIs(kept[0], first)
        self.assertIs(kept[1], second)

    def test_context_trimming_keeps_tool_exchange_as_one_turn(self):
        messages = [
            {"role": "user", "content": "x" * 1000},
            {"role": "user", "content": "Inspect the file."},
            {"role": "assistant", "content": "<tool_call>read</tool_call>"},
            {"role": "tool", "content": "file contents"},
        ]
        kept = fit_context(messages, 128, 0)
        self.assertEqual(kept, messages[1:])


def reference_render(messages, tools=None, **kwargs):
    """Render with the GGUF chat template itself (HF tojson semantics)."""
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                             extensions=["jinja2.ext.loopcontrols"])

    def raise_exception(message):
        raise ValueError(message)

    env.filters["tojson"] = lambda x, **_: json.dumps(x, ensure_ascii=False)
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(TEMPLATE.read_text())
    return template.render(messages=messages, tools=tools, **kwargs)


ECHO = {"type": "function", "function": {
    "name": "exec_command", "description": "Run a shell command.",
    "parameters": {"type": "object",
                   "properties": {"cmd": {"type": "string"},
                                  "timeout_ms": {"type": "number"}},
                   "required": ["cmd"]}}}


@unittest.skipUnless(jinja2 is not None and TEMPLATE.exists(), "jinja2 unavailable")
class TemplateParityTest(unittest.TestCase):
    """Our renderer must equal the checkpoint template byte for byte."""

    def conversation(self):
        return [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "Let me look.",
             "tool_calls": [{"name": "exec_command",
                             "arguments": {"cmd": "cat a.c", "timeout_ms": 500}},
                            {"name": "exec_command", "arguments": {"cmd": "ls"}}]},
            {"role": "tool", "content": "int main(){}"},
            {"role": "tool", "content": "a.c"},
            {"role": "assistant", "content": "", "reasoning_content": "Edit it.",
             "tool_calls": [{"name": "exec_command", "arguments": {"cmd": "make"}}]},
            {"role": "tool", "content": "ok"},
        ]

    def check(self, messages, thinking, effort=None, tools=True):
        kwargs = {"add_generation_prompt": True, "enable_thinking": thinking}
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        expected = reference_render(
            [{**m, "tool_calls": [{"function": c} for c in m.get("tool_calls", [])]}
             if m.get("tool_calls") else m for m in messages],
            [ECHO] if tools else None, **kwargs)
        registry = tool_registry([ECHO]) if tools else None
        self.assertEqual(chat_prompt(messages, registry, thinking, effort), expected)

    def test_tools_calls_and_grouped_tool_responses(self):
        for thinking in (False, True):
            for effort in ("low", "medium", "xhigh"):
                with self.subTest(thinking=thinking, effort=effort):
                    self.check(self.conversation(), thinking,
                               effort if thinking else None)

    def test_without_tools(self):
        messages = [{"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "bye"}]
        self.check(messages, False, tools=False)
        self.check(messages, True, "low", tools=False)


class StreamSplitterTest(unittest.TestCase):
    def run_pieces(self, pieces, thinking):
        from qwen_chat import StreamSplitter
        splitter, out = StreamSplitter(thinking), []
        for piece in pieces:
            out += splitter.feed(piece)
        return out, splitter.content

    def test_reasoning_answer_and_held_tool_call(self):
        out, content = self.run_pieces(
            ["Plan ", "it</th", "ink>\n\nOK ", "<tool", "_call>\n<function=x>"], True)
        self.assertEqual("".join(p for k, p in out if k == "reasoning"), "Plan it")
        self.assertEqual(content, "OK ")
        self.assertNotIn("<tool", "".join(p for _, p in out))

    def test_ordinary_angle_brackets_stream(self):
        out, content = self.run_pieces(["x <b", "> y"], False)
        self.assertEqual(content, "x <b> y")


class RawTurnReplayTest(unittest.TestCase):
    """A returned turn replays as the exact generated bytes."""

    def test_reasoning_blob_round_trip(self):
        registry = tool_registry([ECHO])
        messages = [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "go"}]
        prompt = chat_prompt(messages, registry, True, "medium")
        # Non-canonical spacing the parsed JSON could not reproduce.
        generated = ("Plan it.\n</think>\n\n<tool_call>\n<function=exec_command>\n"
                     "<parameter=cmd>\n  cat a.c\n</parameter>\n</function>\n</tool_call>")
        reasoning, answer = split_generation(generated, True)
        self.assertEqual(reasoning, "Plan it.")
        output = responses_output("resp-1", reasoning, "<think>\n" + generated, "",
                                  [{"type": "function_call", "id": "fc_1",
                                    "call_id": "call_1", "name": "exec_command",
                                    "arguments": json.dumps({"cmd": "cat a.c"})}])
        self.assertEqual(decode_raw(output[0]["encrypted_content"]),
                         "<think>\n" + generated)
        follow = responses_input_messages(
            [{"role": "user", "content": "go"}] + output +
            [{"type": "function_call_output", "call_id": "call_1", "output": "x"}])
        replay = chat_prompt([{"role": "system", "content": "sys"}] + follow,
                             registry, True, "medium")
        self.assertTrue(replay.startswith(prompt + generated + "<|im_end|>\n"))

    def test_foreign_reasoning_is_not_raw(self):
        self.assertIsNone(decode_raw("gAAAAB-opaque"))
        self.assertEqual(decode_raw(encode_raw("x\u00e9")), "x\u00e9")


if __name__ == "__main__":
    unittest.main()
