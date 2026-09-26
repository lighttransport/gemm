"""Qwen3.8 chat rendering that matches the GGUF chat template byte for byte.

The resident runner reuses KV/recurrent state only for an exact token prefix,
so every request must render the conversation exactly as the previous turn's
prompt plus the tokens the model actually generated.  Two things make that
work for coding agents:

* Tool results, tool calls, reasoning frames and the tools block follow the
  checkpoint's template (``tokenizer.chat_template``) instead of an ad-hoc
  ChatML variant.
* Each assistant turn produced here is returned with an opaque
  ``encrypted_content`` blob holding the raw generated text.  Responses API
  clients (Codex) send it back verbatim, and the turn is then re-rendered
  from those exact bytes rather than from the parsed tool-call JSON.
"""
import base64
import json

# Opaque reasoning-item payload: "q38raw1:" + base64(raw assistant text).
RAW_PREFIX = "q38raw1:"

TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
TOOLS_FOOTER = (
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language "
    "BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your "
    "current knowledge and do not tell the user about function calls\n</IMPORTANT>")

REASONING_INSTRUCTIONS = {
    "xhigh": ("Reasoning effort is set to xhigh. Please think carefully through the task, "
              "validate key assumptions, consider plausible alternatives, and prioritize "
              "correctness, consistency, and clarity in the final answer."),
    "medium": "",
    "low": ("Reasoning effort is set to low. Keep your thinking brief and focused, moving "
            "directly to the conclusion without unnecessary elaboration."),
}

THINK_OPEN = "<think>\n"
THINK_EMPTY = "<think>\n\n</think>\n\n"


def template_effort(effort):
    """Map an OpenAI reasoning effort onto the template's three levels."""
    if effort in ("low", "minimal"):
        return "low"
    if effort == "medium":
        return "medium"
    return "xhigh"          # high, xhigh, max, or unspecified (template default)


def encode_raw(raw):
    return RAW_PREFIX + base64.b64encode(raw.encode("utf-8")).decode("ascii")


def decode_raw(blob):
    if not isinstance(blob, str) or not blob.startswith(RAW_PREFIX):
        return None
    try:
        return base64.b64decode(blob[len(RAW_PREFIX):], validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Responses uses input_text/output_text; Chat Completions uses text.
        return "".join(x.get("text", "") for x in content
                       if isinstance(x, dict) and
                       x.get("type") in ("text", "input_text", "output_text", None))
    return ""


def tools_block(registry):
    if not registry:
        return ""
    definitions = [{"type": "function", "function": {
        "name": name, "description": spec["description"],
        "parameters": spec["parameters"]}} for name, spec in registry.items()]
    return (TOOLS_HEADER + "".join("\n" + json.dumps(x, ensure_ascii=False)
                                   for x in definitions) + TOOLS_FOOTER)


def system_text(messages):
    """Leading system/developer messages, merged (the template allows one)."""
    parts = []
    for m in messages:
        if m.get("role") not in ("system", "developer"):
            break
        text = content_text(m.get("content", "")).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def system_frame(messages, registry=None, thinking=False, effort=None):
    """The stable leading system turn; the runner snapshots state after it."""
    instructions = REASONING_INSTRUCTIONS[template_effort(effort)] if thinking else ""
    body = system_text(messages)
    block = tools_block(registry)
    if block:
        return ("<|im_start|>system\n" + (instructions + "\n\n" if instructions else "")
                + block + ("\n\n" + body if body else "") + "<|im_end|>\n")
    if body:
        return ("<|im_start|>system\n" + (instructions + "\n\n" if instructions else "")
                + body + "<|im_end|>\n")
    if instructions:
        return "<|im_start|>system\n" + instructions + "<|im_end|>\n"
    return ""


def render_call(call):
    name = call.get("name", "")
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments else {}
        except ValueError:
            arguments = {"input": arguments}
    if not isinstance(arguments, dict):
        arguments = {"input": json.dumps(arguments, ensure_ascii=False)}
    out = "<tool_call>\n<function=" + name + ">\n"
    for key, value in arguments.items():
        value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        out += "<parameter=" + key + ">\n" + value + "\n</parameter>\n"
    return out + "</function>\n</tool_call>"


def render_assistant(m):
    """One assistant turn (without the trailing <|im_end|>)."""
    raw = m.get("raw")
    if isinstance(raw, str):
        return "<|im_start|>assistant\n" + raw
    reasoning = m.get("reasoning_content")
    reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
    content = content_text(m.get("content", "")).strip()
    out = "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content
    for index, call in enumerate(m.get("tool_calls") or []):
        if index == 0:
            out += ("\n\n" if content else "") + render_call(call)
        else:
            out += "\n" + render_call(call)
    return out


def render_messages(messages, registry=None, thinking=False, effort=None,
                    generation_prompt=True):
    out = [system_frame(messages, registry, thinking, effort)]
    leading = True
    for index, m in enumerate(messages):
        role = m.get("role", "user")
        if leading and role in ("system", "developer"):
            continue
        leading = False
        if role == "assistant":
            out.append(render_assistant(m) + "<|im_end|>\n")
        elif role == "tool":
            previous = messages[index - 1].get("role") if index else None
            following = messages[index + 1].get("role") if index + 1 < len(messages) else None
            if previous != "tool":
                out.append("<|im_start|>user")
            out.append("\n<tool_response>\n" + content_text(m.get("content", "")).strip()
                       + "\n</tool_response>")
            if following != "tool":
                out.append("<|im_end|>\n")
        else:
            # Later system/developer messages have no template slot; keep them
            # as user turns rather than dropping instructions.
            text = content_text(m.get("content", "")).strip()
            out.append("<|im_start|>user\n" + text + "<|im_end|>\n")
    if generation_prompt:
        out.append("<|im_start|>assistant\n" + generation_suffix(thinking))
    return "".join(out)


def generation_suffix(thinking):
    return THINK_OPEN if thinking else THINK_EMPTY


def split_generation(text, thinking):
    """Return (reasoning, answer) from the text generated after the prompt."""
    if not thinking:
        return "", text
    end = text.find("</think>")
    if end < 0:
        return text.strip(), ""
    return text[:end].strip(), text[end + len("</think>"):].lstrip("\n")


def _call_from_item(item):
    """Responses function/custom tool call item -> template tool call."""
    name = item.get("name", "")
    if item.get("namespace"):
        name = item["namespace"] + "." + name
    if item.get("type") == "custom_tool_call":
        return {"name": name, "arguments": {"input": item.get("input", "")}}
    return {"name": name, "arguments": item.get("arguments", "{}")}


def _tool_output_text(output):
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        text = content_text(output)
        if text:
            return text
    return json.dumps(output, ensure_ascii=False)


def responses_input_messages(value):
    """Normalize Responses input into template messages.

    Items of one model turn (reasoning, message, tool calls) become a single
    assistant message.  A reasoning item carrying our raw blob replaces the
    whole turn with the exact generated bytes.
    """
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    messages = []
    direct = []
    turn = None

    def flush():
        nonlocal turn
        if turn is not None:
            messages.append(turn)
        turn = None

    def assistant_turn():
        nonlocal turn
        if turn is None:
            turn = {"role": "assistant", "content": "", "tool_calls": []}
        return turn

    for item in value:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        role = item.get("role")
        if kind == "reasoning":
            if turn is not None and (turn.get("content") or turn["tool_calls"] or
                                     "raw" in turn or turn.get("reasoning_content")):
                flush()
            current = assistant_turn()
            raw = decode_raw(item.get("encrypted_content"))
            if raw is not None:
                current["raw"] = raw
            else:
                summary = item.get("summary") or item.get("content") or []
                current["reasoning_content"] = "\n".join(
                    s.get("text", "") for s in summary if isinstance(s, dict))
        elif role == "assistant":
            current = assistant_turn()
            if current["tool_calls"] or current.get("content"):
                flush()
                current = assistant_turn()
            current["content"] = content_text(item.get("content", ""))
        elif kind in ("function_call", "custom_tool_call"):
            assistant_turn()["tool_calls"].append(_call_from_item(item))
        elif kind in ("function_call_output", "custom_tool_call_output"):
            flush()
            messages.append({"role": "tool",
                             "content": _tool_output_text(item.get("output", ""))})
        elif role:
            flush()
            messages.append({"role": role, "content": item.get("content", "")})
        elif "content" in item:
            # Message-shaped item without a role: treat as user content
            # rather than silently dropping the request.
            flush()
            messages.append({"role": "user", "content": item["content"]})
        elif kind in ("input_text", "output_text", "text"):
            direct.append(item)
    flush()
    if direct:
        messages.append({"role": "user", "content": direct})
    return messages


def chat_input_messages(messages):
    """Normalize Chat Completions messages (OpenAI tool_calls format)."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        m = dict(m)
        if m.get("role") == "assistant":
            calls = []
            for call in m.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function", call)
                calls.append({"name": fn.get("name", ""),
                              "arguments": fn.get("arguments", "{}")})
            m["tool_calls"] = calls
            if m.get("content") is None:
                m["content"] = ""
            reasoning = m.get("reasoning_content", m.get("reasoning"))
            if isinstance(reasoning, str):
                m["reasoning_content"] = reasoning
            raw = decode_raw(m.get("raw_content"))
            if raw is not None:
                m["raw"] = raw
        out.append(m)
    return out
