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
import collections
import hashlib
import json
import threading

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


def _normalized_calls(calls):
    out = []
    for call in calls or []:
        name = call.get("name", "")
        if call.get("namespace"):
            name = call["namespace"] + "." + name
        if call.get("type") == "custom_tool_call":
            out.append([name, {"input": call.get("input", "")}])
            continue
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except ValueError:
                pass
        out.append([name, arguments])
    return json.dumps(out, sort_keys=True, ensure_ascii=False)


class RawTurnCache:
    """Server-side memory of generated assistant turns.

    Clients do not always return the opaque raw blob: Claude Code drops
    thinking blocks from earlier turns on the first request of a resumed
    session, and plain Chat Completions clients never carry it.  Turns are
    therefore also remembered by the tool-call ids this server issued and by
    the hash of a text-only answer.  The ids are random and server-issued, so
    a returned id identifies the generated turn; tool names and the answer
    text must still match.  Arguments may differ: clients store tool inputs
    with defaults filled in (Claude Code adds ``replace_all: false``).
    """

    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.entries = collections.OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def text_key(text):
        return "text:" + hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def call_names(calls):
        return [name for name, _ in json.loads(_normalized_calls(calls))]

    def remember(self, raw, text, calls, ids):
        keys = [i for i in ids if i] or ([self.text_key(text)] if text.strip() else [])
        value = (raw, self.call_names(calls), text.strip())
        with self.lock:
            for key in keys:
                self.entries[key] = value
                self.entries.move_to_end(key)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)

    def remember_ref(self, ref, raw):
        """Store a streamed turn under the reference it was sent with."""
        with self.lock:
            self.entries[ref] = (raw, None, None)
            self.entries.move_to_end(ref)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)

    def resolve(self, messages):
        """Fill in ``raw`` for assistant turns that lost it."""
        with self.lock:
            for m in messages:
                if m.get("role") != "assistant" or isinstance(m.get("raw"), str):
                    continue
                ref = m.get("raw_ref")
                if ref and ref in self.entries:
                    m["raw"] = self.entries[ref][0]
                    self.entries.move_to_end(ref)
                    continue
                ids = m.get("call_ids") or []
                key = ids[0] if ids else self.text_key(content_text(m.get("content", "")))
                entry = self.entries.get(key)
                if entry is None:
                    continue
                raw, names, text = entry
                if (names == self.call_names(m.get("tool_calls")) and
                        (text == content_text(m.get("content", "")).strip())):
                    m["raw"] = raw
                    self.entries.move_to_end(key)
        return messages


def prefix_boundaries(messages, registry=None, thinking=False, effort=None):
    """Stable prompt prefixes the runner snapshots and shares across
    conversations: the whole system turn, and, when tools precede system
    text, the tools block alone.  Agents embed per-project details (paths,
    memory directories) in their system text, but the tool schemas, often
    the larger part, are identical across projects.  The tools boundary
    ends after "\n\n" before a letter, a clean BPE pre-token split."""
    frame = system_frame(messages, registry, thinking, effort)
    out = [frame] if frame else []
    body = system_text(messages)
    if not body:
        return out
    instructions = REASONING_INSTRUCTIONS[template_effort(effort)] if thinking else ""
    head = "<|im_start|>system\n" + (instructions + "\n\n" if instructions else "")
    if registry:
        head += tools_block(registry) + "\n\n"
        if frame.startswith(head) and body[:1].isalpha():
            out.insert(0, head)
    # Agents append per-directory context last (pi: "<cwd>...</cwd>"), so
    # all but the final paragraph is shared by sessions in the same project.
    # The runner drops a candidate that is not a clean token prefix.
    cut = body.rfind("\n\n")
    if cut > 0 and len(body) - cut < len(body) // 4:
        candidate = head + body[:cut + 2]
        if frame.startswith(candidate) and candidate not in out:
            out.insert(len(out) - 1, candidate)
    return out


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


class StreamSplitter:
    """Split streamed generation into reasoning and answer deltas.

    Text before ``</think>`` is reasoning (when thinking).  Answer text is
    streamed until a ``<tool_call>`` may begin; from there it is held back,
    since tool calls are delivered as structured deltas after parsing.
    Partial tag prefixes are held until they resolve.
    """

    THINK_END = "</think>"
    CALL = "<tool_call>"

    def __init__(self, thinking):
        self.reasoning = thinking
        self.pending = ""
        self.after_think = False
        self.in_call = False
        self.content = ""          # answer text already emitted

    @staticmethod
    def _hold(text, tag):
        """Length of a trailing prefix of ``tag`` in ``text``."""
        for n in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:n]):
                return n
        return 0

    def feed(self, piece):
        out = []
        self.pending += piece
        if self.reasoning:
            end = self.pending.find(self.THINK_END)
            if end < 0:
                keep = self._hold(self.pending, self.THINK_END)
                emit = self.pending[:len(self.pending) - keep]
                self.pending = self.pending[len(emit):]
                if emit:
                    out.append(("reasoning", emit))
                return out
            if end:
                out.append(("reasoning", self.pending[:end]))
            self.pending = self.pending[end + len(self.THINK_END):]
            self.reasoning = False
            self.after_think = True
        if self.after_think:
            self.pending = self.pending.lstrip("\n")
            if not self.pending:
                return out
            self.after_think = False
        if self.in_call:
            return out
        start = self.pending.find(self.CALL)
        if start >= 0:
            emit, self.in_call = self.pending[:start], True
            self.pending = self.pending[start:]
        else:
            keep = self._hold(self.pending, self.CALL)
            emit = self.pending[:len(self.pending) - keep]
            self.pending = self.pending[len(emit):]
        if emit:
            self.content += emit
            out.append(("content", emit))
        return out


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
            elif str(item.get("encrypted_content", "")).startswith("q38ref1:"):
                current["raw_ref"] = item["encrypted_content"]
                current["reasoning_content"] = "\n".join(
                    x.get("text", "") for x in item.get("summary") or [] if isinstance(x, dict))
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
            current = assistant_turn()
            current["tool_calls"].append(_call_from_item(item))
            current.setdefault("call_ids", []).append(item.get("call_id"))
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
            m["call_ids"] = [c.get("id") for c in m.get("tool_calls") or []
                             if isinstance(c, dict)]
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
