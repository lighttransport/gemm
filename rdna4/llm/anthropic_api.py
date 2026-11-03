"""Anthropic Messages API (Claude Code) translation for the Qwen3.8 shim.

Requests become the same template messages as the OpenAI paths (see
qwen_chat), so rendering, tool calls and prefix caching are shared.  The raw
generated turn travels in the ``signature`` of a leading thinking block:
Messages API clients must return thinking blocks unmodified, so the next
request replays the turn byte for byte and the runner extends its live state.
"""
import json
import uuid

from qwen_chat import decode_raw, encode_raw

# Tokens of the template reasoning-effort levels, by thinking budget.
EFFORT_BY_BUDGET = ((4096, "low"), (16384, "medium"))


def thinking_request(req):
    """Return (requested, effort) from ``thinking`` and ``output_config``.

    Claude Code sends ``thinking: {"type": "adaptive"}`` with the effort in
    ``output_config.effort``; older clients send a ``budget_tokens``.
    """
    thinking = req.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") not in ("enabled", "adaptive"):
        return False, None
    config = req.get("output_config")
    if isinstance(config, dict) and isinstance(config.get("effort"), str):
        return True, config["effort"]
    budget = thinking.get("budget_tokens")
    effort = "xhigh"
    if isinstance(budget, (int, float)) and not isinstance(budget, bool):
        for limit, name in EFFORT_BY_BUDGET:
            if budget <= limit:
                effort = name
                break
    return True, effort


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and block.get("type") == "image":
                parts.append("[image omitted: this model accepts text only]")
        return "\n".join(parts)
    return ""


def system_messages(system):
    if isinstance(system, str):
        return [{"role": "system", "content": system}] if system else []
    if isinstance(system, list):
        # Claude Code's first block is a per-version billing header, not an
        # instruction; keeping it would split the shared system prefix
        # between CLI versions and entrypoints.
        text = "\n\n".join(b.get("text", "") for b in system
                           if isinstance(b, dict) and b.get("type") == "text" and
                           not b.get("text", "").startswith("x-anthropic-billing-header:"))
        return [{"role": "system", "content": text}] if text else []
    return []


def tool_definitions(tools):
    """Messages tools -> the function-tool shape tool_registry accepts."""
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        if tool.get("type") not in (None, "custom"):
            continue            # server tools (web search, ...) are not local
        out.append({"type": "function", "function": {
            "name": tool["name"], "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}}}})
    return out


def request_messages(req):
    """Messages request -> template messages (system first)."""
    messages = system_messages(req.get("system"))
    for message in req.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if content is None:
            content = ""
        if role == "assistant":
            turn = {"role": "assistant", "content": "", "tool_calls": []}
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "thinking":
                    raw = decode_raw(block.get("signature"))
                    if raw is not None:
                        turn["raw"] = raw
                    else:
                        if str(block.get("signature", "")).startswith("q38ref1:"):
                            turn["raw_ref"] = block["signature"]
                        turn["reasoning_content"] = block.get("thinking", "")
                elif kind == "text":
                    turn["content"] += block.get("text") or ""
                elif kind == "tool_use":
                    turn.setdefault("call_ids", []).append(block.get("id"))
                    turn["tool_calls"].append({"name": block.get("name", ""),
                                               "arguments": block.get("input", {})})
            messages.append(turn)
            continue
        # user: tool results become grouped tool turns, text a user turn.
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        pending = []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                if pending:
                    messages.append({"role": "user", "content": "\n".join(pending)})
                    pending = []
                result = _text(block.get("content", ""))
                if block.get("is_error"):
                    result = "Error: " + result
                messages.append({"role": "tool", "content": result})
            elif block.get("type") in ("text", "image"):
                pending.append(_text([block]))
        if pending:
            messages.append({"role": "user", "content": "\n".join(pending)})
    return messages


def response_content(reasoning, raw_turn, text, calls):
    """Content blocks of one assistant turn (thinking block always first)."""
    blocks = [{"type": "thinking", "thinking": reasoning,
               "signature": encode_raw(raw_turn)}]
    if text:
        blocks.append({"type": "text", "text": text})
    for call in calls:
        arguments = call.get("arguments")
        if call.get("type") == "custom_tool_call":
            tool_input = {"input": call.get("input", "")}
        else:
            try:
                tool_input = json.loads(arguments) if arguments else {}
            except ValueError:
                tool_input = {"input": arguments}
        blocks.append({"type": "tool_use",
                       "id": "toolu_" + uuid.uuid4().hex[:24],
                       "name": call["name"], "input": tool_input})
    return blocks


def stop_reason(calls, finish):
    if calls:
        return "tool_use"
    return "max_tokens" if finish == "length" else "end_turn"


def usage(prompt_tokens, cached, completion_tokens):
    return {"input_tokens": prompt_tokens - cached,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
            "output_tokens": completion_tokens}


def message_object(message_id, model, content, reason, use):
    return {"id": message_id, "type": "message", "role": "assistant",
            "model": model, "content": content, "stop_reason": reason,
            "stop_sequence": None, "usage": use}


def stream_start(message_id, model, prompt_tokens, cached):
    return ("message_start", {"type": "message_start", "message": message_object(
        message_id, model, [], None, usage(prompt_tokens, cached, 0))})
