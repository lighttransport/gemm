#!/usr/bin/env python3
"""Anthropic Messages wire translation for the llmgr native chat path."""

from __future__ import absolute_import

import json
import time
import uuid


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(block.get("text", "")) for block in content
                       if isinstance(block, dict) and block.get("type") == "text")
    return "" if content is None else str(content)


def _input(block):
    value = block.get("input", {})
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {"input": value}
    return value if isinstance(value, dict) else {"input": value}


def request(body, model=None):
    """Translate an Anthropic Messages request to OpenAI chat fields."""
    messages = []
    system = body.get("system")
    if system is not None:
        messages.append({"role": "system", "content": _text(system)})
    for message in body.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            raise ValueError("Anthropic messages need user or assistant roles")
        content = message.get("content", "")
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})
            continue
        if not isinstance(content, list):
            raise ValueError("Anthropic message content must be text or blocks")
        text_parts = []
        tool_calls = []
        tool_results = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("Anthropic content blocks must be objects")
            kind = block.get("type")
            if kind == "text":
                text_parts.append(str(block.get("text", "")))
            elif kind == "tool_use":
                tool_calls.append({
                    "id": block.get("id"), "type": "function",
                    "function": {"name": block.get("name", ""),
                                 "arguments": json.dumps(_input(block),
                                                          ensure_ascii=False)},
                })
            elif kind == "tool_result":
                tool_results.append({
                    "role": "tool", "tool_call_id": block.get("tool_use_id"),
                    "content": _text(block.get("content", "")),
                })
            else:
                raise ValueError("unsupported Anthropic content block: %s" % kind)
        if message["role"] == "assistant" and tool_calls:
            messages.append({"role": "assistant", "content": _text(text_parts),
                             "tool_calls": tool_calls})
        elif text_parts or message["role"] == "assistant":
            messages.append({"role": message["role"], "content": _text(text_parts)})
        messages.extend(tool_results)
    if not messages:
        raise ValueError("messages must be a non-empty array")
    model = model or body.get("model")
    out = {"messages": messages,
           "max_tokens": int(body.get("max_tokens", 256)),
           "stream": bool(body.get("stream"))}
    if model:
        out["model"] = model
    for source, target in (("temperature", "temperature"),
                           ("top_p", "top_p"), ("stop_sequences", "stop"),
                           ("prompt_cache_key", "prompt_cache_key"),
                           ("cache_load", "cache_load"),
                           ("cache_save", "cache_save")):
        if body.get(source) is not None:
            out[target] = body[source]
    tools = body.get("tools") or []
    if tools:
        out["tools"] = [{"type": "function", "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        }} for tool in tools]
    return out


def response(body, chat_response, request_id=None):
    choice = (chat_response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") or []
    for call in calls:
        fn = call.get("function") or {}
        try:
            value = json.loads(fn.get("arguments", "{}"))
        except (TypeError, ValueError):
            value = {"input": fn.get("arguments", "")}
        content.append({"type": "tool_use", "id": call.get("id"),
                        "name": fn.get("name", ""), "input": value})
    usage = chat_response.get("usage") or {}
    stop = "tool_use" if calls else ("max_tokens" if choice.get("finish_reason") == "length"
                                     else "end_turn")
    return {"id": request_id or "msg_" + uuid.uuid4().hex,
            "type": "message", "role": "assistant",
            "model": body.get("model", "claude-compatible"),
            "content": content, "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": int(usage.get("prompt_tokens", 0)),
                       "output_tokens": int(usage.get("completion_tokens", 0))}}


def stream_start(body, request_id):
    return {"type": "message_start", "message": {
        "id": request_id, "type": "message", "role": "assistant",
        "model": body.get("model", "claude-compatible"), "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}}}


def stream_text(index, text):
    return {"type": "content_block_delta", "index": index,
            "delta": {"type": "text_delta", "text": text}}


def stream_stop(index):
    return {"type": "content_block_stop", "index": index}


def stream_done(chat_response):
    choice = (chat_response.get("choices") or [{}])[0]
    calls = (choice.get("message") or {}).get("tool_calls") or []
    stop = "tool_use" if calls else ("max_tokens" if choice.get("finish_reason") == "length"
                                     else "end_turn")
    usage = chat_response.get("usage") or {}
    return {"type": "message_delta", "delta": {"stop_reason": stop,
                                                   "stop_sequence": None},
            "usage": {"output_tokens": int(usage.get("completion_tokens", 0))}}
