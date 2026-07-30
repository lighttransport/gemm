#!/usr/bin/env python3
"""OpenAI request/response translation for the Laguna native token API."""

import json
import os
import re
import sys
import time
import uuid

TOK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "laguna-s21", "tools"))
sys.path.insert(0, TOK_DIR)
import laguna_tok                         # noqa: E402


def _text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") for x in content
                       if isinstance(x, dict) and x.get("type") == "text")
    return "" if content is None else str(content)


def normalize_messages(messages):
    out = []
    for message in messages or []:
        if not isinstance(message, dict) or not message.get("role"):
            raise ValueError("each message must be an object with a role")
        item = dict(message)
        item["content"] = _text_content(item.get("content"))
        for call in item.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except ValueError:
                    fn["arguments"] = {"input": args}
        out.append(item)
    return out


def native_request(body, tokenizer=None, chat=True):
    tok = tokenizer or laguna_tok.Tok(laguna_tok.TOKJSON)
    if chat:
        messages = normalize_messages(body.get("messages"))
        if not messages:
            raise ValueError("messages must be a non-empty array")
        thinking = body.get("enable_thinking", True)
        if body.get("reasoning_effort") == "none":
            thinking = False
        prompt = laguna_tok.render_chat(
            messages, add_generation_prompt=True, enable_thinking=thinking,
            tools=body.get("tools"))
        ids = tok.encode(prompt)
    else:
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            if not all(isinstance(x, int) for x in prompt):
                raise ValueError("only one text prompt or one token-id array is supported")
            ids = prompt
        else:
            ids = tok.encode(str(prompt), add_bos=True)
    max_new = body.get("max_completion_tokens", body.get("max_tokens", 256))
    req = {"ids": ids, "max_new": int(max_new), "stream": True}
    temp = float(body.get("temperature", 1.0))
    req["sample"] = temp > 0
    if temp > 0:
        req["temp"] = temp
    for source, target in (("top_p", "top_p"), ("top_k", "top_k"),
                           ("min_p", "min_p"), ("seed", "seed")):
        if body.get(source) is not None:
            req[target] = body[source]
    return req, tok


_TOOL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_ARG = re.compile(r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", re.S)


def parse_assistant(text):
    """Split Laguna reasoning/content/tool markup into OpenAI fields."""
    reasoning = ""
    content = text
    if "</think>" in content:
        reasoning, content = content.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "", 1)
    elif content.startswith("<think>"):
        reasoning, content = content[7:], ""
    calls = []
    for index, match in enumerate(_TOOL.finditer(content)):
        raw = match.group(1)
        name = raw.split("<arg_key>", 1)[0].strip()
        args = {}
        for key, value in _ARG.findall(raw):
            value = value.strip()
            try:
                args[key.strip()] = json.loads(value)
            except ValueError:
                args[key.strip()] = value
        calls.append({
            "id": "call_%s" % uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)},
            "index": index,
        })
    content = _TOOL.sub("", content).strip()
    return {"content": content or None, "reasoning": reasoning or None,
            "reasoning_content": reasoning or None, "tool_calls": calls or None}


def completion_response(body, text, native, chat=True, request_id=None):
    rid = request_id or ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
    created = int(time.time())
    finish = "tool_calls" if chat and parse_assistant(text)["tool_calls"] else \
             ("length" if native.get("stop") == "length" else "stop")
    if chat:
        parsed = parse_assistant(text)
        message = {"role": "assistant", "content": parsed["content"]}
        if parsed["reasoning"] is not None:
            message["reasoning"] = parsed["reasoning"]
            message["reasoning_content"] = parsed["reasoning_content"]
        if parsed["tool_calls"] is not None:
            for call in parsed["tool_calls"]:
                call.pop("index", None)
            message["tool_calls"] = parsed["tool_calls"]
        choice = {"index": 0, "message": message, "finish_reason": finish}
        kind = "chat.completion"
    else:
        choice = {"index": 0, "text": text, "finish_reason": finish,
                  "logprobs": None}
        kind = "text_completion"
    prompt_n = len(native.get("prompt_ids", []))
    completion_n = int(native.get("n", 0))
    return {"id": rid, "object": kind, "created": created,
            "model": body.get("model", "laguna-s21"), "choices": [choice],
            "usage": {"prompt_tokens": prompt_n,
                      "completion_tokens": completion_n,
                      "total_tokens": prompt_n + completion_n}}
