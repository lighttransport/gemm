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


def _responses_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(x.get("text", "")) for x in content
                       if isinstance(x, dict) and x.get("type") in
                       ("input_text", "output_text", "text"))
    return _text_content(content)


def normalize_messages(messages):
    out = []
    for message in messages or []:
        if not isinstance(message, dict) or not message.get("role"):
            raise ValueError("each message must be an object with a role")
        item = dict(message)
        item["content"] = _text_content(item.get("content"))
        calls = item.get("tool_calls") or []
        normalized_calls = []
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError("tool_calls entries must be objects")
            call = dict(call)
            fn = call.get("function") or {}
            if not isinstance(fn, dict):
                raise ValueError("tool_calls.function must be an object")
            fn = dict(fn)
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args)
                except ValueError:
                    fn["arguments"] = {"input": args}
            call["function"] = fn
            normalized_calls.append(call)
        if "tool_calls" in item:
            item["tool_calls"] = normalized_calls
        out.append(item)
    return out


def load_tokenizer(path):
    """Load a tokenizer from an explicit model configuration path."""
    if not path:
        raise ValueError("tokenizer path is required; set tokenizer or LLMGR_TOKENIZER")
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError("tokenizer file not found: %s" % path)
    return laguna_tok.Tok(path)


def native_request(body, tokenizer=None, tokenizer_path=None, chat=True):
    tok = tokenizer or load_tokenizer(tokenizer_path or body.get("tokenizer"))
    if chat:
        messages = normalize_messages(body.get("messages"))
        if not messages:
            raise ValueError("messages must be a non-empty array")
        thinking = body.get("enable_thinking", True)
        if body.get("reasoning_effort") == "none":
            thinking = False
        try:
            prompt = laguna_tok.render_chat(
                messages, add_generation_prompt=True, enable_thinking=thinking,
                tools=body.get("tools"))
        except (ImportError, SystemExit):
            prompt = "\n".join("%s: %s" % (m["role"], m["content"])
                               for m in messages) + "\nassistant:"
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
    req_n = body.get("n", 1)
    if req_n != 1:
        raise ValueError("OpenAI wrapper currently supports n=1 only (got %r)" % req_n)
    req = {"ids": ids, "max_new": int(max_new), "stream": True}
    temp = float(body.get("temperature", 1.0))
    req["sample"] = temp > 0
    if temp > 0:
        req["temp"] = temp
    for field in ("cache_load", "cache_save"):
        if body.get(field) is not None:
            req[field] = body[field]
    for source, target in (("top_p", "top_p"), ("top_k", "top_k"),
                           ("min_p", "min_p"), ("seed", "seed")):
        if body.get(source) is not None:
            req[target] = body[source]
    return req, tok


def responses_request(body):
    """Translate the useful non-streaming Responses API subset to chat input."""
    item_input = body.get("input")
    if isinstance(item_input, str):
        messages = [{"role": "user", "content": item_input}]
    elif isinstance(item_input, list):
        messages = []
        for item in item_input:
            if not isinstance(item, dict):
                raise ValueError("Responses input items must be objects")
            kind = item.get("type", "message")
            if kind == "message":
                role = item.get("role")
                if not role:
                    raise ValueError("Responses message items need a role")
                messages.append({"role": role,
                                 "content": _responses_text(item.get("content"))})
            elif kind in ("input_text", "output_text"):
                messages.append({"role": "user" if kind == "input_text" else "assistant",
                                 "content": str(item.get("text", ""))})
            elif kind == "function_call_output":
                messages.append({"role": "tool", "content":
                                 str(item.get("output", "")),
                                 "tool_call_id": item.get("call_id")})
            else:
                raise ValueError("unsupported Responses input item type: %s" % kind)
    else:
        raise ValueError("Responses request needs string or array input")
    if body.get("instructions") is not None:
        messages.insert(0, {"role": "system", "content":
                            _responses_text(body["instructions"])})
    out = dict(body)
    out["messages"] = messages
    if "max_output_tokens" in out and "max_completion_tokens" not in out:
        out["max_completion_tokens"] = out["max_output_tokens"]
    return out


def responses_response(body, chat_response, request_id=None):
    """Wrap a Chat Completions result in the Responses object shape."""
    choice = (chat_response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output = []
    content = message.get("content")
    if content is not None:
        output.append({"type": "message", "id": "msg_" + uuid.uuid4().hex,
                       "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": content,
                                    "annotations": []}]})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        output.append({"type": "function_call", "id": call.get("id"),
                       "call_id": call.get("id"), "name": fn.get("name"),
                       "arguments": fn.get("arguments", ""),
                       "status": "completed"})
    usage = chat_response.get("usage") or {}
    return {"id": request_id or "resp_" + uuid.uuid4().hex,
            "object": "response", "created_at": int(time.time()),
            "model": body.get("model", "laguna-s21"), "status": "completed",
            "output": output, "usage": usage}


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
