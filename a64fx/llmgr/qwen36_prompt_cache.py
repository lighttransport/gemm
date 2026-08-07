#!/usr/bin/env python3
"""Warm and persist one llama.cpp slot for a Qwen36 agent system prompt.

The prompt file must contain the exact system message sent by the client.  The
saved file is a llama.cpp slot state, not a portable text cache; its metadata
binds it to the model and prompt digest.
"""

import argparse
import hashlib
import json
import os
import re
import urllib.request


def post(base, path, body):
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as response:
        return json.loads(response.read().decode("utf-8"))


def validate_save(saved, cache_path):
    """Reject a nominally successful response when no slot file was written."""
    if not isinstance(saved, dict):
        raise RuntimeError("slot save returned non-JSON object: %r" % (saved,))
    if saved.get("error"):
        raise RuntimeError("slot save failed: %s" % saved["error"])
    if not os.path.isfile(cache_path):
        raise RuntimeError("slot save returned without creating %s" % cache_path)
    size = os.path.getsize(cache_path)
    if size <= 0:
        raise RuntimeError("slot save created an empty file: %s" % cache_path)
    return size


def restore_slot(base, slot, filename):
    restored = post(base, "/slots/%d?action=restore" % slot,
                    {"filename": filename})
    if not isinstance(restored, dict) or restored.get("error"):
        raise RuntimeError("slot restore failed: %s" % (
            restored.get("error", restored) if isinstance(restored, dict)
            else restored,))
    return restored


def _content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return "" if content is None else str(content)


def extract_system_prompt(request):
    """Extract ordered system/developer instructions from supported wire JSON."""
    parts = []
    instructions = request.get("instructions")
    if instructions:
        parts.append(_content_text(instructions))
    system = request.get("system")
    if system:
        parts.append(_content_text(system))
    for message in request.get("messages", request.get("input", [])) or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") in ("system", "developer"):
            parts.append(_content_text(message.get("content")))
    return "\n\n".join(x for x in parts if x)


def _openai_tool(tool, namespace=None):
    """Convert Responses/Anthropic tool shapes to chat-completions tools."""
    if not isinstance(tool, dict):
        return None
    kind = tool.get("type")
    if kind == "function" and isinstance(tool.get("function"), dict):
        return {"type": "function", "function": tool["function"]}
    if kind in ("custom", "function"):
        name = tool.get("name")
        if not name:
            return None
        if namespace:
            name = "%s_%s" % (namespace, name)
        return {"type": "function", "function": {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", tool.get("input_schema", {
                "type": "object", "properties": {}})),
        }}
    if kind == "namespace":
        result = []
        for child in tool.get("tools", []):
            normalized = _openai_tool(child, tool.get("name"))
            if isinstance(normalized, list):
                result.extend(normalized)
            elif normalized:
                result.append(normalized)
        return result
    # Anthropic tools omit type or use a direct input_schema.
    if tool.get("name") and "input_schema" in tool:
        return [_openai_tool({"type": "custom", "name": tool["name"],
                              "description": tool.get("description", ""),
                              "input_schema": tool["input_schema"]})]
    return None


def extract_tools(request):
    """Return normalized tools from Anthropic, OpenAI, or Responses JSON."""
    raw = list(request.get("tools", []) or [])
    for item in request.get("input", []) or []:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            raw.extend(item.get("tools", []) or [])
    result = []
    for tool in raw:
        normalized = _openai_tool(tool)
        if isinstance(normalized, list):
            result.extend(x for x in normalized if x)
        elif normalized:
            result.append(normalized)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default=os.environ.get("QWEN36_SERVER",
                                                        "http://127.0.0.1:8081"))
    p.add_argument("--agent", required=True,
                   help="agent label used in the cache filename (letters, digits, ._-)")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-file",
                        help="exact system-message text captured from the agent")
    source.add_argument("--request-file",
                        help="OpenAI/Responses/Anthropic request JSON")
    p.add_argument("--output-dir", default=os.environ.get(
        "LLMGR_QWEN36_CACHE_DIR", "~/.cache/llmgr/qwen36"))
    p.add_argument("--model", default=os.environ.get("QWEN36_MODEL", ""))
    p.add_argument("--slot", type=int, default=0)
    p.add_argument("--reuse", action="store_true",
                   help="restore an existing matching slot instead of warming it")
    p.add_argument("--extract-only", action="store_true",
                   help="extract a request's system/developer prompt and save text")
    p.add_argument("--prompt-output",
                   help="output text path for --extract-only (default: stdout)")
    p.add_argument("--warm-user", default="hello",
                   help="user turn required by some chat templates (default: hello)")
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.agent):
        p.error("agent must contain only letters, digits, '.', '_' or '-'")

    if args.extract_only and not args.request_file:
        p.error("--extract-only requires --request-file")

    with open(args.request_file or args.prompt_file, "r", encoding="utf-8") as f:
        source_data = f.read()
    if args.request_file:
        try:
            request = json.loads(source_data)
        except ValueError as exc:
            p.error("request file is not valid JSON: %s" % exc)
        prompt = extract_system_prompt(request)
        tools = extract_tools(request)
    else:
        prompt = source_data
        tools = []
    if not prompt.strip():
        p.error("prompt file is empty")
    if args.extract_only:
        if args.prompt_output:
            with open(args.prompt_output, "w", encoding="utf-8") as f:
                f.write(prompt)
        else:
            print(prompt, end="")
        return
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    tool_data = json.dumps(tools, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    prompt_hash = hashlib.sha256((prompt + "\n\n" + tool_data).encode("utf-8")).hexdigest()
    model_hash = hashlib.sha256(os.path.abspath(args.model).encode()).hexdigest() \
        if args.model else "unknown"
    stem = "%s-%s" % (args.agent, prompt_hash[:16])
    filename = stem + ".slot.bin"
    cache_path = os.path.join(output_dir, filename)

    if args.reuse:
        if not os.path.isfile(cache_path):
            p.error("cache does not exist: %s" % cache_path)
        restored = restore_slot(args.server, args.slot, filename)
        print(json.dumps({"agent": args.agent, "cache": cache_path,
                          "restored": True, "restore_result": restored},
                         indent=2))
        return

    # max_tokens=0 leaves the slot after prompt evaluation, without appending
    # a generated token that would pollute the reusable prefix state.
    warm_request = {
        "model": args.model or "qwen36",
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": args.warm_user}],
        "max_tokens": 0,
        "temperature": 0,
        "stream": False,
    }
    if tools:
        warm_request["tools"] = tools
        warm_request["tool_choice"] = "auto"
    result = post(args.server, "/v1/chat/completions", warm_request)
    if "error" in result:
        raise RuntimeError("prompt warmup failed: %s" % result["error"])
    saved = post(args.server, "/slots/%d?action=save" % args.slot,
                 {"filename": filename})
    cache_bytes = validate_save(saved, cache_path)
    metadata = {
        "schema_version": 1,
        "agent": args.agent,
        "model": os.path.abspath(args.model) if args.model else None,
        "model_sha256": model_hash,
        "prompt_sha256": prompt_hash,
        "tool_count": len(tools),
        "tools_sha256": hashlib.sha256(tool_data.encode("utf-8")).hexdigest(),
        "prompt_source": os.path.abspath(args.request_file or args.prompt_file),
        "warm_user": args.warm_user,
        "slot": args.slot,
        "filename": filename,
        "cache_bytes": cache_bytes,
        "save_result": saved,
    }
    meta_path = os.path.join(output_dir, stem + ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"agent": args.agent, "cache": cache_path,
                     "metadata": meta_path, "save_result": saved}, indent=2))


if __name__ == "__main__":
    main()
