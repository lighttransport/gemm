"""Translate Qwen XML tool calls to Responses API output items.

This module only transports calls; the client retains execution authority.
"""
import json
import re
import uuid


def tool_registry(tools, namespace=None):
    registry = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            child_namespace = tool.get("name")
            if isinstance(child_namespace, str) and child_namespace:
                registry.update(tool_registry(tool.get("tools", []), child_namespace))
            continue
        kind = tool.get("type")
        if kind not in ("function", "custom"):
            continue
        spec = tool.get("function", tool)
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str) or not spec["name"]:
            continue
        name = spec["name"]
        qualified = f"{namespace}.{name}" if namespace else name
        parameters = spec.get("parameters", {}) if kind == "function" else {
            "type": "object", "properties": {"input": {"type": "string"}},
            "required": ["input"]}
        if not isinstance(parameters, dict):
            parameters = {}
        registry[qualified] = {"name": name, "namespace": namespace, "type": kind,
                               "description": spec.get("description", ""),
                               "parameters": parameters}
    return registry


def tool_instructions(registry):
    if not registry:
        return ""
    definitions = [{"type": "function", "function": {
        "name": name, "description": spec["description"],
        "parameters": spec["parameters"]}} for name, spec in registry.items()]
    return ("# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
            + "\n".join(json.dumps(x, ensure_ascii=False) for x in definitions)
            + "\n</tools>\n\nIf calling a function, reply using this format with no suffix:\n"
              "<tool_call>\n<function=example_function_name>\n"
              "<parameter=example_parameter>\nvalue\n</parameter>\n"
              "</function>\n</tool_call>\n"
              "Specify all required parameters. Optional text may precede the call. "
              "For a custom tool, put its raw text in the input parameter.")


def render_call(name, arguments):
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return ("<tool_call>\n<function=" + name + ">\n" + "".join(
        "<parameter=" + key + ">\n"
        + (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
        + "\n</parameter>\n" for key, value in arguments.items())
        + "</function>\n</tool_call>")


def parse_calls(text, registry):
    pattern = r"<tool_call>\s*<function=([^>]+)>\s*(.*?)\s*</function>\s*</tool_call>"
    matches = list(re.finditer(pattern, text, re.S))
    if not matches or not registry:
        return text, []
    prefix = text[:matches[0].start()].rstrip()
    if prefix.count("```") % 2 or text[matches[-1].end():].strip():
        return text, []
    calls = []
    for index, match in enumerate(matches):
        if index and text[matches[index - 1].end():match.start()].strip():
            return text, []
        name, body = match.groups()
        name = name.strip()
        if not name:
            return text, []
        spec = registry.get(name)
        if spec is None:
            # Qwen commonly emits the bare function name even when the API
            # presents a namespaced tool. Accept that spelling only when it
            # maps to exactly one registered function; ambiguity remains
            # plain text rather than risking a call to the wrong tool.
            candidates = [item for qualified, item in registry.items()
                          if item["name"] == name and "." in qualified]
            if len(candidates) == 1:
                spec = candidates[0]
        if spec is None:
            return text, []
        arguments = {}
        parameter_pattern = r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>"
        parameters = list(re.finditer(parameter_pattern, body, re.S))
        if re.sub(parameter_pattern, "", body, flags=re.S).strip():
            return text, []
        properties = spec["parameters"].get("properties", {})
        for parameter in parameters:
            key, value = parameter.groups()
            key = key.strip()
            if not key:
                return text, []
            if key in arguments or key not in properties:
                return text, []
            if properties[key].get("type") != "string":
                try:
                    value = json.loads(value)
                except ValueError:
                    return text, []
            arguments[key] = value
        if any(key not in arguments for key in spec["parameters"].get("required", [])):
            return text, []
        item = {"id": "fc_" + uuid.uuid4().hex, "call_id": "call_" + uuid.uuid4().hex,
                "name": spec["name"], "status": "completed"}
        if spec["namespace"]:
            item["namespace"] = spec["namespace"]
        if spec["type"] == "custom":
            item.update(type="custom_tool_call", input=arguments["input"])
        else:
            item.update(type="function_call", arguments=json.dumps(arguments, ensure_ascii=False))
        calls.append(item)
    return prefix, calls


def call_events(response_id, items):
    """Buffered Responses tool-call events, in protocol order."""
    base = {"id": response_id, "object": "response", "status": "in_progress", "output": []}
    yield {"type": "response.created", "response": base}
    yield {"type": "response.in_progress", "response": base}
    for index, item in enumerate(items):
        field = "input" if item["type"] == "custom_tool_call" else "arguments"
        event_prefix = "response.custom_tool_call_input" if field == "input" else "response.function_call_arguments"
        common = {"response_id": response_id, "output_index": index, "item_id": item["id"]}
        yield {"type": "response.output_item.added", "response_id": response_id,
               "output_index": index, "item": {**item, "status": "in_progress", field: ""}}
        yield {"type": event_prefix + ".delta", **common, "delta": item[field]}
        yield {"type": event_prefix + ".done", **common, field: item[field]}
        yield {"type": "response.output_item.done", "response_id": response_id,
               "output_index": index, "item": item}
