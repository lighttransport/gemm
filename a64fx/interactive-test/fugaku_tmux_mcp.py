#!/usr/bin/env python3
"""Minimal stdio MCP server for Fugaku interactive tmux sessions.

This intentionally exposes a small API around the validated workflow:

  local MCP client -> ssh fugaku -> ssh <pinned frontend> -> tmux

The extra pinned frontend hop matters because Fugaku's `ssh fugaku` entry point
can load-balance across frontend nodes, while tmux sockets live in frontend-local
/tmp.
"""

import json
import os
import shlex
import subprocess
import sys


SERVER_NAME = "fugaku-tmux-interactive"
SERVER_VERSION = "0.1.0"

DEFAULT_OUTER = os.environ.get("FUGAKU_SSH", "fugaku")
DEFAULT_FRONTEND = os.environ.get("FUGAKU_FRONTEND", "fn01sv03")
DEFAULT_WORKDIR = os.environ.get("FUGAKU_WORKDIR", "~/work/gemm/ds4p")
DEFAULT_SESSION = os.environ.get("FUGAKU_TMUX_SESSION", "ds4p-int")
DEFAULT_TRANSCRIPT = os.environ.get(
    "FUGAKU_TRANSCRIPT",
    "a64fx/interactive-test/mcp-live.transcript",
)
DEFAULT_LAUNCHER = os.environ.get("FUGAKU_INTERACTIVE_LAUNCHER", "sh interactive-1.sh")
DEFAULT_TIMEOUT = float(os.environ.get("FUGAKU_MCP_TIMEOUT", "30"))


def rpc_result(req_id, result):
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def rpc_error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def text_result(text, is_error=False):
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def tool_text(req_id, text, is_error=False):
    return rpc_result(req_id, text_result(text, is_error=is_error))


def arg_str(args, key, default):
    val = args.get(key, default)
    if val is None:
        return default
    return str(val)


def arg_int(args, key, default, lo, hi):
    try:
        val = int(args.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, val))


def nested_ssh_script(script, outer=DEFAULT_OUTER, frontend=DEFAULT_FRONTEND, timeout=DEFAULT_TIMEOUT):
    if outer in ("", "local", "none") and frontend in ("", "local", "none"):
        return subprocess.run(
            ["bash", "-lc", script],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    if outer in ("", "local", "none"):
        cmd = f"bash -lc {shlex.quote(script)}"
        return subprocess.run(
            ["ssh", frontend, cmd],
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    nested = f"ssh {shlex.quote(frontend)} {shlex.quote('bash -lc ' + shlex.quote(script))}"
    return subprocess.run(
        ["ssh", outer, nested],
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def run_remote(script, args):
    timeout = float(args.get("timeout_seconds", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT)
    cp = nested_ssh_script(
        script,
        outer=arg_str(args, "outer", DEFAULT_OUTER),
        frontend=arg_str(args, "frontend", DEFAULT_FRONTEND),
        timeout=timeout,
    )
    out = cp.stdout
    if cp.stderr:
        out += ("\n" if out else "") + "[stderr]\n" + cp.stderr
    return cp.returncode, out


def shell_quote(value):
    return shlex.quote(value)


def tool_start(args):
    session = arg_str(args, "session", DEFAULT_SESSION)
    workdir = arg_str(args, "workdir", DEFAULT_WORKDIR)
    transcript = arg_str(args, "transcript", DEFAULT_TRANSCRIPT)
    launcher = arg_str(args, "launcher", DEFAULT_LAUNCHER)
    replace = bool(args.get("replace", False))

    kill_line = f"tmux kill-session -t {shell_quote(session)} 2>/dev/null || true\n" if replace else ""
    script = f"""
set -eu
{kill_line}cd {workdir}
mkdir -p {shell_quote(os.path.dirname(transcript) or ".")}
if tmux has-session -t {shell_quote(session)} 2>/dev/null; then
  echo "tmux session already exists: {session}"
else
  tmux new-session -d -s {shell_quote(session)} \
    "cd {workdir} && script -q -f {shell_quote(transcript)}"
  tmux send-keys -t {shell_quote(session)} {shell_quote(launcher)} C-m
  echo "started tmux session: {session}"
  echo "transcript: {transcript}"
fi
tmux ls | grep -F {shell_quote(session)} || true
tmux capture-pane -pt {shell_quote(session)} -S -60 || true
"""
    code, out = run_remote(script, args)
    prefix = "OK" if code == 0 else f"ERROR exit={code}"
    return f"{prefix}\n{out}".rstrip()


def tool_send(args):
    session = arg_str(args, "session", DEFAULT_SESSION)
    command = arg_str(args, "command", "")
    if not command:
        return "ERROR: command is required"
    enter = bool(args.get("enter", True))
    suffix = " C-m" if enter else ""
    script = f"tmux send-keys -t {shell_quote(session)} {shell_quote(command)}{suffix}"
    code, out = run_remote(script, args)
    prefix = "OK" if code == 0 else f"ERROR exit={code}"
    return f"{prefix}\nsent: {command}\n{out}".rstrip()


def tool_capture(args):
    session = arg_str(args, "session", DEFAULT_SESSION)
    lines = arg_int(args, "lines", 120, 1, 5000)
    script = f"tmux capture-pane -pt {shell_quote(session)} -S -{lines}"
    code, out = run_remote(script, args)
    prefix = "OK" if code == 0 else f"ERROR exit={code}"
    return f"{prefix}\n{out}".rstrip()


def tool_tail(args):
    workdir = arg_str(args, "workdir", DEFAULT_WORKDIR)
    transcript = arg_str(args, "transcript", DEFAULT_TRANSCRIPT)
    lines = arg_int(args, "lines", 120, 1, 5000)
    script = f"cd {workdir} && tail -n {lines} {shell_quote(transcript)}"
    code, out = run_remote(script, args)
    prefix = "OK" if code == 0 else f"ERROR exit={code}"
    return f"{prefix}\n{out}".rstrip()


def tool_status(args):
    session = arg_str(args, "session", DEFAULT_SESSION)
    workdir = arg_str(args, "workdir", DEFAULT_WORKDIR)
    transcript = arg_str(args, "transcript", DEFAULT_TRANSCRIPT)
    script = f"""
set +e
echo "frontend=$(hostname)"
tmux has-session -t {shell_quote(session)} 2>/dev/null
echo "tmux_has_session=$?"
tmux ls 2>/dev/null | grep -F {shell_quote(session)} || true
cd {workdir}
if [ -f {shell_quote(transcript)} ]; then
  echo "transcript={transcript}"
  grep -aE "pjsub Job|Interactive job|SENTINEL" {shell_quote(transcript)} | tail -20 || true
else
  echo "transcript missing: {transcript}"
fi
pjstat 2>/dev/null | tail -n +1 | grep -E "STDIN|JOB_ID|492" || true
"""
    code, out = run_remote(script, args)
    prefix = "OK" if code == 0 else f"ERROR exit={code}"
    return f"{prefix}\n{out}".rstrip()


def tool_stop(args):
    session = arg_str(args, "session", DEFAULT_SESSION)
    exits = arg_int(args, "exits", 2, 1, 4)
    script_lines = []
    for _ in range(exits):
        script_lines.append(f"tmux send-keys -t {shell_quote(session)} exit C-m")
        script_lines.append("sleep 1")
    script_lines.append(f"tmux capture-pane -pt {shell_quote(session)} -S -80 || true")
    code, out = run_remote("\n".join(script_lines), args)
    prefix = "OK" if code == 0 else f"ERROR exit={code}"
    return f"{prefix}\n{out}".rstrip()


TOOLS = {
    "fugaku_interactive_start": {
        "description": "Start a pinned-frontend tmux+script session and launch pjsub --interact.",
        "handler": tool_start,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "session": {"type": "string"},
                "frontend": {"type": "string"},
                "outer": {"type": "string"},
                "workdir": {"type": "string"},
                "transcript": {"type": "string"},
                "launcher": {"type": "string"},
                "replace": {"type": "boolean"},
                "timeout_seconds": {"type": "number"},
            },
        },
    },
    "fugaku_interactive_send": {
        "description": "Send one command to the live interactive shell through tmux send-keys.",
        "handler": tool_send,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "session": {"type": "string"},
                "frontend": {"type": "string"},
                "outer": {"type": "string"},
                "command": {"type": "string"},
                "enter": {"type": "boolean"},
                "timeout_seconds": {"type": "number"},
            },
            "required": ["command"],
        },
    },
    "fugaku_interactive_capture": {
        "description": "Capture recent tmux pane text from the pinned frontend.",
        "handler": tool_capture,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "session": {"type": "string"},
                "frontend": {"type": "string"},
                "outer": {"type": "string"},
                "lines": {"type": "integer", "minimum": 1, "maximum": 5000},
                "timeout_seconds": {"type": "number"},
            },
        },
    },
    "fugaku_interactive_tail": {
        "description": "Tail the script(1) transcript file for the interactive session.",
        "handler": tool_tail,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "frontend": {"type": "string"},
                "outer": {"type": "string"},
                "workdir": {"type": "string"},
                "transcript": {"type": "string"},
                "lines": {"type": "integer", "minimum": 1, "maximum": 5000},
                "timeout_seconds": {"type": "number"},
            },
        },
    },
    "fugaku_interactive_status": {
        "description": "Report pinned frontend, tmux session, transcript markers, and PJM STDIN jobs.",
        "handler": tool_status,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "session": {"type": "string"},
                "frontend": {"type": "string"},
                "outer": {"type": "string"},
                "workdir": {"type": "string"},
                "transcript": {"type": "string"},
                "timeout_seconds": {"type": "number"},
            },
        },
    },
    "fugaku_interactive_stop": {
        "description": "Send exit commands to the tmux-owned interactive shell and capture the result.",
        "handler": tool_stop,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "session": {"type": "string"},
                "frontend": {"type": "string"},
                "outer": {"type": "string"},
                "exits": {"type": "integer", "minimum": 1, "maximum": 4},
                "timeout_seconds": {"type": "number"},
            },
        },
    },
}


def list_tools():
    return [
        {
            "name": name,
            "description": spec["description"],
            "inputSchema": spec["schema"],
        }
        for name, spec in TOOLS.items()
    ]


def handle(req):
    req_id = req.get("id")
    method = req.get("method", "")

    if method == "initialize":
        return rpc_result(
            req_id,
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return rpc_result(req_id, {"tools": list_tools()})
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name", "")
        args = params.get("arguments") or {}
        spec = TOOLS.get(name)
        if not spec:
            return rpc_error(req_id, -32602, f"unknown tool: {name}")
        try:
            text = spec["handler"](args)
            return tool_text(req_id, text, is_error=text.startswith("ERROR"))
        except subprocess.TimeoutExpired as exc:
            return tool_text(req_id, f"ERROR timeout after {exc.timeout}s", is_error=True)
        except Exception as exc:  # Keep MCP server alive on tool errors.
            return tool_text(req_id, f"ERROR {type(exc).__name__}: {exc}", is_error=True)

    return rpc_error(req_id, -32601, f"method not found: {method}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = rpc_error(None, -32700, f"parse error: {exc}")
        else:
            resp = handle(req)
        if resp is not None:
            print(json.dumps(resp, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
