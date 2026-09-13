#!/usr/bin/env python3
"""Load bash-over-HTTP connection settings for shell scripts and clients.

Project configuration overrides the user configuration.  The command-line
interface emits shell-safe assignments consumed by the launcher scripts; the
Python API is used by the HTTP client.
"""

import argparse
import json
import os
import shlex
import sys


def _config_paths(project_dir=None):
    explicit = os.environ.get("BASH_HTTP_CONFIG")
    if explicit:
        return [os.path.expanduser(explicit)]

    paths = []
    if project_dir:
        paths.append(os.path.join(os.path.abspath(project_dir),
                                  ".bash-over-http.json"))
    else:
        current = os.path.abspath(os.getcwd())
        while True:
            paths.append(os.path.join(current, ".bash-over-http.json"))
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

    config_home = os.environ.get("XDG_CONFIG_HOME",
                                 os.path.expanduser("~/.config"))
    paths.append(os.path.join(config_home, "bash-over-http", "setup.json"))
    return paths


def _read(path):
    try:
        with open(path) as stream:
            value = json.load(stream)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read bash-over-http config %s: %s" %
                         (path, exc))
    if not isinstance(value, dict):
        raise ValueError("bash-over-http config must be a JSON object: %s" % path)
    return value


def _merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load(project_dir=None):
    """Return merged config and the paths that contributed to it."""
    merged = {}
    used = []
    paths = _config_paths(project_dir)
    # Global is the base; project config wins when both are present.
    for path in reversed(paths):
        if os.path.isfile(path):
            merged = _merge(merged, _read(path))
            used.append(path)
    return _expand(merged), list(reversed(used))


def _get(config, *path):
    value = config
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def environment(config):
    """Map the documented JSON keys to existing launcher environment names."""
    values = {}

    def put(name, *path):
        value = _get(config, *path)
        if value is not None:
            values[name] = value

    put("LOCAL_DIR", "local", "dir")
    put("LOCAL_PORT", "local", "port")
    put("CONTROL_DIR", "local", "state_dir")
    put("REMOTE", "remote", "ssh_host")
    put("REMOTE_REPO", "remote", "dir")
    put("REMOTE_PORT", "remote", "port")
    put("FRONTEND_PORT", "remote", "port")
    put("FRONTEND_HOST", "remote", "hostname")
    put("FRONTEND_SSH_TARGET", "remote", "hostname")
    put("SERVER_HOST", "server", "host")
    put("SERVER_PORT", "server", "port")
    for name in ("PROJECT_ID", "RSCGRP", "NODES", "ELAPSE", "A64FX_MODE",
                 "GFSCACHE", "LOCALTMP_SIZE", "MAX_RETRY",
                 "MONITOR_INTERVAL", "HEALTH_EVERY", "KEEPALIVE_SECONDS",
                 "WAIT_TIME", "PORT_OFFSET"):
        put(name, "job", name.lower())
    return values


def shell_output(config, used):
    values = environment(config)
    values["CONFIG_FILE"] = used[-1] if used else ""
    return "\n".join("BASH_HTTP_%s=%s" % (key, shlex.quote(str(value)))
                     for key, value in sorted(values.items()))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args(argv)
    try:
        config, used = load(args.project_dir)
    except ValueError as exc:
        print("bash-over-http: %s" % exc, file=sys.stderr)
        return 2
    if args.shell:
        print(shell_output(config, used))
    else:
        print(json.dumps(config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
