"""Host-side launcher for Unreal Editor python scripts (headless, offscreen; Vulkan on Linux, DX12 on Windows).

usage: python -m server.vhuman.metahuman.launch <script.py> [args...] [--timeout S] [--no-lock] [--result PATH]
Runs UnrealEditor-Cmd with -ExecutePythonScript under the repository GPU lock, captures the log, and returns 0 only if the
script wrote its result JSON with ok=true. Never passes -nullrhi (MetaHuman previews/eye textures need the GPU).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from . import DATASET_ROOT, PROJECT, UE_EDITOR_CMD, UE_RHI, UE_SCRIPTS, WINDOWS

BASE_ARGS = ['-RenderOffScreen', '-unattended', '-nosplash', '-NoSound', '-ResX=1024', '-ResY=1024',
             '-NoMetaHumanAccountPortalLoginFallback', '-log', '-stdout', '-FullStdOutLogOutput', '-NoLoadStartupPackages']


def command(script, args, log_path, windows=WINDOWS, rhi=UE_RHI):
    """argv list on Linux; one command-line string on Windows, where UE wants -ExecutePythonScript="<script> <args>" quoted
    as a whole (subprocess.list2cmdline would quote the entire -flag=value token instead)."""
    script = Path(script)
    if not script.is_absolute():
        script = UE_SCRIPTS / script
    if windows:
        def q(a):
            a = str(a).replace('\\', '/')
            if '"' in a or ' ' in a:      # UE splits the quoted value on spaces; keep worker paths space-free
                raise ValueError(f'space or double quote in UE script argument: {a!r}')
            return a
        py = ' '.join([script.as_posix()] + [q(a) for a in args])
        parts = [f'"{UE_EDITOR_CMD}"', f'"{PROJECT}"', f'-ExecutePythonScript="{py}"', f"-{rhi}", *BASE_ARGS, f'-abslog="{log_path}"']
        return ' '.join(parts)
    py = ' '.join([str(script)] + [shlex.quote(a) for a in args])
    return [str(UE_EDITOR_CMD), str(PROJECT), f'-ExecutePythonScript={py}', f"-{rhi}", *BASE_ARGS, f'-abslog={log_path}']


def run(script, args=(), *, timeout=3600., use_lock=True, result=None, env_extra=None, min_free_mib=6144):
    logs = DATASET_ROOT / 'v1' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    log_path = logs / f'{Path(script).stem}-{stamp}.log'
    env = dict(os.environ, VHUMAN_MH_OUT=str(DATASET_ROOT))
    # the root filesystem (home) filled up on 2026-11-03: keep every UE cache on nvme01 (Zen DataPath is set in DefaultEngine.ini)
    if not WINDOWS:
        env['UE-LocalDataCachePath'] = str(PROJECT.parent.parent / 'DDC')
        env['XDG_CONFIG_HOME'] = str(PROJECT.parent.parent / 'xdg-config')     # UE Linux puts ~/.config/Epic (Zen, crash reports, user settings) here
        env.setdefault('DISPLAY', ':0')
    if env_extra:
        env.update(env_extra)
    cmd = command(script, list(args), log_path)
    if result:
        Path(result).unlink(missing_ok=True)
    print('launch:', cmd if isinstance(cmd, str) else ' '.join(shlex.quote(c) for c in cmd), flush=True)
    t0 = time.time()

    def _no_core():
        # the editor segfaults at shutdown after the script finished; without this apport spends minutes dumping a core
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def _run():
        try:
            proc = subprocess.run(cmd, env=env, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  preexec_fn=None if WINDOWS else _no_core)
            return proc.returncode
        except subprocess.TimeoutExpired:
            return -9
    if use_lock:
        from .. import gpu
        with gpu.device_session(min_free_mib, timeout=timeout, check_memory=False):
            code = _run()
            time.sleep(8)          # let the Zen service / crash reporter of this session wind down before the next editor starts
    else:
        code = _run()
    elapsed = time.time() - t0
    ok = None
    if result and Path(result).is_file():
        ok = bool(json.loads(Path(result).read_text()).get('ok'))
    print(f'exit {code} after {elapsed:.0f}s; result {result}: {ok}; log {log_path}', flush=True)
    return code, ok, log_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('script'); ap.add_argument('args', nargs='*')
    ap.add_argument('--timeout', type=float, default=3600.); ap.add_argument('--no-lock', action='store_true')
    ap.add_argument('--result', default=None)
    a = ap.parse_args()
    code, ok, _ = run(a.script, a.args, timeout=a.timeout, use_lock=not a.no_lock, result=a.result)
    sys.exit(0 if (ok if a.result else code == 0) else 1)


if __name__ == '__main__':
    main()
