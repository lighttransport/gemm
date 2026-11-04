"""Drive MetaHuman UE jobs on the Windows worker over ssh (Linux side). Setup notes: server/vhuman/vhuman-win10.md.

usage: python -m server.vhuman.metahuman.winjob sync                          (repo server/ tree -> WIN_REPO, incl. uncommitted)
       python -m server.vhuman.metahuman.winjob render <id>[,<id>...] [--env Studio|VHStudio] [--tag T] [--timeout S]
       python -m server.vhuman.metahuman.winjob compare <id>[,<id>...] [--env ...]   (Windows renders vs the Linux ones)
       python -m server.vhuman.metahuman.winjob status <tag> | kill <tag>

render = push (params.json + cameras.json of each id and a batch file; nothing else leaves Linux) -> run render_batch.py
detached on Windows (WMI Win32_Process.Create, so closing ssh does not kill it; launch.py there, DX12) -> wait -> pull the
renders into DATASET_ROOT/win_v1/samples/<id>/renders (the Linux dataset is never overwritten).
The worker is a dedicated machine: launch.py takes its own lock there, so Windows jobs may overlap Linux GPU jobs.
Transfers are files over scp + tar.exe (the ssh shell is PowerShell 5.1, which re-encodes piped binary stdin).
"""
import argparse
import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

from . import DATASET_ROOT

HOST = os.environ.get('VHUMAN_WIN_HOST', '')    # ssh host/alias of the worker (not stored in the repo)
WIN_ROOT = 'D:/work/vhuman-win'              # job scripts, logs, transfer tarballs
WIN_REPO = 'D:/work/gemm-vhuman'             # plain copy of the repo's server/ tree (not a git checkout)
WIN_DATA = 'D:/data/vhuman-mh'               # worker dataset root (job-scoped copies only)
WIN_PROJECT = 'D:/work/vhuman-mh/MHGen/MHGen.uproject'
WIN_PY = 'D:/work/vhuman-win/venv/Scripts/python.exe'
ROOT = Path(__file__).resolve().parents[3]
LOCAL_TMP = ROOT / 'tmp' / 'win'
NOISE = ('post-quantum', 'store now, decrypt later', 'server may need to be upgraded')


def ssh(ps, timeout=600, check=True):
    """Run a PowerShell command on the worker; returns stdout without the OpenSSH key-exchange warnings."""
    if not HOST:
        raise RuntimeError('set VHUMAN_WIN_HOST to the Windows worker ssh host')
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST, ps], capture_output=True, text=True, timeout=timeout)
    out = '\n'.join(line for line in (r.stdout + r.stderr).splitlines() if not any(n in line for n in NOISE)).strip()
    if check and r.returncode != 0:
        raise RuntimeError(f'ssh failed ({r.returncode}): {out}')
    return out


def scp_to(local, remote):
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', str(local), f'{HOST}:{remote}'], check=True, timeout=1800)


def scp_from(remote, local):
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', f'{HOST}:{remote}', str(local)], check=True, timeout=1800)


def win(path):
    return str(path).replace('/', '\\')


def env_block():
    return '\n'.join([
        f"$env:PYTHONPATH = '{win(WIN_REPO)}'",
        f"$env:VHUMAN_MH_DATA = '{win(WIN_DATA)}'",
        f"$env:VHUMAN_MH_PROJECT = '{win(WIN_PROJECT)}'",
        "$env:VHUMAN_UE_RHI = 'dx12'",
        f"$env:TEMP = '{win(WIN_ROOT)}\\tmp'; $env:TMP = $env:TEMP",
    ])


def job_script(tag, module_args, timeout):
    """PowerShell job: env, launch.py (blocking inside the detached process), exit code + finish time to <tag>.exit."""
    logs = f'{win(WIN_ROOT)}\\logs'
    args = ' '.join(module_args)
    return '\n'.join([
        env_block(),
        f"Set-Location '{win(WIN_REPO)}'",
        f"\"pid $PID started $(Get-Date -Format s)\" | Set-Content '{logs}\\{tag}.pid'",
        f"& '{win(WIN_PY)}' -m server.vhuman.metahuman.launch {args} --timeout {int(timeout)} *> '{logs}\\{tag}.out'",
        f"\"exit $LASTEXITCODE finished $(Get-Date -Format s)\" | Set-Content '{logs}\\{tag}.exit'",
    ]) + '\n'


def sync():
    """Copy the tracked + untracked-but-not-ignored files under server/ (working tree state) to WIN_REPO."""
    files = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', 'server'], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split()
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    tar = LOCAL_TMP / 'server.tar'
    with tarfile.open(tar, 'w') as t:
        for f in files:
            if (ROOT / f).is_file():
                t.add(ROOT / f, arcname=f)
    ssh(f"New-Item -ItemType Directory -Force '{win(WIN_REPO)}','{win(WIN_ROOT)}\\logs','{win(WIN_ROOT)}\\tmp' | Out-Null; "
        f"if (Test-Path '{win(WIN_REPO)}\\server') {{ Remove-Item -Recurse -Force '{win(WIN_REPO)}\\server' }}")
    scp_to(tar, f'{WIN_ROOT}/server.tar')
    ssh(f"tar -xf '{win(WIN_ROOT)}\\server.tar' -C '{win(WIN_REPO)}'")
    print(f'synced {len(files)} files ({tar.stat().st_size / 1e6:.1f} MB) to {WIN_REPO}')
    return len(files)


def batch_for(ids, env):
    batch = dict(ids=list(ids), env=env)
    if env != 'Studio':
        full = json.loads((DATASET_ROOT / 'v1' / 'render_batches' / f'{env.lower()}.json').read_text())
        batch['lights'] = {i: full['lights'][i] for i in ids}
        batch['calibration'] = full.get('calibration')
    return batch


def push(ids, env, tag):
    """Inputs of a render batch: per id params.json + cameras.json, and the batch file itself."""
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    tar = LOCAL_TMP / f'{tag}_in.tar'
    with tarfile.open(tar, 'w') as t:
        for sid in ids:
            for f in ('params.json', 'cameras.json'):
                t.add(DATASET_ROOT / 'v1' / 'samples' / sid / f, arcname=f'v1/samples/{sid}/{f}')
        data = json.dumps(batch_for(ids, env), indent=1).encode()
        info = tarfile.TarInfo(f'v1/render_batches/{tag}.json'); info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    ssh(f"New-Item -ItemType Directory -Force '{win(WIN_DATA)}' | Out-Null")
    scp_to(tar, f'{WIN_ROOT}/{tag}_in.tar')
    ssh(f"tar -xf '{win(WIN_ROOT)}\\{tag}_in.tar' -C '{win(WIN_DATA)}'")
    return f'{WIN_DATA}/v1/render_batches/{tag}.json'


def start(tag, module_args, timeout):
    LOCAL_TMP.mkdir(parents=True, exist_ok=True)
    ps1 = LOCAL_TMP / f'{tag}.ps1'
    ps1.write_text(job_script(tag, module_args, timeout))
    scp_to(ps1, f'{WIN_ROOT}/{tag}.ps1')
    logs = f'{win(WIN_ROOT)}\\logs'
    cmd = f'powershell -NoProfile -ExecutionPolicy Bypass -File {win(WIN_ROOT)}\\{tag}.ps1'
    out = ssh(f"Remove-Item '{logs}\\{tag}.exit','{logs}\\{tag}.pid' -ErrorAction SilentlyContinue; "
              f"$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine='{cmd}'; CurrentDirectory='{win(WIN_ROOT)}'}}; "
              f"\"$($r.ReturnValue) $($r.ProcessId)\"")
    rc, pid = out.split()[-2:]
    if rc != '0':
        raise RuntimeError(f'Win32_Process.Create returned {rc}')
    print(f'started {tag} (powershell pid {pid})', flush=True)
    return int(pid)


def status(tag):
    logs = f'{win(WIN_ROOT)}\\logs'
    return ssh(f"Get-Content '{logs}\\{tag}.exit' -ErrorAction SilentlyContinue; Get-Content '{logs}\\{tag}.out' -Tail 3 -ErrorAction SilentlyContinue", check=False)


def wait(tag, timeout, poll=30):
    deadline = time.time() + timeout + 300
    logs = f'{win(WIN_ROOT)}\\logs'
    while time.time() < deadline:
        out = ssh(f"Get-Content '{logs}\\{tag}.exit' -ErrorAction SilentlyContinue", check=False)
        if out.startswith('exit'):
            return int(out.split()[1]) if out.split()[1].lstrip('-').isdigit() else None
        time.sleep(poll)
    raise TimeoutError(f'{tag} did not finish in {timeout:.0f}s')


def kill(tag):
    """Stop the job's powershell and its children (launch.py, UnrealEditor-Cmd) by PID, never by name."""
    logs = f'{win(WIN_ROOT)}\\logs'
    return ssh(f"$p = (Get-Content '{logs}\\{tag}.pid').Split()[1]; "
               "function Kids($id) { Get-CimInstance Win32_Process -Filter \"ParentProcessId=$id\" | % { Kids $_.ProcessId; $_.ProcessId } }; "
               "$all = @(Kids $p) + @($p); foreach ($i in $all) { Stop-Process -Id $i -Force -ErrorAction SilentlyContinue }; \"stopped $($all -join ',')\"")


def pull(ids, env, tag, dest=None):
    dest = Path(dest or DATASET_ROOT / 'win_v1')
    suffix = env.lower()
    names = ' '.join(f"'v1/samples/{sid}/renders'" for sid in ids)
    ssh(f"Set-Location '{win(WIN_DATA)}'; tar -cf '{win(WIN_ROOT)}\\{tag}_out.tar' {names}")
    local = LOCAL_TMP / f'{tag}_out.tar'
    scp_from(f'{WIN_ROOT}/{tag}_out.tar', local)
    with tarfile.open(local) as t:
        members = [m for m in t.getmembers() if m.isfile() and (f'__{suffix}.' in m.name or '__basecolor.' in m.name or 'renders_done' in m.name)]
        t.extractall(dest, members=members, filter='data')
    for sid in ids:
        done = dest / 'v1' / 'samples' / sid / 'renders' / ('renders_done.json' if env == 'Studio' else f'renders_done_{suffix}.json')
        rec = json.loads(done.read_text()) if done.is_file() else {}
        print(sid, 'ok' if rec.get('ok') else 'MISSING', len(rec.get('files', [])), 'files ->', done.parent, flush=True)
    return dest


def compare(ids, env, dest=None):
    """Windows (DX12, RX 570) vs Linux (Vulkan, RX 9070 XT) UE renders of the same head and cameras: relative linear
    difference and aligned skin-crop LPIPS / dE on the shared display transform (compare_render.report's)."""
    import numpy as np
    from ..face_parsing import FaceParser
    from ..reconstruction import ab_metrics as am
    from .compare_render import read_exr
    dest = Path(dest or DATASET_ROOT / 'win_v1'); parser = FaceParser(); rows = {}
    disp = lambda x: np.uint8(np.clip(np.where(x <= .0031308, 12.92 * x, 1.055 * np.clip(x, 0, None) ** (1 / 2.4) - .055), 0, 1) * 255 + .5)
    for sid in ids:
        for f in sorted((dest / 'v1' / 'samples' / sid / 'renders').glob(f'*__{env.lower()}.exr')):
            lin = DATASET_ROOT / 'v1' / 'samples' / sid / 'renders' / f.name
            if not lin.is_file():
                continue
            a = read_exr(lin)[..., :3].astype(np.float64); b = read_exr(f)[..., :3].astype(np.float64)
            m = (a.max(-1) > 1e-4) | (b.max(-1) > 1e-4)
            k = .6 / max(float(np.percentile(a[m], 95)), 1e-6)
            r = am.compare_region(disp(a * k), disp(b * k), parser, ['skin', 'nose'], aligned=True)
            cam = f.name.split('__')[0]
            rows[f'{sid}/{cam}'] = dict(rel_abs_diff=float(np.abs(a - b)[m].mean() / max(a[m].mean(), 1e-9)),
                                        p99_abs=float(np.percentile(np.abs(a - b)[m], 99)), lpips=r.get('lpips'),
                                        delta_e=(r.get('delta_e') or {}).get('mean'), status=r['status'])
            print(f"{sid} {cam:10s} rel {rows[f'{sid}/{cam}']['rel_abs_diff']:.3%} lpips {r.get('lpips') or float('nan'):.4f} "
                  f"dE {rows[f'{sid}/{cam}']['delta_e'] or float('nan'):.2f}", flush=True)
    out = dest / f'compare_win_linux_{env.lower()}.json'
    out.write_text(json.dumps(rows, indent=1) + '\n')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=('sync', 'render', 'compare', 'status', 'kill'))
    ap.add_argument('arg', nargs='?', default='')
    ap.add_argument('--env', default='Studio', choices=('Studio', 'VHStudio'))
    ap.add_argument('--tag', default=''); ap.add_argument('--timeout', type=float, default=7200.)
    ap.add_argument('--no-sync', action='store_true')
    a = ap.parse_args()
    ids = [x for x in a.arg.split(',') if x]
    if a.cmd == 'sync':
        sync(); return
    if a.cmd in ('status', 'kill'):
        print(status(a.arg) if a.cmd == 'status' else kill(a.arg)); return
    if a.cmd == 'compare':
        compare(ids, a.env); return
    tag = a.tag or f"render_{a.env.lower()}_{time.strftime('%Y%m%d-%H%M%S')}"
    if not a.no_sync:
        sync()
    batch = push(ids, a.env, tag)
    start(tag, ['render_batch.py', batch], a.timeout)
    code = wait(tag, a.timeout)
    print(f'{tag}: launch exit {code}', flush=True)
    print(status(tag))
    pull(ids, a.env, tag)


if __name__ == '__main__':
    main()
