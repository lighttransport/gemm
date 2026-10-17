"""Shared GEMM Python storage; branch paths remain compatibility aliases."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def environments_root():
    default = Path('/mnt/nvme01/venvs/gemm') if Path('/mnt/nvme01').is_mount() else ROOT.parent / '.venvs'
    return Path(os.environ.get('GEMM_VENVS_ROOT', default)).expanduser()


def cache_path():
    default = Path('/mnt/nvme01/.cache/uv') if Path('/mnt/nvme01').is_mount() else ROOT.parent / '.uv-cache'
    return Path(os.environ.get('UV_CACHE_DIR', default)).expanduser()


def environment_path(legacy, *, link=False):
    legacy = Path(legacy).absolute()
    # Resolve parent aliases (~/work/gemm) without losing the legacy key itself.
    legacy = legacy.parent.resolve() / legacy.name
    if legacy.is_symlink():
        return legacy.resolve()
    # Existing installations stay usable until explicitly migrated and checked.
    if legacy.exists():
        return legacy
    shared = environments_root()
    registry = shared / 'registry.json'
    entries = json.loads(registry.read_text()) if registry.exists() else {}
    if str(legacy) in entries:
        target = Path(entries[str(legacy)]['destination'])
    else:
        try:
            role = str(legacy.relative_to(ROOT.parent))
        except ValueError:
            role = str(legacy)
        role = re.sub(r'[^a-z0-9-]+', '-', role.lower()).strip('-')
        fingerprint = hashlib.sha256(str(legacy).encode())
        # Separate fresh profiles when a project's lock changes. Migrated
        # installations use their recorded full dependency/ABI fingerprint.
        for parent in (legacy.parent, *legacy.parents):
            if not parent.is_relative_to(ROOT):
                break
            lock_files = [parent / name for name in ('uv.lock', 'pyproject.toml')]
            found = [p for p in lock_files if p.is_file()]
            if found:
                for p in found:
                    fingerprint.update(p.read_bytes())
                break
        target = shared / (role + '-' + fingerprint.hexdigest()[:12])
    if link:
        shared.mkdir(parents=True, exist_ok=True)
        legacy.parent.mkdir(parents=True, exist_ok=True)
        try:
            legacy.symlink_to(target, target_is_directory=True)
        except FileExistsError:
            return environment_path(legacy)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('cache', 'venv'))
    parser.add_argument('legacy', nargs='?')
    parser.add_argument('--link', action='store_true', help='create a compatibility alias for a new environment')
    args = parser.parse_args()
    if args.kind == 'venv' and not args.legacy:
        parser.error('venv requires the legacy environment path')
    print(cache_path() if args.kind == 'cache' else environment_path(args.legacy, link=args.link))


if __name__ == '__main__':
    main()
