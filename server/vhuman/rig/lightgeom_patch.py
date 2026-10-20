"""Apply/verify the pinned, repository-shipped LightGeom CPU source patch.

Keeps the optional checkout reproducible without requiring an unpublished
upstream commit. Refuses unrelated tracked edits and verifies every patched
file, including new files that an ordinary git diff would omit.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PATCHES=Path(__file__).with_name('patches')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt():
    record=json.loads((PATCHES/'lightgeom-full-face.json').read_text())
    if record.get('schema')!='vhuman.lightgeom_patch.v1':
        raise ValueError('unsupported LightGeom patch receipt')
    if digest(PATCHES/'lightgeom-full-face.patch')!=record['patch_sha256']:
        raise ValueError('LightGeom source patch hash mismatch')
    return record


def verify(checkout):
    checkout=Path(checkout).resolve();record=receipt()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    if head!=record['base_revision']:
        return False
    modified=set(subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=checkout,text=True).splitlines())
    if modified-set(record['files']):
        return False
    for name,expected in record['files'].items():
        path=(checkout/name).resolve()
        if checkout not in path.parents or not path.is_file() or digest(path)!=expected:
            return False
    return True


def apply(checkout):
    checkout=Path(checkout).resolve();record=receipt()
    if verify(checkout):
        return record
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=checkout,text=True)
    if head!=record['base_revision'] or dirty.strip():
        raise ValueError('LightGeom must be clean at the pinned base or match the complete CPU patch; existing edits were preserved')
    patch=PATCHES/'lightgeom-full-face.patch'
    subprocess.run(['git','apply','--check',str(patch.resolve())],cwd=checkout,check=True)
    subprocess.run(['git','apply',str(patch.resolve())],cwd=checkout,check=True)
    if not verify(checkout):
        raise ValueError('applied LightGeom source files failed hash verification')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('checkout',type=Path)
    args=parser.parse_args()
    if args.verify:
        return 0 if verify(args.checkout) else 1
    result=apply(args.checkout)
    print('LightGeom CPU source patch '+result['patch_sha256'])
    return 0


if __name__=='__main__':
    raise SystemExit(main())
