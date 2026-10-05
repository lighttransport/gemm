"""Install a checksum-verified, project-local Blender Cycles binary."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

VERSION = '4.5.14'
ARCHIVE_SHA256 = '9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3'


def install(root):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    name = f'blender-{VERSION}-linux-x64'
    target = root/name
    filename = name+'.tar.xz'
    base = 'https://download.blender.org/release/Blender4.5/'
    expected = ARCHIVE_SHA256
    manifest=target/'vhuman-installation.json'
    if manifest.is_file() and (target/'blender').is_file():
        saved=json.loads(manifest.read_text())
        if saved.get('archive_sha256')!=expected:
            raise ValueError('existing Blender installation checksum changed')
        if saved.get('binary_sha256') and saved['binary_sha256']!=file_sha256(target/'blender'):
            raise ValueError('existing Blender binary checksum changed')
        return target/'blender'
    archive=root/(filename+'.partial')
    subprocess.run(['curl','--silent','--show-error','--fail','--location','--connect-timeout','20',
                    '--max-time','1800',base+filename,'--output',str(archive)],check=True)
    digest=hashlib.sha256()
    with archive.open('rb') as downloaded:
        while chunk:=downloaded.read(1<<20):digest.update(chunk)
    if digest.hexdigest()!=expected:
        archive.unlink();raise ValueError('Blender archive checksum mismatch')
    if shutil.disk_usage(root).free<5<<30:
        raise ValueError('insufficient disk headroom for Blender extraction')
    with tarfile.open(archive,'r:xz') as tar:
        for member in tar.getmembers():
            if not (root/member.name).resolve().is_relative_to(root.resolve()):
                raise ValueError('unsafe Blender archive member')
        tar.extractall(root,filter='data')
    manifest.write_text(json.dumps(dict(version=VERSION,url=base+filename,archive_sha256=expected,
                                       binary_sha256=file_sha256(target/'blender')),indent=2))
    archive.unlink()
    return target/'blender'


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1<<20):digest.update(chunk)
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('/mnt/disk01/data/vhuman/tools'))
    args=ap.parse_args()
    print(install(args.out))


if __name__=='__main__':main()
